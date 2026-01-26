import json
import pathlib
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from module.models.arch_helpers import build_model, compute_x_y_stats, setup_distributed
from module.training.datasets import TCTimeWindowDataset, load_split_arrays
from module.training.losses import build_weighted_loss
from module.training.masks import extract_bc_rim_from_y, make_rim_mask_like, make_smooth_phi



def _apply_model_config(args, config_path):
    if not config_path:
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in [
        "num_vars",
        "num_times",
        "height",
        "width",
        "num_blocks",
        "film_zdim",
        "e_channels",
        "hidden_factor",
        "mlp_expansion_ratio",
        "stem_channels",
    ]:
        if key in cfg:
            setattr(args, key, cfg[key])


class AfnoTcpTrainer:
    name = "afno_tcp_v1"

    def run(self, args):
        _apply_model_config(args, args.model_config_path)
        assert args.temp_dir, "temp_dir is required"

        use_dist, rank, local_rank, world_size = setup_distributed()
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

        train_data, val_data, _ = load_split_arrays(args.temp_dir)
        train_dataset = TCTimeWindowDataset(train_data, np.arange(train_data.shape[0]), args.step_in)
        val_dataset = None
        if val_data is not None:
            val_dataset = TCTimeWindowDataset(val_data, np.arange(val_data.shape[0]), args.step_in)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                persistent_workers=(args.num_workers > 0),
            )

        train_sampler = None
        val_sampler = None
        if use_dist:
            train_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                persistent_workers=(args.num_workers > 0),
            )
            if val_loader is not None:
                val_sampler = DistributedSampler(
                    val_loader.dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )
                val_loader = DataLoader(
                    val_loader.dataset,
                    batch_size=args.batch_size,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    persistent_workers=(args.num_workers > 0),
                )

        stats_device = device if use_dist and device.type == "cuda" else torch.device("cpu")
        x_mean, x_std, y_mean, y_std = compute_x_y_stats(train_loader, stats_device, use_dist)
        print("y_std min/max:", y_std.min().item(), y_std.max().item(), flush=True)
        print("y_mean min/max:", y_mean.min().item(), y_mean.max().item(), flush=True)
        print("x_std min/max:", x_std.min().item(), x_std.max().item(), flush=True)
        print("x_mean min/max:", x_mean.min().item(), x_mean.max().item(), flush=True)

        model = build_model(
            args,
            device,
            x_mean,
            x_std,
            y_mean,
            y_std,
            use_dist,
            local_rank,
        )
        if args.init_model_path:
            state = torch.load(args.init_model_path, map_location="cpu")
            base_model = model.module if use_dist else model
            base_model.load_state_dict(state, strict=False)

        phi = make_smooth_phi(
            H=args.height,
            W=args.width,
            rim=args.rim,
            device=device,
            dtype=torch.float32,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=7,
            min_lr=1e-10,
            verbose=True,
        )
        loss_fn = build_weighted_loss(args)

        checkpoint_dir = pathlib.Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / args.checkpoint_name
        best_val_loss = float("inf")

        if not use_dist or rank == 0:
            print(f"Using device: {device}", flush=True)
        for epoch in range(args.num_epochs):
            epoch_start = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                x = batch["fields"].to(device)
                y = batch["target_fields"].to(device)

                base_model = model.module if use_dist else model
                if base_model.x_scaler is not None:
                    x = base_model.x_scaler.norm(x)

                y = base_model.y_scaler.norm(y)

                B_fill = extract_bc_rim_from_y(y, rim=args.rim)
                bc_mask = make_rim_mask_like(y, rim=args.rim)
                bc_in = torch.cat([B_fill, bc_mask], dim=1)
                y_free = model(x, bc_in)
                y_pred = phi * y_free + (1.0 - phi) * B_fill

                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * x.shape[0]
                train_count += x.shape[0]

            if use_dist:
                loss_tensor = torch.tensor([train_loss_sum, train_count], device=device, dtype=torch.float64)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = (loss_tensor[0] / loss_tensor[1].clamp_min(1.0)).item()
            else:
                train_loss = train_loss_sum / max(1, train_count)

            val_loss = train_loss
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)

                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["fields"].to(device)
                        y = batch["target_fields"].to(device)
                        base_model = model.module if use_dist else model
                        if base_model.x_scaler is not None:
                            x = base_model.x_scaler.norm(x)
                        y = base_model.y_scaler.norm(y)

                        B_fill = extract_bc_rim_from_y(y, rim=args.rim)
                        bc_mask = make_rim_mask_like(y, rim=args.rim)
                        bc_in = torch.cat([B_fill, bc_mask], dim=1)
                        y_free = model(x, bc_in)
                        y_pred = phi * y_free + (1.0 - phi) * B_fill

                        loss = loss_fn(y_pred, y)

                        val_loss_sum += loss.item() * x.shape[0]
                        val_count += x.shape[0]

                if use_dist:
                    loss_tensor = torch.tensor([val_loss_sum, val_count], device=device, dtype=torch.float64)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = (loss_tensor[0] / loss_tensor[1].clamp_min(1.0)).item()
                else:
                    val_loss = val_loss_sum / max(1, val_count)
            epoch_time = time.perf_counter() - epoch_start
            if not use_dist or rank == 0:
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"time={epoch_time:.2f}s",
                    flush=True,
                )
            scheduler.step(val_loss)
            if (not use_dist or rank == 0) and val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model.module.state_dict() if use_dist else model.state_dict()
                torch.save(state_dict, checkpoint_path)
                print(f"  Saved best checkpoint to {checkpoint_path} (val_loss={val_loss:.6f})", flush=True)
