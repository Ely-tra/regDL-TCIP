import os

import torch
import torch.distributed as dist

from module.models.registry import resolve_model_class


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    if world_size <= 1:
        return False, 0, 0, 1

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    if "MASTER_ADDR" not in os.environ:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        if "[" in nodelist and "]" in nodelist:
            prefix = nodelist.split("[", 1)[0]
            inside = nodelist.split("[", 1)[1].split("]", 1)[0]
            first = inside.split(",", 1)[0].split("-", 1)[0]
            master_addr = f"{prefix}{first}"
        else:
            master_addr = nodelist.split(",", 1)[0]
        os.environ["MASTER_ADDR"] = master_addr
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("CUDA device requested but none are visible.")
    if local_rank >= visible_gpus:
        local_rank = 0
        os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def compute_x_y_stats(loader, stats_device, use_dist: bool):
    x_sum = None
    x_sumsq = None
    x_count = None
    y_sum = None
    y_sumsq = None
    y_count = None

    for batch in loader:
        x = batch["fields"]
        y = batch["target_fields"]
        if x_sum is None:
            num_vars = x.shape[2]
            x_sum = torch.zeros(num_vars, dtype=torch.float64, device=stats_device)
            x_sumsq = torch.zeros(num_vars, dtype=torch.float64, device=stats_device)
            y_sum = torch.zeros(num_vars, dtype=torch.float64, device=stats_device)
            y_sumsq = torch.zeros(num_vars, dtype=torch.float64, device=stats_device)
            x_count = torch.zeros(1, dtype=torch.float64, device=stats_device)
            y_count = torch.zeros(1, dtype=torch.float64, device=stats_device)

        x_d = x.double().to(stats_device)
        y_d = y.double().to(stats_device)

        x_sum += x_d.sum(dim=(0, 1, 3, 4))
        x_sumsq += (x_d * x_d).sum(dim=(0, 1, 3, 4))
        x_count += x_d.shape[0] * x_d.shape[1] * x_d.shape[3] * x_d.shape[4]

        y_sum += y_d.sum(dim=(0, 2, 3))
        y_sumsq += (y_d * y_d).sum(dim=(0, 2, 3))
        y_count += y_d.shape[0] * y_d.shape[2] * y_d.shape[3]

    if use_dist:
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sumsq, op=dist.ReduceOp.SUM)
        dist.all_reduce(y_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(y_sumsq, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(y_count, op=dist.ReduceOp.SUM)

    x_mean = (x_sum / x_count).float()
    y_mean = (y_sum / y_count).float()

    x_var = (x_sumsq / x_count) - x_mean.double().pow(2)
    y_var = (y_sumsq / y_count) - y_mean.double().pow(2)

    x_std = torch.sqrt(x_var.clamp_min(0.0)).float()
    y_std = torch.sqrt(y_var.clamp_min(0.0)).float()

    return x_mean.cpu(), x_std.cpu(), y_mean.cpu(), y_std.cpu()


def build_model(args, device, x_mean, x_std, y_mean, y_std, use_dist, local_rank):
    model_cls = resolve_model_class(args)
    model = model_cls(
        num_vars=args.num_vars,
        num_times=args.num_times,
        H=args.height,
        W=args.width,
        num_blocks=args.num_blocks,
        film_zdim=args.film_zdim,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        return_physical=False,
        mlp_expansion_ratio=args.mlp_expansion_ratio,
        hidden_factor=args.hidden_factor,
        channels=args.e_channels,
        stem_channels=args.stem_channels,
    ).to(device)
    if use_dist and device.type == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    return model
