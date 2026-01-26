from module.models.blocks import BLOCK_REGISTRY


class AfnoV1Builder:
    name = "afno_v1"
    yaml_name = "AFNO_v1.yaml"

    def build_config(self, args) -> dict:
        block_type = "CondAFNO2DBlock"
        if block_type not in BLOCK_REGISTRY:
            raise ValueError(f"Missing block definition: {block_type}")

        return {
            "architecture": "AFNO_v1",
            "inputs": {
                "num_vars": args.num_vars,
                "num_times": args.num_times,
                "height": args.height,
                "width": args.width,
            },
            "stem": {
                "type": "Conv2d",
                "stem_channels": args.stem_channels,
            },
            "bc_encoder": {
                "type": "BCEncoder",
                "in_channels": args.num_vars + 1,
                "z_dim": args.film_zdim,
            },
            "trunk": {
                "block_type": block_type,
                "num_blocks": args.num_blocks,
                "channels": args.e_channels,
                "hidden_factor": args.hidden_factor,
                "mlp_expansion_ratio": args.mlp_expansion_ratio,
                "hard_threshold": 0.0,
            },
            "output": {
                "num_vars": args.num_vars,
            },
        }
