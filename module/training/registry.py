from .pipelines import AfnoTcpTrainer

_TRAINERS = {
    "afno_tcp_v1": AfnoTcpTrainer(),
}


def available_trainers() -> tuple[str, ...]:
    return tuple(_TRAINERS.keys())


def resolve_trainer(args):
    name = getattr(args, "trainer", None) or "afno_tcp_v1"
    if name not in _TRAINERS:
        raise ValueError(f"Unknown trainer: {name}")
    return _TRAINERS[name]


def run_training(args):
    trainer = resolve_trainer(args)
    trainer.run(args)
