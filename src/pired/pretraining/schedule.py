import math

import torch

LR_SCHEDULES = ("wsd", "cosine", "linear")


def build_optimizer(model, settings, device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": settings.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=settings.lr, betas=settings.betas, eps=settings.eps,
                             fused=device.type == "cuda")


def decay_start_step(settings):
    if settings.lr_schedule == "wsd":
        return max(settings.warmup_steps, settings.total_steps - round(settings.decay_fraction * settings.total_steps))
    return settings.warmup_steps


def lr_factor(step, settings):
    if step < settings.warmup_steps:
        return (step + 1) / settings.warmup_steps
    start = decay_start_step(settings)
    if step < start:
        return 1.0
    progress = min(1.0, (step - start) / max(1, settings.total_steps - start))
    floor = settings.min_lr_ratio
    if settings.lr_schedule == "wsd":
        return floor + (1 - floor) * (1 - math.sqrt(progress))
    if settings.lr_schedule == "cosine":
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return floor + (1 - floor) * (1 - progress)


def build_scheduler(optimizer, settings):
    if settings.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr_schedule {settings.lr_schedule!r}; use one of {LR_SCHEDULES}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, settings))
