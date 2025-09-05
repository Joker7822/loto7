
# early_stopping.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EarlyStopConfig:
    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "min"
    restore_best: bool = True

class EarlyStopping:
    def __init__(self, cfg: EarlyStopConfig):
        assert cfg.mode in ("min", "max")
        self.cfg = cfg
        self.best = None
        self.best_state = None
        self.counter = 0
        self.stopped = False

    def _is_better(self, score: float) -> bool:
        if self.best is None:
            return True
        if self.cfg.mode == "min":
            return (self.best - score) > self.cfg.min_delta
        else:
            return (score - self.best) > self.cfg.min_delta

    def step(self, score: float, model=None) -> bool:
        if self._is_better(score):
            self.best = score
            self.counter = 0
            if self.cfg.restore_best and model is not None:
                try:
                    self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                except Exception:
                    self.best_state = None
        else:
            self.counter += 1
            if self.counter >= self.cfg.patience:
                self.stopped = True
                if self.cfg.restore_best and model is not None and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
        return False
