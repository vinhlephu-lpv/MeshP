"""Keeps at most ONE heavy AI model on the GPU at a time (4 GB VRAM budget)."""
from __future__ import annotations

import gc
import threading
from typing import Any, Callable, Dict, Optional

import torch


class ModelManager:
    def __init__(self) -> None:
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._current_name: Optional[str] = None
        self._current_model: Any = None
        self._lock = threading.RLock()

    def register(self, name: str, loader: Callable[[], Any]) -> None:
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        """Return model `name`, unloading whatever else is on the GPU first."""
        with self._lock:
            if self._current_name == name:
                return self._current_model
            self.unload()
            self._current_model = self._loaders[name]()
            self._current_name = name
            return self._current_model

    def unload(self) -> None:
        with self._lock:
            self._current_model = None
            self._current_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def vram_report() -> str:
        if not torch.cuda.is_available():
            return "CPU mode"
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return f"VRAM allocated {alloc:.2f} GB | peak {peak:.2f} GB"


manager = ModelManager()
