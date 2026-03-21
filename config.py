#!/usr/bin/env python3
"""
Configurazione centralizzata per training e monitoraggio.
Evita hardcoding di percorsi e directory.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configurazione condivisa training-monitor."""

    # Paths
    base_model: str = "./models/SmolLM-135M-Instruct"
    dataset_path: str = "./datasets/italian_unified/train.jsonl"

    # Output directories
    output_dir: str = "./output_qlora_optimized"
    tensorboard_log_dir: str = "./logs_qlora_optimized"

    # Log files
    training_log: str = "train_qlora_optimized.log"
    monitor_log: str = "monitor_training.log"

    # Training hyperparameters
    max_samples: Optional[int] = None  # None = all, None = full training
    max_seq_length: int = 256
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    max_grad_norm: float = 0.0

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list] = None

    # Monitoring
    logging_steps: int = 10
    save_steps: int = 500
    tensorboard_port: int = 6006

    def __post_init__(self):
        """Inizializza directory se non esistono."""
        for attr in ["output_dir", "tensorboard_log_dir"]:
            path = Path(getattr(self, attr))
            path.mkdir(parents=True, exist_ok=True)
            path.mkdir(exist_ok=True)

    def get_config_dict(self) -> dict:
        """Restituisce come dict."""
        return {
            "base_model": self.base_model,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "training_log": self.training_log,
            "monitor_log": self.monitor_log,
            "max_samples": self.max_samples,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "tensorboard_port": self.tensorboard_port,
        }

    @classmethod
    def from_dict(cls, d):
        """Crea istanza da dizionario."""
        return TrainingConfig(
            base_model=d["base_model"],
            dataset_path=d["dataset_path"],
            output_dir=d["output_dir"],
            tensorboard_log_dir=d["tensorboard_log_dir"],
            training_log=d["training_log"],
            monitor_log=d["monitor_log"],
            max_samples=d.get("max_samples"),
            max_seq_length=d.get("max_seq_length"),
            batch_size=d.get("batch_size"),
            gradient_accumulation_steps=d.get("gradient_accumulation_steps"),
            num_epochs=d.get("num_epochs"),
            learning_rate=d.get("learning_rate"),
            weight_decay=d.get("weight_decay"),
            warmup_ratio=d.get("warmup_ratio"),
            max_grad_norm=d.get("max_grad_norm"),
            lora_r=d.get("lora_r"),
            lora_alpha=d.get("lora_alpha"),
            lora_dropout=d.get("lora_dropout"),
            target_modules=d.get("target_modules"),
            logging_steps=d.get("logging_steps"),
            save_steps=d.get("save_steps"),
            tensorboard_port=d.get("tensorboard_port"),
        )

    @classmethod
    def full_training(cls):
        """Config per training completo (55k campioni)."""
        config = cls()
        config.max_samples = 55000
        config.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]  # Full attention
        return config

    @classmethod
    def test_training(cls):
        """Config per test (1000 campioni)."""
        config = cls()
        config.max_samples = 1000
        return config

    @classmethod
    def quick_test(cls):
        """Config per test veloce (100 campioni)."""
        config = cls()
        config.max_samples = 100
        config.num_epochs = 1
        return config
