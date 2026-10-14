"""
Main entry point for HOI mesh training.
Refactored version with modular components.
"""
import tyro

from optimization.config.config import Config
from optimization.pipelines.trainer import Runner

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    runner = Runner(cfg)
    runner.train()
