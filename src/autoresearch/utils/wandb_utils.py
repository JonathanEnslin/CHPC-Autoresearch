"""Weights & Biases logging utilities."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def initialize_wandb(
    cfg: DictConfig,
    run_dir: Path,
    wandb_run_id: Optional[str] = None,
    additional_config: Optional[Dict] = None,
    group: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Initialize Weights & Biases logging.

    Args:
        cfg: Configuration object
        run_dir: Run directory
        wandb_run_id: Existing run ID for resuming
        additional_config: Additional configuration to log
        group: Optional group name for wandb (overrides cfg.wandb.group)

    Returns:
        Tuple of (wandb_enabled, wandb_run_id)
    """
    wandb_enabled = False

    if cfg.wandb.enabled:
        try:
            import wandb
            from omegaconf import OmegaConf

            # Login to wandb using API key from environment variable
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
            else:
                log.warning("WANDB_API_KEY not found in environment variables")

            # Convert full config to dict (resolve all interpolations)
            wandb_config = OmegaConf.to_container(cfg, resolve=True)

            # Add additional config if provided
            if additional_config:
                wandb_config.update(additional_config)

            # Compute run name
            run_name = (
                cfg.wandb.name
                if cfg.wandb.name
                else cfg.experiment_name
            )

            # Get group (use parameter if provided, otherwise from config)
            wandb_group = group if group is not None else cfg.wandb.get("group")

            # Resume or start new run
            if wandb_run_id is not None:
                wandb.init(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    mode=cfg.wandb.mode,
                    id=wandb_run_id,
                    resume="must",
                    group=wandb_group,
                    config=wandb_config,
                    dir=str(run_dir),
                )
            else:
                wandb.init(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    mode=cfg.wandb.mode,
                    name=run_name,
                    group=wandb_group,
                    tags=cfg.wandb.tags if cfg.wandb.tags else None,
                    notes=cfg.wandb.notes,
                    config=wandb_config,
                    dir=str(run_dir),
                )
                wandb_run_id = wandb.run.id

            wandb_enabled = True
            log.info("Wandb initialized: %s", wandb.run.url)

        except ImportError as e:
            log.warning("Failed to initialize Wandb: %s", e)
            wandb_enabled = False

    return wandb_enabled, wandb_run_id
