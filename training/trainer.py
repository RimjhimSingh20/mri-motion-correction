import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from metrics.image_quality import MetricTracker


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup → cosine annealing to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


class Trainer:
    """
    Training loop with:
      - Mixed-precision (AMP) via torch.cuda.amp
      - Gradient accumulation
      - Cosine LR schedule with linear warmup
      - Best-model checkpointing (ranked by val SSIM)
      - Optional TensorBoard logging
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metric_tracker: MetricTracker,
        cfg: dict,
        output_dir: str,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metric_tracker = metric_tracker
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        train_cfg = cfg["training"]
        self.use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip: float = train_cfg.get("grad_clip", 1.0)
        self.accum_steps: int = train_cfg.get("gradient_accumulation_steps", 1)

        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 10),
            total_epochs=train_cfg["epochs"],
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )

        self.writer = None
        if cfg.get("logging", {}).get("use_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError:
                pass

        self.best_val_ssim = -float("inf")
        self.history: Dict = {
            "train_loss": [],
            "val_loss": [],
            "val_ssim": [],
            "val_psnr": [],
            "val_nrmse": [],
        }

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            inp = batch["input"].to(self.device, non_blocking=True)
            tgt = batch["target"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                pred = self.model(inp)
                loss = self.criterion(pred, tgt) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        self.metric_tracker.reset()

        for batch in self.val_loader:
            inp = batch["input"].to(self.device, non_blocking=True)
            tgt = batch["target"].to(self.device, non_blocking=True)
            pred = self.model(inp)
            total_loss += self.criterion(pred, tgt).item()
            self.metric_tracker.update(pred, tgt)

        metrics = self.metric_tracker.compute()
        metrics["loss"] = total_loss / max(1, len(self.val_loader))
        return metrics

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "metrics": metrics,
            "cfg": self.cfg,
        }
        path = self.output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.output_dir / "checkpoints" / "best.pt")

    # ------------------------------------------------------------------
    def fit(self) -> Dict:
        cfg = self.cfg
        n_epochs = cfg["training"]["epochs"]
        log_cfg = cfg.get("logging", {})
        log_interval = log_cfg.get("log_interval", 10)
        val_interval = log_cfg.get("val_interval", 1)
        ckpt_interval = log_cfg.get("checkpoint_interval", 10)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()
            self.history["train_loss"].append(train_loss)

            if epoch % val_interval == 0:
                val_metrics = self._val_epoch()
                val_ssim = val_metrics.get("ssim", {}).get("mean", 0.0)
                val_psnr = val_metrics.get("psnr", {}).get("mean", 0.0)
                val_nrmse = val_metrics.get("nrmse", {}).get("mean", 999.0)

                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_ssim"].append(val_ssim)
                self.history["val_psnr"].append(val_psnr)
                self.history["val_nrmse"].append(val_nrmse)

                is_best = val_ssim > self.best_val_ssim
                if is_best:
                    self.best_val_ssim = val_ssim

                if epoch % ckpt_interval == 0 or is_best:
                    self._save_checkpoint(epoch, val_metrics, is_best)

                if self.writer:
                    self.writer.add_scalar("Loss/train", train_loss, epoch)
                    self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                    self.writer.add_scalar("Metrics/SSIM", val_ssim, epoch)
                    self.writer.add_scalar("Metrics/PSNR", val_psnr, epoch)
                    self.writer.add_scalar("Metrics/NRMSE", val_nrmse, epoch)
                    self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

                elapsed = time.time() - t0
                if epoch % log_interval == 0 or is_best:
                    print(
                        f"Epoch {epoch:4d}/{n_epochs} | "
                        f"loss {train_loss:.4f} | "
                        f"val_loss {val_metrics['loss']:.4f} | "
                        f"SSIM {val_ssim:.4f} | "
                        f"PSNR {val_psnr:.2f} dB | "
                        f"NRMSE {val_nrmse:.4f} | "
                        f"LR {self.scheduler.get_last_lr()[0]:.2e} | "
                        f"{elapsed:.1f}s"
                        + (" [BEST]" if is_best else "")
                    )

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        if self.writer:
            self.writer.close()

        print(f"\nTraining complete — best val SSIM: {self.best_val_ssim:.4f}")
        return self.history
