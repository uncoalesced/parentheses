# Engineered by uncoalesced
"""
Dual-Stream Curriculum Interleaver & Dual-Validation Gate for Session 3 (Steps 4001 -> 6000)
Blends primary (`processed_240m_apache`) and secondary (`processed_v5`) NVMe memmaps
across the 16 gradient accumulation micro-steps with zero extra VRAM footprint.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from scripts.license_guard import verify_apache_manifest


class DualCurriculumLoader:
    def __init__(
        self,
        primary_dir: Path,
        secondary_dir: Optional[Path] = None,
        seq_len: int = 4096,
    ) -> None:
        # Enforce 100% Apache-2.0 license guard on both directories at startup
        verify_apache_manifest(str(primary_dir))
        self.seq_len = seq_len
        self.prim_train = np.memmap(primary_dir / "train.bin", dtype=np.uint16, mode="r")
        self.prim_val = np.memmap(primary_dir / "val.bin", dtype=np.uint16, mode="r") if (primary_dir / "val.bin").exists() else None
        self.prim_bounds = np.load(primary_dir / "train_boundaries.npy", mmap_mode="r") if (primary_dir / "train_boundaries.npy").exists() else None

        self.sec_train = None
        self.sec_val = None
        self.sec_bounds = None

        if secondary_dir is not None:
            verify_apache_manifest(str(secondary_dir))
            self.sec_train = np.memmap(secondary_dir / "train.bin", dtype=np.uint16, mode="r")
            sec_val_path = secondary_dir / "val.bin"
            if sec_val_path.exists():
                self.sec_val = np.memmap(sec_val_path, dtype=np.uint16, mode="r")
            sec_bounds_path = secondary_dir / "train_boundaries.npy"
            if sec_bounds_path.exists():
                self.sec_bounds = np.load(sec_bounds_path, mmap_mode="r")

    def use_secondary(self, step: int, micro_step: int, grad_accum: int = 16) -> bool:
        """
        Deterministically interleaves primary and secondary shards across micro-steps:
        - Steps 4001..4250 (Phase 3A): 8/16 (50.0%) secondary (`processed_v5`)
        - Steps 4251..6000 (Phase 3B): 10/16 (62.5%) secondary (`processed_v5`)
        """
        if self.sec_train is None:
            return False
        if step <= 4250:
            # 8 out of 16 micro-steps pull from processed_v5 (50.0%)
            return (micro_step % 2) == 1
        # Exactly 10 out of 16 micro-steps pull from processed_v5 (62.5%) (indices: 0,1, 3,4, 6,7, 9,10, 12,13)
        return (micro_step % 3) != 2 and micro_step < 15

    def sample_micro_batch(
        self,
        step: int,
        micro_step: int,
        batch_size: int,
        device: str,
        boundary_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from_sec = self.use_secondary(step, micro_step)
        data = self.sec_train if from_sec else self.prim_train
        bounds = self.sec_bounds if from_sec else self.prim_bounds

        max_idx = len(data) - self.seq_len - 1
        ix = torch.randint(0, max_idx, (batch_size,))

        x_np = np.stack([data[i : i + self.seq_len].astype(np.int64) for i in ix.tolist()])
        y_np = np.stack([data[i + 1 : i + 1 + self.seq_len].astype(np.int64) for i in ix.tolist()])

        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)

        if bounds is not None and boundary_fn is not None:
            reset_mask = boundary_fn(bounds, ix.numpy(), self.seq_len)
            if not isinstance(reset_mask, torch.Tensor):
                reset_mask = torch.tensor(reset_mask, dtype=torch.uint8)
            reset_mask = reset_mask.to(device)
        else:
            reset_mask = torch.ones((batch_size, self.seq_len), dtype=torch.uint8, device=device)

        # Hard EOT (token_id == 0) recurrent state isolation
        eot_mask = (x != 0).to(torch.uint8)
        prev_eot = torch.ones_like(eot_mask)
        prev_eot[:, 1:] = (x[:, :-1] != 0).to(torch.uint8)
        reset_mask = reset_mask & eot_mask & prev_eot

        return x, y, reset_mask
