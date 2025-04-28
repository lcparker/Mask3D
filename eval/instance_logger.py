import wandb
import torch
import numpy as np
import colorsys
from PIL import Image
from typing import List


class WandbInstanceImageLogger:
    """Log *(image | prediction | ground‑truth)* strips where **every integer id → a
    maximally‑distinct colour**.

    *   Black (id 0) is reserved for background.
    *   The colour wheel uses the *golden‑angle hop* so consecutive ids are far
        apart perceptually, avoiding the early "all orange" problem.
    *   The palette expands lazily to fit the highest label value seen.
    """

    GOLDEN_RATIO_CONJUGATE: float = 0.618033988749895  # 1/φ

    def __init__(
        self,
        *,
        prefix: str = "val/output_slices",
        separator_px: int = 4,
        base_palette_size: int = 32,  # start small; grows on demand
    ) -> None:
        self.prefix: str = prefix
        self.separator_px: int = separator_px
        self._palette: List[int] = self._make_distinct_palette(base_palette_size)

    def log(
        self,
        volume: torch.Tensor,
        predicted_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        step: int,
        *,
        n_slices: int = 8,
        log_to_wandb: bool = True,
    ) -> List[wandb.Image]:
        if not (
            volume.shape == predicted_labels.shape == ground_truth_labels.shape
        ):
            raise ValueError(
                "volume, predicted_labels, and ground_truth_labels must share the same D×H×W geometry."
            )

        max_id_tensor = torch.stack(
            [predicted_labels.max(), ground_truth_labels.max()]
        ).max()
        self._ensure_palette_size(int(max_id_tensor.item()))

        depth = int(volume.shape[1])
        slice_indices = torch.linspace(0, depth - 1, n_slices).long()
        images: List[wandb.Image] = []

        for z in slice_indices:
            z_int: int = int(z)
            # left panel: grayscale input
            slice_np = volume[:, z_int].detach().cpu().numpy()
            slice_uint8 = self.__to_uint8(slice_np)
            grayscale_rgb = np.stack([slice_uint8] * 3, axis=-1)

            # middle / right panels
            pred_rgb = self._label_to_rgb(
                predicted_labels[:, z_int].detach().cpu().numpy()
            )
            gt_rgb = self._label_to_rgb(
                ground_truth_labels[:, z_int].detach().cpu().numpy()
            )

            strip = self.__hstack_with_separator(
                [grayscale_rgb, pred_rgb, gt_rgb], self.separator_px
            )
            images.append(
                wandb.Image(strip, caption=f"step={step}, slice={z_int}")
            )

        if log_to_wandb:
            wandb.log({self.prefix: images}, step=step)
        return images

    def _make_distinct_palette(self, n_colors: int) -> List[int]:
        """Return a flat RGB palette list (id 0 → black) with *n_colors* hues.

        We step around the hue circle by the golden‑ratio conjugate so that early
        labels are maximally separated – ids 1‑16 each fall in a different
        perceptual region (red, cyan, lime, magenta …).  Saturation is kept at
        0.85 and value at 1.0 for high contrast without neon glare.
        """
        palette: List[int] = [0, 0, 0]  # reserve slot‑0 black
        hue: float = 0.0
        for _ in range(n_colors):
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            palette.extend([int(r * 255), int(g * 255), int(b * 255)])
            hue = (hue + self.GOLDEN_RATIO_CONJUGATE) % 1.0
        return palette

    def _ensure_palette_size(self, required_max_id: int) -> None:
        current_capacity = len(self._palette) // 3 - 1  # exclude id‑0
        if required_max_id > current_capacity:
            new_capacity = required_max_id + 16  # grow with slack
            self._palette = self._make_distinct_palette(new_capacity)

    def _label_to_rgb(self, label_hw: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(label_hw.astype(np.uint8), mode="P")
        pil_img.putpalette(self._palette)
        return np.asarray(pil_img.convert("RGB"))

    @staticmethod
    def __to_uint8(image_2d: np.ndarray) -> np.ndarray:
        if image_2d.dtype == np.uint8:
            return image_2d
        img = image_2d.astype(np.float32)
        img -= img.min()
        max_val = img.max()
        if max_val > 0:
            img /= max_val
        return (img * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def __hstack_with_separator(
        panels: List[np.ndarray], separator_px: int
    ) -> np.ndarray:
        if separator_px <= 0 or len(panels) == 1:
            return np.concatenate(panels, axis=1)
        height = panels[0].shape[0]
        white = 255 * np.ones((height, separator_px, 3), dtype=np.uint8)
        parts: List[np.ndarray] = []
        for i, panel in enumerate(panels):
            parts.append(panel)
            if i < len(panels) - 1:
                parts.append(white)
        return np.concatenate(parts, axis=1)
