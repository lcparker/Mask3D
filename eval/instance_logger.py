import wandb
import torch
import numpy as np
import colorsys
from PIL import Image

class WandbInstanceImageLogger:
    def __init__(self, num_instances: int = 100, prefix="val/output_slices"):
        self.prefix = prefix
        self.palette = self.make_hsv_palette(num_instances)


    def make_hsv_palette(self, n_colors: int) -> list[int]:
        """
        Generate a flat RGB palette list for PIL of length 3*n_colors,
        by evenly sampling the HSV hue dimension.
        """
        # hues evenly spaced in [0,1)
        hues = np.linspace(0, 1, n_colors, endpoint=False)
        palette = [0,0,0] # black for 0
        for h in hues:
            # full saturation and value
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            palette.extend([
                int(r * 255),
                int(g * 255),
                int(b * 255),
            ])
        return palette


    def label_to_rgb_pil(self, label_2d: np.ndarray, palette: list[int]) -> np.ndarray:
        """
        Convert H×W int label array (0…M‑1) to an H×W×3 RGB image via a PIL palette.
        """
        pil = Image.fromarray(label_2d.astype(np.uint8), mode="P")
        pil.putpalette(palette)
        return np.array(pil.convert("RGB"))


    def log(self, 
        volume: torch.Tensor,
        predicted_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        step: int,
        n_slices: int = 8,
        batch_index: int = 0):
        assert len(volume.shape) == len(predicted_labels.shape) == len(ground_truth_labels.shape) == 5
        assert volume.shape[2:] == predicted_labels.shape[2:]
        B, M, D, H, W = volume.shape
        slice_idxs = torch.linspace(0, D - 1, n_slices).long()
        images = []
        for m in range(M):
            for idx in slice_idxs:
                image = wandb.Image(
                    volume[batch_index, m, idx].detach().cpu().numpy(),
                    masks = {
                        "predictions": {
                            "mask_data": predicted_labels[batch_index, m, idx].detach().cpu().numpy(),
                        },
                        "ground_truth": {
                            "mask_data": ground_truth_labels[batch_index, m, idx].detach().cpu().numpy(),
                        },
                    },
                    caption=f"batch={batch_index}, ch={m}, slice={int(idx)}"
                )
                images.append(image)
        wandb.log({self.prefix: images}, step=step)
