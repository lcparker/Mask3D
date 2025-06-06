from pathlib import Path
from typing import NamedTuple
import torch
from torch.nn import functional as F

import numpy as np
from torch.utils.data import IterableDataset, Dataset
from volume_pointcloud_conversion import dense_volume_with_labels_to_points
from synthetic_pages.datasets.synthetic_datamodule_cubes import SyntheticInstanceCubesDataset
from synthetic_pages.datasets.real_scroll_datamodule_cubes import InstanceCubesDataset
from synthetic_pages.datasets.instance_volume_batch import InstanceVolumeBatch

synthetic_cubes = SyntheticInstanceCubesDataset(
  reference_volume_filename= "reference_volume.nrrd",
  reference_label_filename= "reference_labels.nrrd",
  spatial_transform= True,
  layer_dropout=True,
  layer_shuffle= True,
  remove_duplicate_labels=False,
  num_layers_range=(2,6),
  output_volume_size=(48, 48, 48),
)

class PapyrusBatch(NamedTuple):
    coordinates: np.ndarray # (M ,3)
    features: np.ndarray # (M, 1) -- intensity values
    instance_labels: np.ndarray # (M, 3) -- each row is segment mask, instance mask, and segment mask
    name: str
    original_colors: None # Unused
    original_normals: None # Unused
    original_coordinates: torch.Tensor # (M ,3)
    id: None # Unused
    volume_batch: dict

class PapyrusDataset(Dataset):
    def __init__(self, mode="train", label_offset=0):
        super().__init__()
        if not mode in ["train", "validation"]:
            raise ValueError("mode must be either 'train' or 'val'")

        # self.cube_dataset = InstanceCubesDataset(Path("/workspace/code/cubes/") / ("training" if mode == "train" else "validation"))
        # self.cube_dataset = InstanceCubesDataset(Path("/Users/lachlan/Code/cubes") / ("training" if mode == "train" else "validation"))
        self.cube_dataset = synthetic_cubes

        self.label_offset = label_offset
        self.label_info = None
        self.label_info = [
            {
                "name": "Papyrus",
            },
            {
                "name": "DoesNotExist",
            },
        ]
        
    def __len__(self):
        return len(self.cube_dataset)
        
    def __getitem__(self, index):
        batch = self.cube_dataset[index]
        coords, features, point_labels = dense_volume_with_labels_to_points(
            batch.vol, 
            batch.lbl, 
            min_density=batch.vol.max()/2, 
            subsample_factor=2
        )
        
         # Get instance IDs once, excluding background (0)
        num_points = len(point_labels)
        
        segmentation_instance_tensor = torch.zeros((num_points, 3), dtype=torch.long)
        segmentation_instance_tensor[:, 0] = 1  # Segment mask
        segmentation_instance_tensor[:, 1] = point_labels  # Instance masks
        segmentation_instance_tensor[:, 2] = 1  # Segment masks
        
        # Return the updated output
        coords = coords.numpy()
        features = features.numpy()
        segmentation_instance_tensor = segmentation_instance_tensor.numpy()
        return PapyrusBatch(
            coords, 
            features, 
            segmentation_instance_tensor, 
            f"data_{index}", 
            None, 
            None, 
            coords.copy(), 
            None,
            batch)