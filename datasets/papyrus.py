from pathlib import Path
from typing import List, Tuple
import torch

import numpy as np
import MinkowskiEngine as EM
from synthetic_pages.nrrd_file import Nrrd
from volume_pointcloud_conversion import dense_volume_with_labels_to_points


class PapyrusExample:
    def __init__(self, volume_nrrd: Nrrd, labelmap_nrrd: Nrrd):
        # Convert to point cloud format 
        coords, features, labels = dense_volume_with_labels_to_points(
            volume_nrrd, 
            labelmap_nrrd,
            min_density=0.1,  # Tune this threshold based on your data
            subsample_factor=4 # Can adjust based on memory constraints
        )
        
        # Create the required data format for Mask3D training
        self.coordinates = coords  # Coordinates of points
        self.features = features   # Density values at points
        self.labels = labels       # Instance labels at points
        self.num_instances = len(np.unique(labels[labels > 0]))
        self.point2segment = None  # Optional, for segment-based training
        
        # Create target format Mask3D expects
        self.target = {
            "labels": labels,  # Instance IDs
            "masks": torch.zeros(self.num_instances, len(coords)),  # Binary masks for each instance
        }
        
        # Fill in target masks
        for i, inst_id in enumerate(np.unique(labels[labels > 0])):
            self.target["masks"][i] = (labels == inst_id)
            
class PapyrusDataset(torch.utils.data.Dataset):
    def __init__(self, nrrd_paths: List[Tuple[Path, Path]]):
        """
        Args:
            nrrd_paths: List of (volume_path, labelmap_path) pairs
        """
        self.examples = []
        for vol_path, label_path in nrrd_paths:
            volume = Nrrd.from_file(vol_path)
            labelmap = Nrrd.from_file(label_path)
            self.examples.append(PapyrusExample(volume, labelmap))
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            ME.SparseTensor(
                features=example.features,
                coordinates=example.coordinates,
            ),
            example.target,
            str(idx)  # Filename/identifier
        )