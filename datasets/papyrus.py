from pathlib import Path
from typing import List, Tuple
import torch

import numpy as np
import MinkowskiEngine as ME
from synthetic_pages.nrrd_file import Nrrd
from volume_pointcloud_conversion import dense_volume_with_labels_to_points
from scrolls_instance_segmentation.data.synthetic_datamodule_cubes import SyntheticInstanceCubesDataset


class PapyrusVolume:
    def __init__(self, volume_nrrd: Nrrd, labelmap_nrrd: Nrrd):
        # Convert to point cloud format 
        coords, features, labels = dense_volume_with_labels_to_points(
            volume_nrrd, 
            labelmap_nrrd,
            min_density=0.1, # Tune this threshold based on your data
            subsample_factor=4 # Can adjust based on memory constraints
        )
        
        # Create the required data format for Mask3D training
        self.coordinates = coords  # Coordinates of points
        self.features = features   # Density values at points
        self.labels = labels       # Instance labels at points
        self.num_instances = len(np.unique(labels[labels > 0]))
        
        # Create target format Mask3D expects
        self.target = {
            "labels": labels,  # Instance IDs # BUG: these are probably semantic class labels
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
        self.papyrus_volumes = []
        for vol_path, label_path in nrrd_paths:
            volume = Nrrd.from_file(vol_path)
            labelmap = Nrrd.from_file(label_path)
            self.papyrus_volumes.append(PapyrusVolume(volume, labelmap))
            
    def __len__(self):
        return len(self.papyrus_volumes)
        
    def __getitem__(self, idx):
        example = self.papyrus_volumes[idx]
        return (
            ME.SparseTensor(
                features=example.features,
                coordinates=example.coordinates,
            ),
            example.target,
            str(idx)  # Filename/identifier
        )
        
class SyntheticPapyrusDataset(torch.utils.data.Dataset):
    def __init__(self, reference_volume_filename, reference_label_filename, 
                 mode="train", spatial_transform=True, layer_dropout=False, 
                 layer_shuffle=True):
        super().__init__()
        self.synthetic_gen = SyntheticInstanceCubesDataset(
            reference_volume_filename=reference_volume_filename,
            reference_label_filename=reference_label_filename,
            spatial_transform=spatial_transform,
            layer_dropout=layer_dropout,
            layer_shuffle=layer_shuffle
        )
        self.label_info = None
        
    def __len__(self):
        return 500  # Same as synthetic dataset
        
    def __getitem__(self, idx):
        batch = self.synthetic_gen._gather_batch()
        coords, features, point_labels = dense_volume_with_labels_to_points(
            batch['vol'], batch['lbl'], min_density=0.1
        )
        
        # Get instance IDs once, excluding background (0)
        instance_ids = torch.unique(point_labels)[1:]
        num_points = len(point_labels)
        
        # Create the (M, 2) tensor
        segmentation_instance_tensor = torch.zeros((num_points, 2), dtype=torch.long)
        segmentation_instance_tensor[:, 0] = 1  # Segmentation index (1 for every label)
        segmentation_instance_tensor[:, 1] = point_labels  # Instance labels
        
        # Return the updated output
        return (coords.numpy(), 
                features.numpy(), 
                segmentation_instance_tensor.numpy(), 
                None, 
                None, 
                None, 
                None, 
                idx)
