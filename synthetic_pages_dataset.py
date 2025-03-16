from synthetic_pages.nrrd_file import Nrrd
import numpy as np
import torch

from volume_pointcloud_conversion import dense_volume_to_points, dense_volume_with_labels_to_points
from scrolls_instance_segmentation.data.synthetic_datamodule_cubes import SyntheticInstanceCubesDataset


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
        
    def __len__(self):
        return 500  # Same as synthetic dataset
        
    def __getitem__(self, idx):
        batch = self.synthetic_gen._gather_batch()
        coords, features, point_labels = dense_volume_with_labels_to_points(
            batch['vol'], batch['lbl'], min_density=0.1
        )
        
        # Get instance IDs once, excluding background (0)
        instance_ids = torch.unique(point_labels)[1:]
        num_instances = len(instance_ids)
        num_points = len(point_labels)
        
        # Create both outputs
        target = {
            "labels": torch.ones(num_instances, dtype=torch.long),
            "masks": torch.zeros((num_instances, num_points), dtype=torch.bool)
        }
        
        # Fill masks in one pass through instance IDs
        for i, instance_id in enumerate(instance_ids):
            target["masks"][i] = (point_labels == instance_id)
        
        return coords, features, target, f"synthetic_{idx}"