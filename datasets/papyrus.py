from pathlib import Path
from typing import List, Tuple
import torch
from torch.nn import functional as F

import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import IterableDataset
from synthetic_pages.nrrd_file import Nrrd
from volume_pointcloud_conversion import dense_volume_with_labels_to_points
from scrolls_instance_segmentation.data.synthetic_datamodule_cubes import SyntheticInstanceCubesDataset
from scrolls_instance_segmentation.data.real_scroll_datamodule_cubes import InstanceCubesDataset

synthetic_cubes = SyntheticInstanceCubesDataset(
  reference_volume_filename= "reference_volume.nrrd",
  reference_label_filename= "reference_labels.nrrd",
  spatial_transform= True,
  layer_dropout= True,
  layer_shuffle= True
)

class SyntheticPapyrusDataset(IterableDataset):
    def __init__(self, mode="train", label_offset=0):
        super().__init__()
        if not mode in ["train", "validation"]:
            raise ValueError("mode must be either 'train' or 'val'")

        # self.cube_dataset = InstanceCubesDataset(Path("/workspace/code/cubes/") / ("training" if mode == "train" else "validation"))
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
        
    def __iter__(self):
        idx=0
        for batch in self.cube_dataset:
            new_size = (96,96,96)
            batch['vol'] = F.interpolate(
                batch['vol'][None, None].float(), 
                size=new_size, 
                mode='trilinear', 
                align_corners=True
            )[0][0]
            batch['lbl'] = F.interpolate(
                batch['lbl'][None].float(), 
                size=new_size, 
                mode='trilinear', 
                align_corners=True
            )[0].long()
            coords, features, point_labels = dense_volume_with_labels_to_points(
                batch['vol'], 
                batch['lbl'], 
                min_density=batch['vol'].max()/2, 
                subsample_factor=2
            )
            
            # Get instance IDs once, excluding background (0)
            instance_ids = torch.unique(point_labels)[1:]
            num_points = len(point_labels)
            
            # Create the (M, 3) tensor
            
            segmentation_instance_tensor = torch.zeros((num_points, 3), dtype=torch.long)
            segmentation_instance_tensor[:, 0] = 1  # Segment mask
            segmentation_instance_tensor[:, 1] = point_labels  # Instance masks
            segmentation_instance_tensor[:, 2] = 1  # Segment masks

            # Return the updated output
            yield (coords.numpy(), 
                    features.numpy(), 
                    segmentation_instance_tensor.numpy(), 
                    f"data_{idx}", 
                    None, 
                    None, 
                    coords.numpy().copy(), 
                    None)
            idx += 1