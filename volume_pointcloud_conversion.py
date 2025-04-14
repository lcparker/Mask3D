from synthetic_pages.nrrd_file import Nrrd
import numpy as np
import torch

def dense_volume_to_points(nrrd_obj: Nrrd, min_density=0.1, subsample_factor=4):
    """
    Convert Nrrd volume to point cloud format suitable for Mask3D.
    
    Args:
        nrrd_obj: Nrrd - object containing volume data and metadata
        min_density: float - threshold for considering a voxel as papyrus
        subsample_factor: int - factor to subsample points by to control memory usage
    
    Returns:
        coords: torch.Tensor of shape [N, 3] - point coordinates
        features: torch.Tensor of shape [N, 1] - density values at each point 
    """
    # Get volume data
    volume = nrrd_obj.volume
    
    # Get coordinates of non-air voxels
    occupied = np.where(volume > min_density)
    
    # Stack coordinates into (N, 3) array
    coords = np.stack(occupied, axis=1)
    
    # Get values for these points
    features = volume[occupied][:, None]  # Add feature dimension
    
    # Subsample if requested
    N = coords.shape[0]
    if subsample_factor > 1:
        idx = np.random.choice(N, N//subsample_factor, replace=False)
        coords = coords[idx]
        features = features[idx]
    
    # Convert to torch tensors
    return (torch.from_numpy(coords).float(), 
            torch.from_numpy(features).float())

def dense_volume_with_labels_to_points(volume_HWD: torch.Tensor, 
                                       labels_NHWD: torch.Tensor, 
                                       min_density=0.1, 
                                       subsample_factor=4):
    """
    Convert Nrrd volume and one-hot encoded label map to point cloud format with instance labels.
    
    Args:
        volume: torch.Tensor - 3D array containing volume data
        labelmap: torch.Tensor - one-hot encoded label map of shape (N, 256, 256, 256), where N is the number of labels
        min_density: float - threshold for considering a voxel as papyrus
        subsample_factor: int - factor to subsample points by to control memory usage
    
    Returns:
        coords: torch.Tensor of shape [M, 3] - point coordinates
        features: torch.Tensor of shape [M, 1] - density values
        labels: torch.Tensor of shape [M] - instance labels
    """
    assert len(volume_HWD.shape) == 3, "Volume must be a 3D array"
    assert len(labels_NHWD.shape) == 4, "Labelmap must be a 4D array"
    # Ensure volume and labelmap shapes match
    if volume_HWD.shape != labels_NHWD.shape[1:]:
        raise ValueError("Volume shape and labelmap spatial dimensions do not match")
    
    # Convert one-hot labelmap to label indices
    labelmap_HWD = torch.argmax(labels_NHWD, dim=0) # HACK the right way to do this is to pass both as tensors
    
    # Get coordinates of non-air voxels that have labels
    occupied = torch.where((volume_HWD > min_density) & (labelmap_HWD > 0))  # Exclude 'air' label (0)
    
    # Stack coordinates
    coords = torch.stack(occupied, axis=1)  # Shape: (M, 3), where M is the number of occupied voxels
    features = volume_HWD[occupied][:, None]  # Shape: (M, 1), density values
    point_labels = labelmap_HWD[occupied]  # Shape: (M,), instance labels
    
    # Subsample points while maintaining structure
    N = coords.shape[0]
    if subsample_factor > 1:
        idx = np.random.choice(N, N // subsample_factor, replace=False)
        coords = coords[idx]
        features = features[idx]
        point_labels = point_labels[idx]
    
    # Convert to torch tensors
    return (coords.float(), 
            torch.hstack([features, coords]).float(),
            point_labels.long())

# Example usage:
def process_nrrd_pair(volume_nrrd: Nrrd, label_nrrd: Nrrd):
    coords, features, labels = dense_volume_with_labels_to_points(volume_nrrd, label_nrrd)
    
    return {
        'coords': coords,
        'features': features,
        'labels': labels
    }