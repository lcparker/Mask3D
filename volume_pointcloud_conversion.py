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

def dense_volume_with_labels_to_points(volume_nrrd: Nrrd, label_nrrd: Nrrd, min_density=0.1, subsample_factor=4):
    """
    Convert Nrrd volume and label map to point cloud format with instance labels.
    
    Args:
        volume_nrrd: Nrrd - object containing volume/density data
        label_nrrd: Nrrd - object containing instance labels (0=air, >0=sheet_id)
        min_density: float - threshold for considering a voxel as papyrus
        subsample_factor: int - factor to subsample points by to control memory usage
    
    Returns:
        coords: torch.Tensor of shape [N, 3] - point coordinates
        features: torch.Tensor of shape [N, 1] - density values
        labels: torch.Tensor of shape [N] - instance labels
    """
    # Get volume and label data
    volume = volume_nrrd.volume
    labelmap = label_nrrd.volume
    
    # Ensure volumes match
    if volume.shape != labelmap.shape:
        raise ValueError("Volume and labelmap shapes do not match")
    
    # Get coordinates of non-air voxels that have labels
    occupied = np.where((volume > min_density) & (labelmap > 0))
    
    # Stack coordinates
    coords = np.stack(occupied, axis=1) # Uses indices as coordinates
    features = volume[occupied][:, None] # Not sure about this...
    point_labels = labelmap[occupied]
    
    # Subsample points while maintaining sheet structure
    N = coords.shape[0]
    if subsample_factor > 1:
        idx = np.random.choice(N, N//subsample_factor, replace=False)
        coords = coords[idx]
        features = features[idx]
        point_labels = point_labels[idx]
    
    # Convert to torch tensors
    return (torch.from_numpy(coords).float(), 
            torch.from_numpy(features).float(),
            torch.from_numpy(point_labels).long())

# Example usage:
def process_nrrd_pair(volume_nrrd: Nrrd, label_nrrd: Nrrd):
    coords, features, labels = dense_volume_with_labels_to_points(volume_nrrd, label_nrrd)
    
    return {
        'coords': coords,
        'features': features,
        'labels': labels
    }