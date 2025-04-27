import torch

class PredictionMatcher:

    def __init__(self,
        attempted_match_threshold: float = 0.2,
        successful_match_threshold: float = 0.7
    ):
        self.attempted_match_threshold = attempted_match_threshold
        self.successful_match_threshold = successful_match_threshold

    def matched_predictions(
        self,
        predictions_BHWD: torch.Tensor, 
        ground_truth_BHWD: torch.Tensor, 
    ):
        """
        Calculates successful and attempted matches between predictions and ground truth labels for each item in the batch.

        An attempted match is defined as a prediction that exceeds `attempted_match_threshold` overlap with the ground truth label.
        A successful match is defined as a prediction that exceeds `successful_match_threshold` overlap with the ground truth label.
        """
        assert predictions_BHWD.shape == ground_truth_BHWD.shape, "Predictions and ground truth must have the same shape"
        assert len(predictions_BHWD.shape) == 4, "Input tensors must be 4D (B, H, W, D)"
        assert predictions_BHWD.device == ground_truth_BHWD.device, "Predictions and ground truth must be on the same device"
        B, H, W, D = ground_truth_BHWD.shape

        overlaps = []
        for b in range(B):  # Iterate over the batch
            gt_labels = torch.unique(ground_truth_BHWD[b])  # Get unique labels in the ground truth
            pred_labels = torch.unique(predictions_BHWD[b])  # Get unique labels in the predictions
            batch_overlaps = {
                "predictions_attempted_matches": {label.item(): [] for label in pred_labels}, 
                "ground_truth_successful_matches": {label.item(): [] for label in gt_labels},
            }

            for pred_label in pred_labels:
                if pred_label == 0:  # Skip background label (assumed to be 0)
                    continue

                pred_mask = (predictions_BHWD[b] == pred_label)  # Binary mask for the current prediction label
                pred_mask_voxel_count = pred_mask.sum().item()

                for gt_label in gt_labels:
                    if gt_label == 0:  # Skip background label (assumed to be 0)
                        continue

                    gt_mask = (ground_truth_BHWD[b] == gt_label)  # Binary mask for the current ground truth label
                    gt_voxel_count = gt_mask.sum().item()  # Total number of voxels for this ground truth label
                    overlap = (gt_mask & pred_mask).sum().item()  # Count overlapping voxels

                    # Check if overlap exceeds the threshold
                    if overlap / gt_voxel_count > self.successful_match_threshold:
                        batch_overlaps["ground_truth_successful_matches"][gt_label.item()].append(pred_label.item())
                    if overlap / gt_voxel_count > self.attempted_match_threshold:
                        batch_overlaps["predictions_attempted_matches"][pred_label.item()].append(gt_label.item())
                    
            overlaps.append(batch_overlaps)

        return overlaps