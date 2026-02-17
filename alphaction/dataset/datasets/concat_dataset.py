import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.dataset.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    
    Supports multimodal (image + text) datasets with text prompts and text features.
    """

    def __init__(self, datasets, multimodal=False, text_features=None):
        """Initialize concatenated dataset.
        
        Args:
            datasets: List of datasets to concatenate
            multimodal: Whether to use multimodal mode (text prompts/features)
            text_features: Optional text features for open vocabulary detection
        """
        super().__init__(datasets)
        self.multimodal = multimodal
        self.text_features = text_features
        
        # Propagate multimodal settings to child datasets
        if multimodal:
            for dataset in datasets:
                if hasattr(dataset, 'set_multimodal'):
                    dataset.set_multimodal(True)
                if text_features is not None and hasattr(dataset, 'set_text_features'):
                    dataset.set_text_features(text_features)

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_video_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_video_info(sample_idx)

    def get_sample_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        sample_dataset = self.datasets[dataset_idx]
        if hasattr(sample_dataset, "get_sample_info"):
            return sample_dataset.get_sample_info(sample_idx)
        return sample_dataset.get_video_info(sample_idx)

    def get_text_prompts(self, idx):
        """Get text prompts for a specific sample index.
        
        Args:
            idx: Sample index
            
        Returns:
            Text prompts for the sample if multimodal, None otherwise
        """
        if not self.multimodal:
            return None
            
        dataset_idx, sample_idx = self.get_idxs(idx)
        sample_dataset = self.datasets[dataset_idx]
        
        if hasattr(sample_dataset, "get_text_prompts"):
            return sample_dataset.get_text_prompts(sample_idx)
        return None

    def get_text_features(self, idx):
        """Get text features for a specific sample index.
        
        Args:
            idx: Sample index
            
        Returns:
            Text features for the sample if available, None otherwise
        """
        if self.text_features is None:
            return None
            
        dataset_idx, sample_idx = self.get_idxs(idx)
        sample_dataset = self.datasets[dataset_idx]
        
        if hasattr(sample_dataset, "get_text_features"):
            return sample_dataset.get_text_features(sample_idx)
        
        # Return global text features if dataset doesn't have specific ones
        return self.text_features

    def set_multimodal(self, multimodal):
        """Enable or disable multimodal mode.
        
        Args:
            multimodal: Boolean to enable/disable multimodal mode
        """
        self.multimodal = multimodal
        for dataset in self.datasets:
            if hasattr(dataset, 'set_multimodal'):
                dataset.set_multimodal(multimodal)

    def set_text_features(self, text_features):
        """Set text features for all child datasets.
        
        Args:
            text_features: Numpy array of text features [num_classes, feature_dim]
        """
        self.text_features = text_features
        for dataset in self.datasets:
            if hasattr(dataset, 'set_text_features'):
                dataset.set_text_features(text_features)
