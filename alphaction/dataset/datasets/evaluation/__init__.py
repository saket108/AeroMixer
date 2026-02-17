from alphaction.dataset import datasets

from .images import image_evaluation

# Import pascal_evaluation for multimodal support
from . import pascal_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args. May include:
            - use_multimodal: bool, whether to use multimodal evaluation
            - text_prompts: list of text prompts for open vocabulary detection
            - text_features: numpy array of text features for detections
    
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    dataset_type = dataset.__class__.__name__.lower()

    image_cls = getattr(datasets, "ImageDataset", None)
    if (image_cls is not None and isinstance(dataset, image_cls)) or dataset_type == "imagedataset":
        return image_evaluation(**args)

    raise NotImplementedError("Unsupported dataset type {}.".format(dataset.__class__.__name__))


def evaluate_multimodal(dataset, predictions, output_folder, text_prompts=None, text_features=None, **kwargs):
    """Evaluate dataset with multimodal (image + text) support.
    
    This function supports open vocabulary detection where text prompts and
    text features from vision-language models (like CLIP) are used.
    
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        text_prompts: Optional list of text prompts for each detection class
        text_features: Optional numpy array of text features [num_classes, feature_dim]
        **kwargs: other args.
    
    Returns:
        evaluation result with multimodal metrics
    """
    args = dict(
        dataset=dataset, 
        predictions=predictions, 
        output_folder=output_folder,
        text_prompts=text_prompts,
        text_features=text_features,
        **kwargs
    )
    
    # Import multimodal evaluation functions
    from .pascal_evaluation import object_detection_evaluation as ode
    
    # Use multimodal evaluator if available
    dataset_type = dataset.__class__.__name__.lower()
    image_cls = getattr(datasets, "ImageDataset", None)
    
    if (image_cls is not None and isinstance(dataset, image_cls)) or dataset_type == "imagedataset":
        return image_evaluation(**args)
    
    raise NotImplementedError("Unsupported dataset type {}.".format(dataset.__class__.__name__))
