from alphaction.dataset import datasets

from .images import image_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
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

    video_cls = getattr(datasets, "VideoDataset", None)
    if (video_cls is not None and isinstance(dataset, video_cls)) or dataset_type == "videodataset":
        # Generic video dataset is evaluated as frame-level AP using image evaluator.
        return image_evaluation(**args)

    raise NotImplementedError("Unsupported dataset type {}.".format(dataset.__class__.__name__))
