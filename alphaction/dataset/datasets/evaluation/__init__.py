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

    # Legacy video dataset evaluators remain available if those dataset classes are used directly.
    if dataset_type == "ava":
        from .ava import ava_evaluation

        return ava_evaluation(**args)
    if dataset_type == "jhmdb":
        from .jhmdb import jhmdb_evaluation

        return jhmdb_evaluation(**args)
    if dataset_type == "ucf24":
        from .ucf24 import ucf24_evaluation

        return ucf24_evaluation(**args)

    image_cls = getattr(datasets, "ImageDataset", None)
    if (image_cls is not None and isinstance(dataset, image_cls)) or dataset_type == "imagedataset":
        return image_evaluation(**args)

    raise NotImplementedError("Unsupported dataset type {}.".format(dataset.__class__.__name__))
