import logging
from .image_eval import do_image_evaluation


def image_evaluation(dataset, predictions, output_folder, **kwargs):
    logger = logging.getLogger("alphaction.inference")
    logger.info("performing image evaluation.")
    return do_image_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        metric=kwargs.get("metric", "image_ap"),
        save_csv=kwargs.get("save_csv", False),
    )
