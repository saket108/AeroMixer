import logging

from .image_eval import (
    do_image_evaluation,
    do_multimodal_image_evaluation,
    evaluate_with_text_prompts,
    compute_text_similarity_scores,
)


def image_evaluation(dataset, predictions, output_folder, **kwargs):
    """Standard image evaluation.
    
    Args:
        dataset: Dataset object
        predictions: Model predictions
        output_folder: Folder to save results
        **kwargs: Additional arguments
        
    Returns:
        tuple: (eval_res, results)
    """
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


def multimodal_image_evaluation(dataset, predictions, output_folder, **kwargs):
    """Multimodal (image + text) evaluation.
    
    Args:
        dataset: Dataset object
        predictions: Model predictions
        output_folder: Folder to save results
        **kwargs: Additional arguments including:
            - text_prompts: Text prompts for open vocabulary detection
            - metric: Evaluation metric
            - save_csv: Whether to save CSV
            
    Returns:
        tuple: (eval_res, results)
    """
    logger = logging.getLogger("alphaction.inference")
    logger.info("performing multimodal image evaluation.")
    
    text_prompts = kwargs.get("text_prompts", None)
    
    if text_prompts is not None:
        # Use text prompts for open vocabulary detection
        return evaluate_with_text_prompts(
            predictions=predictions,
            dataset=dataset,
            text_prompts=text_prompts,
            output_folder=output_folder,
            logger=logger,
        )
    
    # Use standard multimodal evaluation
    return do_multimodal_image_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        metric=kwargs.get("metric", "image_ap"),
        text_prompts=text_prompts,
        save_csv=kwargs.get("save_csv", False),
    )


def evaluate_with_prompts(dataset, predictions, prompts, output_folder=None, **kwargs):
    """Evaluate with custom prompts.
    
    This is a convenience function for evaluating with custom text prompts.
    
    Args:
        dataset: Dataset object
        predictions: Model predictions
        prompts: Text prompts (dict or list)
        output_folder: Folder to save results
        **kwargs: Additional arguments
        
    Returns:
        dict: Evaluation results
    """
    logger = logging.getLogger("alphaction.inference")
    
    # Determine if prompts are for open vocabulary detection
    if isinstance(prompts, dict):
        # Dict mapping class IDs to text prompts
        text_prompts = prompts
    elif isinstance(prompts, list):
        # List of text prompts
        text_prompts = prompts
    else:
        raise ValueError(f"Invalid prompts type: {type(prompts)}")
    
    return evaluate_with_text_prompts(
        predictions=predictions,
        dataset=dataset,
        text_prompts=text_prompts,
        output_folder=output_folder,
        logger=logger,
    )


def create_text_prompts_from_classes(class_names, template="a photo of {}"):
    """Create text prompts from class names.
    
    Args:
        class_names: List of class names
        template: Prompt template (use {} as placeholder)
        
    Returns:
        dict: Class ID to text prompt mapping
    """
    prompts = {}
    for i, class_name in enumerate(class_names):
        prompts[i] = template.format(class_name)
    return prompts


def create_text_prompts_from_classes_list(class_names, template="a photo of {}"):
    """Create text prompts from class names as a list.
    
    Args:
        class_names: List of class names
        template: Prompt template (use {} as placeholder)
        
    Returns:
        list: Text prompts
    """
    return [template.format(class_name) for class_name in class_names]
