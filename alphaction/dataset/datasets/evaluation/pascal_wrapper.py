from .pascal_evaluation import object_detection_evaluation, standard_fields
import numpy as np



def parse_id(activity_list=None, class_num=24):
    if activity_list is None:  # use the class ID instead
        activity_list = ['Class{}'.format(i) for i in range(class_num)]
        # activity_list = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
    categories = []
    for i, act_name in enumerate(activity_list):
        categories.append({'id': i + 1, 'name': act_name})
    return categories


class STDetectionEvaluaterUCF(object):
    '''
    evaluater class designed for multi-iou thresholds
        based on https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py
    parameters:
        dataset that provide GT annos, in the format of AWSCVMotionDataset
        tiou_thresholds: a list of iou thresholds
    attributes:
        clear(): clear detection results, GT is kept
        load_detection_from_path(), load anno from a list of path, in the format of [confi x1 y1 x2 y2 scoresx15]
        evaluate(): run evaluation code
    '''

    def __init__(self, tiou_thresholds=[0.5], load_from_dataset=False, activity_list=None, class_num=24):
        categories = parse_id(activity_list=activity_list, class_num=class_num)
        self.class_num = class_num
        self.categories = categories
        self.tiou_thresholds = tiou_thresholds
        self.lst_pascal_evaluator = []
        self.load_from_dataset = load_from_dataset
        self.exclude_key = []
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(categories, matching_iou_threshold=iou))

    def clear(self):
        for evaluator in self.lst_pascal_evaluator:
            evaluator.clear()

    def load_ground_truth(self, ground_truth):
        # write into evaluator
        for image_key, info in ground_truth.items():
            boxes = info['bbox'].copy()  # normalized coordinates
            labels = np.array(info['labels'], dtype=int)
            resolution = info['resolution']
            boxes_eval = []
            labels_eval = []
            for box, label in zip(boxes, labels):
                area = (box[3] - box[1]) * resolution[0] * (box[2] - box[0]) * resolution[1]
                if area < 10: continue  # ignore too small boxes
                boxes_eval.append(box)
                labels_eval.append(label)
            if len(boxes_eval) == 0:  # no boxes
                self.exclude_key.append(image_key)  # mark the excluded frames to filter the detections later
                continue
            
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        standard_fields.InputDataFields.groundtruth_boxes:
                            np.vstack(boxes_eval),
                        standard_fields.InputDataFields.groundtruth_classes:
                            np.array(labels_eval, dtype=int),
                        standard_fields.InputDataFields.groundtruth_difficult:
                            np.zeros(len(boxes_eval), dtype=bool)
                    })
    

    def load_detection(self, detections):
        """ Load detection results from dict memory
        """
        for image_key, info in detections.items():
            # filtering out results that are in the excluded frames
            if image_key in self.exclude_key or len(info['boxes']) == 0:
                continue

            # sorted by confidence:
            boxes, labels, scores = info['boxes'], info['action_ids'], info['scores']
            index = np.argsort(-scores)
            boxes, labels, scores = boxes[index], labels[index], scores[index]

            # add info into evaluator
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_detected_image_info(
                    image_key, {
                        standard_fields.DetectionResultFields.detection_boxes: boxes,
                        standard_fields.DetectionResultFields.detection_classes: labels,
                        standard_fields.DetectionResultFields.detection_scores: scores
                    })

    def evaluate(self):
        result = {}
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            metrics = evaluator.evaluate()
            result.update(metrics)
        return result


class MultimodalSTDetectionEvaluaterUCF(STDetectionEvaluaterUCF):
    '''Multimodal (Image + Text) detection evaluator for activity recognition.
    
    Supports open vocabulary detection where text prompts and text features
    from vision-language models (like CLIP) are used.
    '''

    def __init__(self, tiou_thresholds=[0.5], load_from_dataset=False, activity_list=None, class_num=24,
                 text_prompts=None, text_features=None):
        """Initialize multimodal evaluator.
        
        Args:
            tiou_thresholds: List of IOU thresholds for evaluation
            load_from_dataset: Whether to load from dataset
            activity_list: List of activity names
            class_num: Number of classes
            text_prompts: Optional list of text prompts for each class
            text_features: Optional numpy array of text features [class_num, feature_dim]
        """
        super().__init__(tiou_thresholds, load_from_dataset, activity_list, class_num)
        
        self.text_prompts = text_prompts or [f"a photo of {act}" for act in (activity_list or ['Class{}'.format(i) for i in range(class_num)])]
        self.text_features = text_features
        
        # Use multimodal evaluator
        self.lst_pascal_evaluator = []
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(
                    self._create_categories_with_text(), matching_iou_threshold=iou))

    def _create_categories_with_text(self):
        """Create categories with text prompt information."""
        categories = []
        for i, act_name in enumerate(self.categories):
            categories.append({
                'id': act_name['id'],
                'name': act_name['name'],
                'text_prompt': self.text_prompts[i] if i < len(self.text_prompts) else f"a photo of {act_name['name']}"
            })
        return categories

    def load_ground_truth(self, ground_truth, text_prompts=None):
        """Load ground truth with optional text prompts.
        
        Args:
            ground_truth: Ground truth data dictionary
            text_prompts: Optional text prompts for each ground truth box
        """
        for image_key, info in ground_truth.items():
            boxes = info['bbox'].copy()
            labels = np.array(info['labels'], dtype=int)
            resolution = info['resolution']
            boxes_eval = []
            labels_eval = []
            text_prompts_eval = []
            
            for idx, (box, label) in enumerate(zip(boxes, labels)):
                area = (box[3] - box[1]) * resolution[0] * (box[2] - box[0]) * resolution[1]
                if area < 10: continue
                boxes_eval.append(box)
                labels_eval.append(label)
                
                # Add text prompt for this box
                if text_prompts and idx < len(text_prompts):
                    text_prompts_eval.append(text_prompts[idx])
                else:
                    # Use default prompt based on class
                    class_idx = label - 1  # Convert to 0-indexed
                    if class_idx < len(self.text_prompts):
                        text_prompts_eval.append(self.text_prompts[class_idx])
                    else:
                        text_prompts_eval.append(f"a photo of Class{class_idx}")
            
            if len(boxes_eval) == 0:
                self.exclude_key.append(image_key)
                continue
            
            for evaluator in self.lst_pascal_evaluator:
                gt_dict = {
                    standard_fields.InputDataFields.groundtruth_boxes:
                        np.vstack(boxes_eval),
                    standard_fields.InputDataFields.groundtruth_classes:
                        np.array(labels_eval, dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                        np.zeros(len(boxes_eval), dtype=bool)
                }
                
                # Add text prompts if available
                if text_prompts_eval:
                    gt_dict[standard_fields.InputDataFields.groundtruth_text_prompts] = text_prompts_eval
                
                evaluator.add_single_ground_truth_image_info(image_key, gt_dict)

    def load_detection(self, detections, text_prompts=None, text_features=None):
        """Load detection results with optional text prompts and features.
        
        Args:
            detections: Detection results dictionary
            text_prompts: Optional text prompts for each detection
            text_features: Optional text features for each detection
        """
        use_text_features = text_features is not None or self.text_features is not None
        features_to_use = text_features if text_features is not None else self.text_features
        
        for image_key, info in detections.items():
            if image_key in self.exclude_key or len(info['boxes']) == 0:
                continue

            boxes, labels, scores = info['boxes'], info['action_ids'], info['scores']
            index = np.argsort(-scores)
            boxes, labels, scores = boxes[index], labels[index], scores[index]
            
            # Prepare text prompts for detections
            det_text_prompts = None
            if text_prompts:
                det_text_prompts = [text_prompts[i] for i in index]
            else:
                # Generate default prompts based on class labels
                det_text_prompts = []
                for label in labels:
                    class_idx = label - 1
                    if class_idx < len(self.text_prompts):
                        det_text_prompts.append(self.text_prompts[class_idx])
                    else:
                        det_text_prompts.append(f"a photo of Class{class_idx}")

            for evaluator in self.lst_pascal_evaluator:
                det_dict = {
                    standard_fields.DetectionResultFields.detection_boxes: boxes,
                    standard_fields.DetectionResultFields.detection_classes: labels,
                    standard_fields.DetectionResultFields.detection_scores: scores
                }
                
                # Add text prompts
                det_dict[standard_fields.DetectionResultFields.detection_text_prompts] = det_text_prompts
                
                # Add text features if available
                if use_text_features and features_to_use is not None:
                    # Get features for the detected classes
                    det_text_features = features_to_use[labels - 1]  # Convert to 0-indexed
                    det_dict[standard_fields.DetectionResultFields.detection_text_features] = det_text_features
                
                evaluator.add_single_detected_image_info(image_key, det_dict)

    def evaluate(self):
        """Evaluate and return metrics including multimodal metrics."""
        result = {}
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            metrics = evaluator.evaluate()
            result.update(metrics)
        
        # Add text-based metrics if available
        if self.text_features is not None:
            result['multimodal_text_features_used'] = True
            result['num_text_prompts'] = len(self.text_prompts)
        
        return result


def frame_mAP_pascal(_results, _targets, vocab, logger, iou_list=[0.5]):
    evaluater = STDetectionEvaluaterUCF(tiou_thresholds=iou_list, activity_list=vocab, class_num=len(vocab))

    logger.info("Adding ground truth into evaluator")
    evaluater.load_ground_truth(_targets)

    logger.info("Adding predictions into evaluator")
    evaluater.load_detection(_results)

    eval_res = evaluater.evaluate()
    
    return eval_res


def frame_mAP_multimodal(_results, _targets, vocab, logger, iou_list=[0.5], 
                          text_prompts=None, text_features=None):
    """Compute multimodal mAP with text prompts and text features.
    
    Args:
        _results: Detection results
        _targets: Ground truth annotations
        vocab: Vocabulary of class names
        logger: Logger object
        iou_list: List of IOU thresholds
        text_prompts: Optional list of text prompts for each class
        text_features: Optional numpy array of text features [num_classes, feature_dim]
    
    Returns:
        Evaluation results with multimodal metrics
    """
    evaluater = MultimodalSTDetectionEvaluaterUCF(
        tiou_thresholds=iou_list, 
        activity_list=vocab, 
        class_num=len(vocab),
        text_prompts=text_prompts,
        text_features=text_features
    )

    logger.info("Adding ground truth into multimodal evaluator")
    evaluater.load_ground_truth(_targets, text_prompts=text_prompts)

    logger.info("Adding predictions into multimodal evaluator")
    evaluater.load_detection(_results, text_prompts=text_prompts, text_features=text_features)

    eval_res = evaluater.evaluate()
    
    return eval_res
