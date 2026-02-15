import math
import torch


def batch_different_inputs(inputs, size_divisible=0):
    '''
    :param inputs: a list of tensors
    :param size_divisible: output_size(width and height) should be divisble by this param
    :return: batched inputs as a single tensor
    '''
    assert isinstance(inputs, (tuple, list))
    max_size = tuple(max(s) for s in zip(*[clip.shape for clip in inputs]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size[3] = int(math.ceil(max_size[3] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(inputs),) + max_size
    batched_clips = inputs[0].new(*batch_shape).zero_()
    for clip, pad_clip in zip(inputs, batched_clips):
        pad_clip[:clip.shape[0], :clip.shape[1], :clip.shape[2], :clip.shape[3]].copy_(clip)

    return batched_clips


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched tensors and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.divisible = size_divisible
        self.size_divisible = self.divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        primary_inputs = batch_different_inputs(transposed_batch[0], self.size_divisible)
        if transposed_batch[1][0] is not None:
            secondary_inputs = batch_different_inputs(transposed_batch[1], self.size_divisible)
        else:
            secondary_inputs = None
        whwh = torch.stack(transposed_batch[2])
        boxes = transposed_batch[3]
        label_arrs = transposed_batch[4]
        metadata = transposed_batch[5]
        clip_ids = transposed_batch[6]
        return primary_inputs, secondary_inputs, whwh, boxes, label_arrs, metadata, clip_ids
