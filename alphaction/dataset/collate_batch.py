import torch
import math


# ---------------------------------------------------------
# IMAGE BATCH PADDING (DETECTION STYLE)
# ---------------------------------------------------------

def pad_images(images, size_divisible=0):
    """
    images: list of [C,H,W]
    returns: [B,C,Hmax,Wmax]
    """

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    if size_divisible > 0:
        max_h = int(math.ceil(max_h / size_divisible) * size_divisible)
        max_w = int(math.ceil(max_w / size_divisible) * size_divisible)

    batch = images[0].new_zeros(len(images), images[0].shape[0], max_h, max_w)

    for i, img in enumerate(images):
        c, h, w = img.shape
        batch[i, :, :h, :w] = img

    return batch


# ---------------------------------------------------------
# COLLATOR
# ---------------------------------------------------------

class BatchCollator(object):
    """
    Collate function for image detection
    """

    def __init__(self, size_divisible=32):
        self.size_divisible = size_divisible

    def __call__(self, batch):

        primary_inputs = [b[0] for b in batch]
        secondary_inputs = None

        whwh = torch.stack([b[2] for b in batch])

        boxes = [torch.tensor(b[3], dtype=torch.float32) for b in batch]
        labels = [torch.tensor(b[4], dtype=torch.float32) if b[4] is not None else None for b in batch]

        metadata = [b[5] for b in batch]
        indices = [b[6] for b in batch]

        # pad images
        primary_inputs = pad_images(primary_inputs, self.size_divisible)

        return primary_inputs, secondary_inputs, whwh, boxes, labels, metadata, indices
