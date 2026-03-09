import logging

import numpy as np
import torch
import torch.utils.data
from alphaction.config import uses_text_branch
from alphaction.utils.comm import get_world_size
from . import datasets as D
from .collate_batch import BatchCollator

logger = logging.getLogger("alphaction.dataset.build")


# --------------------------------------------------------
# IMAGE ONLY DATASET BUILDER
# --------------------------------------------------------

def build_dataset(cfg, split):
    """
    Always build ImageDataset.
    We removed video support completely.
    """
    dataset = D.ImageDataset(cfg, split)
    return [dataset]


# --------------------------------------------------------
# SAMPLER
# --------------------------------------------------------

def _compute_balanced_image_weights(dataset, cfg):
    num_classes = int(getattr(dataset, "num_classes", 0))
    samples = getattr(dataset, "samples", None)
    if num_classes <= 0 or not isinstance(samples, list) or len(samples) == 0:
        return None, None

    class_hist = torch.zeros(num_classes, dtype=torch.float64)
    labels_per_image = []
    for sample in samples:
        labels_raw = None
        if isinstance(sample, dict):
            labels_raw = sample.get("labels", None)
        labels = np.asarray(labels_raw if labels_raw is not None else [], dtype=np.int64).reshape(-1)
        labels = labels[(labels >= 0) & (labels < num_classes)]
        labels_per_image.append(labels)
        if labels.size > 0:
            binc = np.bincount(labels, minlength=num_classes)[:num_classes].astype(np.float64)
            class_hist += torch.from_numpy(binc)

    if float(class_hist.sum().item()) <= 0:
        return None, None

    min_count = max(1.0, float(getattr(cfg.DATALOADER, "BALANCED_SAMPLING_MIN_COUNT", 1.0)))
    power = float(getattr(cfg.DATALOADER, "BALANCED_SAMPLING_POWER", 0.75))
    empty_weight = max(1e-6, float(getattr(cfg.DATALOADER, "BALANCED_SAMPLING_EMPTY_WEIGHT", 0.25)))

    class_hist_safe = torch.clamp(class_hist, min=min_count)
    inv = (class_hist_safe.mean() / class_hist_safe).pow(power)
    inv = inv / torch.clamp(inv.mean(), min=1e-6)

    image_weights = torch.empty(len(labels_per_image), dtype=torch.double)
    for idx, labels in enumerate(labels_per_image):
        if labels.size == 0:
            weight = empty_weight
        else:
            weight = float(inv[torch.from_numpy(labels)].mean().item())
        image_weights[idx] = max(weight, 1e-6)

    image_weights = image_weights / torch.clamp(image_weights.mean(), min=1e-6)
    return image_weights, class_hist


def _compute_tile_group_ids(dataset):
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, list) or len(samples) == 0:
        return None, 0

    base_counts = {}
    for sample in samples:
        tile_meta = sample.get("tile_meta") if isinstance(sample, dict) else None
        if not isinstance(tile_meta, dict) or not bool(tile_meta.get("is_tiled", False)):
            continue
        base_id = str(tile_meta.get("base_image_id", "")).strip()
        if not base_id:
            continue
        base_counts[base_id] = int(base_counts.get(base_id, 0)) + 1

    group_ids = []
    group_lookup = {}
    next_group_id = 0
    tiled_samples = 0
    fallback_group_id = None
    repeated_tiled_samples = sum(
        count for count in base_counts.values() if int(count) > 1
    )

    for sample in samples:
        tile_meta = sample.get("tile_meta") if isinstance(sample, dict) else None
        base_id = None
        if isinstance(tile_meta, dict) and bool(tile_meta.get("is_tiled", False)):
            base_id = str(tile_meta.get("base_image_id", "")).strip() or None
        if base_id is not None:
            tiled_samples += 1
            if int(base_counts.get(base_id, 0)) > 1:
                if base_id not in group_lookup:
                    group_lookup[base_id] = next_group_id
                    next_group_id += 1
                group_ids.append(group_lookup[base_id])
            else:
                if fallback_group_id is None:
                    fallback_group_id = next_group_id
                    next_group_id += 1
                group_ids.append(fallback_group_id)
        else:
            if fallback_group_id is None:
                fallback_group_id = next_group_id
                next_group_id += 1
            group_ids.append(fallback_group_id)

    if repeated_tiled_samples <= 1:
        return None, tiled_samples
    return group_ids, tiled_samples


def make_data_sampler(dataset, shuffle, distributed, cfg=None, is_train=False):
    tile_grouping_active = False
    if (
        is_train
        and cfg is not None
        and bool(getattr(cfg.DATALOADER, "TILE_GROUP_BATCHING", False))
    ):
        group_ids, tiled_samples = _compute_tile_group_ids(dataset)
        tile_grouping_active = group_ids is not None and int(tiled_samples) > 1

    if distributed:
        if (
            is_train
            and cfg is not None
            and bool(getattr(cfg.DATALOADER, "BALANCED_SAMPLING", False))
            and not tile_grouping_active
        ):
            logger.warning(
                "DATALOADER.BALANCED_SAMPLING is enabled but distributed mode is active; "
                "falling back to DistributedSampler."
            )
        from . import samplers
        return samplers.DistributedSampler(dataset, shuffle=shuffle)

    if (
        is_train
        and shuffle
        and cfg is not None
        and bool(getattr(cfg.DATALOADER, "BALANCED_SAMPLING", False))
    ):
        if tile_grouping_active:
            logger.info(
                "DATALOADER.TILE_GROUP_BATCHING takes precedence over BALANCED_SAMPLING "
                "for tiled training data."
            )
        else:
            image_weights, class_hist = _compute_balanced_image_weights(dataset, cfg)
            if image_weights is not None:
                logger.info(
                    "Using WeightedRandomSampler for balanced sampling "
                    "(num_samples=%d, num_classes=%d, class_hist=%s)",
                    len(image_weights),
                    int(len(class_hist)),
                    [int(x) for x in class_hist.tolist()],
                )
                return torch.utils.data.WeightedRandomSampler(
                    weights=image_weights,
                    num_samples=len(image_weights),
                    replacement=True,
                )

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def make_batch_data_sampler(
    dataset,
    sampler,
    samples_per_gpu,
    num_iters=None,
    start_iter=0,
    drop_last=False,
    cfg=None,
):
    use_tile_grouping = bool(
        cfg is not None and getattr(cfg.DATALOADER, "TILE_GROUP_BATCHING", False)
    )
    batch_sampler = None

    if use_tile_grouping:
        group_ids, tiled_samples = _compute_tile_group_ids(dataset)
        if group_ids is None:
            if tiled_samples > 0:
                logger.info(
                    "DATALOADER.TILE_GROUP_BATCHING enabled but insufficient tiled samples were found; "
                    "falling back to standard batching."
                )
        elif isinstance(sampler, torch.utils.data.WeightedRandomSampler):
            logger.warning(
                "DATALOADER.TILE_GROUP_BATCHING is enabled but WeightedRandomSampler is active; "
                "falling back to standard batching. Disable BALANCED_SAMPLING to activate tile grouping."
            )
        else:
            from . import samplers

            logger.info(
                "Using GroupedBatchSampler for tile-aware batching "
                "(tiled_samples=%d, batch_size=%d).",
                int(tiled_samples),
                int(samples_per_gpu),
            )
            batch_sampler = samplers.GroupedBatchSampler(
                sampler,
                group_ids=group_ids,
                batch_size=samples_per_gpu,
                drop_uneven=drop_last,
            )

    if batch_sampler is None:
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            samples_per_gpu,
            drop_last=drop_last
        )

    if num_iters is not None:
        from . import samplers
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler,
            num_iters,
            start_iter
        )
    return batch_sampler


# --------------------------------------------------------
# DATALOADER
# --------------------------------------------------------

def make_data_loader(
    cfg, is_train=True, is_distributed=False, start_iter=0, split_override=None
):
    num_gpus = get_world_size()

    if is_train:
        images_per_batch = cfg.SOLVER.IMAGES_PER_BATCH
        assert images_per_batch % num_gpus == 0
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        drop_last = True
        split = str(split_override or "train")
    else:
        images_per_batch = cfg.TEST.IMAGES_PER_BATCH
        assert images_per_batch % num_gpus == 0
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False
        drop_last = False
        split = str(split_override or "test")

    datasets = build_dataset(cfg, split=split)

    data_loaders = []
    vocabularies = []
    iter_per_epoch_all = []

    for dataset in datasets:
        sampler = make_data_sampler(
            dataset,
            shuffle,
            is_distributed,
            cfg=cfg,
            is_train=is_train,
        )

        base_batch_sampler = make_batch_data_sampler(
            dataset,
            sampler,
            images_per_gpu,
            num_iters=None,
            start_iter=start_iter,
            drop_last=drop_last,
            cfg=cfg,
        )

        if is_train:
            iter_per_epoch = max(1, len(base_batch_sampler))
            iter_per_epoch_all.append(iter_per_epoch)
            num_iters = cfg.SOLVER.MAX_EPOCH * iter_per_epoch
        else:
            num_iters = None

        batch_sampler = make_batch_data_sampler(
            dataset,
            sampler,
            images_per_gpu,
            num_iters,
            start_iter,
            drop_last,
            cfg=cfg,
        )

        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

        data_loaders.append(data_loader)

        if uses_text_branch(cfg):
            vocabularies.append(dataset.text_input)
        else:
            vocabularies.append(None)

    if is_train:
        vocabulary_train = None
        if vocabularies[0] is not None:
            vocabulary_train = vocabularies[0]['closed']
        return data_loaders[0], vocabulary_train, iter_per_epoch_all[0]

    return data_loaders, vocabularies, 0
