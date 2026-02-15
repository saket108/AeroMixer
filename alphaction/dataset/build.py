import bisect
import copy
import torch.utils.data
from alphaction.utils.comm import get_world_size
from . import datasets as D
from . import samplers
from .collate_batch import BatchCollator


def _is_image_dataset_name(name):
    return str(name).lower() in ["images", "imagefolder", "image"]


def _is_image_input(cfg):
    if len(cfg.DATA.DATASETS) == 0:
        return str(getattr(cfg.DATA, "INPUT_TYPE", "video")).lower() == "image"
    return _is_image_dataset_name(cfg.DATA.DATASETS[0]) or str(getattr(cfg.DATA, "INPUT_TYPE", "video")).lower() == "image"


def _resolve_batch_size(cfg, is_train):
    use_image = _is_image_input(cfg)
    if is_train:
        if use_image and getattr(cfg.SOLVER, "IMAGES_PER_BATCH", -1) > 0:
            return cfg.SOLVER.IMAGES_PER_BATCH
        return cfg.SOLVER.VIDEOS_PER_BATCH
    if use_image and getattr(cfg.TEST, "IMAGES_PER_BATCH", -1) > 0:
        return cfg.TEST.IMAGES_PER_BATCH
    return cfg.TEST.VIDEOS_PER_BATCH


def build_dataset(cfg, split):
    dataset_name = str(cfg.DATA.DATASETS[0]).lower()

    if dataset_name == 'ucf24':
        dataset = D.UCF24(cfg, split)
    elif dataset_name == 'jhmdb':
        dataset = D.Jhmdb(cfg, split)
    elif dataset_name == 'ava_v2.2':
        dataset = D.Ava(cfg, split)
    elif _is_image_dataset_name(dataset_name) or str(getattr(cfg.DATA, "INPUT_TYPE", "video")).lower() == "image":
        dataset = D.ImageDataset(cfg, split)
    else:
        raise NotImplementedError("Unsupported dataset '{}'".format(cfg.DATA.DATASETS[0]))

    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    info_fn = dataset.get_sample_info if hasattr(dataset, "get_sample_info") else dataset.get_video_info
    for i in range(len(dataset)):
        sample_info = info_fn(i)
        aspect_ratio = float(sample_info["height"]) / float(sample_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, samples_per_batch, num_iters=None, start_iter=0, drop_last=False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, samples_per_batch, drop_uneven=drop_last
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, samples_per_batch, drop_last=drop_last
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    samples_per_batch = _resolve_batch_size(cfg, is_train)
    if is_train:
        assert (
                samples_per_batch % num_gpus == 0
        ), "Train batch size ({}) must be divisible by the number of GPUs ({}) used.".format(samples_per_batch, num_gpus)
        samples_per_gpu = samples_per_batch // num_gpus
        shuffle = True
        drop_last = True
        split = 'train'
    else:
        assert (
                samples_per_batch % num_gpus == 0
        ), "Test batch size ({}) must be divisible by the number of GPUs ({}) used.".format(samples_per_batch, num_gpus)
        samples_per_gpu = samples_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        drop_last = False
        start_iter = 0
        split = 'test'

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    # build dataset
    datasets = build_dataset(cfg, split=split)

    # build sampler and dataloader
    data_loaders, vocabularies, iter_per_epoch_all = [], [], []
    for dataset in datasets:
        if is_train:
            iter_per_epoch = int(len(dataset) // samples_per_batch) if cfg.SOLVER.ITER_PER_EPOCH == -1 else cfg.SOLVER.ITER_PER_EPOCH
            iter_per_epoch_all.append(iter_per_epoch)
        num_iters = cfg.SOLVER.MAX_EPOCH * iter_per_epoch if is_train else None
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, samples_per_gpu, num_iters, start_iter, drop_last
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
        if cfg.DATA.OPEN_VOCABULARY:
            vocabularies.append(dataset.text_input)
        else:
            vocabularies.append(None)
    if is_train:
        assert len(data_loaders) == 1
        vocabulary_train = None
        if vocabularies[0] is not None:
            vocabulary_train = vocabularies[0]['closed']
        return data_loaders[0], vocabulary_train, iter_per_epoch_all[0]
    
    vocabularies_val = []
    if len(vocabularies) > 0:
        for vocab in vocabularies:
            if vocab is None:
                vocabularies_val.append(None)
            elif cfg.TEST.EVAL_OPEN:
                vocabularies_val.append(vocab['open'])
            else:
                vocabularies_val.append(vocab['closed'])
    
    return data_loaders, vocabularies_val, iter_per_epoch_all
