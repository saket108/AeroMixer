import torch.utils.data
from alphaction.utils.comm import get_world_size
from . import datasets as D
from .collate_batch import BatchCollator


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

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        from . import samplers
        return samplers.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def make_batch_data_sampler(dataset, sampler, samples_per_gpu, num_iters=None, start_iter=0, drop_last=False):
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

def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()

    if is_train:
        images_per_batch = cfg.SOLVER.IMAGES_PER_BATCH
        assert images_per_batch % num_gpus == 0
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        drop_last = True
        split = "train"
    else:
        images_per_batch = cfg.TEST.IMAGES_PER_BATCH
        assert images_per_batch % num_gpus == 0
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False
        drop_last = False
        split = "test"

    datasets = build_dataset(cfg, split=split)

    data_loaders = []
    vocabularies = []
    iter_per_epoch_all = []

    for dataset in datasets:

        if is_train:
            iter_per_epoch = len(dataset) // images_per_batch
            iter_per_epoch_all.append(iter_per_epoch)
            num_iters = cfg.SOLVER.MAX_EPOCH * iter_per_epoch
        else:
            num_iters = None

        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
            dataset,
            sampler,
            images_per_gpu,
            num_iters,
            start_iter,
            drop_last
        )

        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

        data_loaders.append(data_loader)

        if cfg.DATA.OPEN_VOCABULARY:
            vocabularies.append(dataset.text_input)
        else:
            vocabularies.append(None)

    if is_train:
        vocabulary_train = None
        if vocabularies[0] is not None:
            vocabulary_train = vocabularies[0]['closed']
        return data_loaders[0], vocabulary_train, iter_per_epoch_all[0]

    return data_loaders, vocabularies, 0
