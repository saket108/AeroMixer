import argparse
import os

import torch
from alphaction.config import auto_sync_dataset_class_counts, cfg
from alphaction.dataset import make_data_loader
from alphaction.engine.inference import inference
from alphaction.modeling.detector import build_detection_model
from alphaction.modeling.runtime import configure_text_encoder
from alphaction.utils.checkpoint import ActionCheckpointer
from torch.utils.collect_env import get_pretty_env_info
from alphaction.utils.comm import synchronize, get_rank
from alphaction.utils.logger import setup_logger

# pytorch issue #973 (Unix-only)
try:
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
except ImportError:
    resource = None


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--verbose-startup",
        dest="verbose_startup",
        action="store_true",
        help="Print full environment and configuration details at startup.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if distributed:
        if device.type == "cuda":
            torch.cuda.set_device(args.local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    # Merge config file.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    class_sync = auto_sync_dataset_class_counts(cfg)
    cfg.freeze()

    # Print experimental infos.
    save_dir = ""
    logger = setup_logger("alphaction", save_dir, get_rank())
    logger.info("Using {} process(es)".format(num_gpus))
    logger.info("Config file: {}".format(args.config_file))
    logger.info(
        "Runtime summary: input_type=%s, annotation_format=%s, data_dir=%s, metric=%s, backbone=%s",
        cfg.DATA.INPUT_TYPE,
        cfg.DATA.ANNOTATION_FORMAT,
        cfg.DATA.PATH_TO_DATA_DIR,
        cfg.TEST.METRIC,
        cfg.MODEL.BACKBONE.CONV_BODY,
    )
    if class_sync and class_sync.get("applied"):
        logger.info(
            "Auto-synced detector class counts from dataset (%s split): num_classes=%d class_names=%s",
            class_sync.get("split"),
            int(class_sync.get("num_classes", 0)),
            class_sync.get("class_names", []),
        )
    if args.verbose_startup:
        logger.info(cfg)
        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + get_pretty_env_info())
    else:
        logger.info(
            "Startup logs condensed. Use --verbose-startup for full env/config dump."
        )

    # Build the model.
    if cfg.MODEL.DET not in ("AeroLiteDetector", "STMDetector"):
        raise ValueError(
            f"Unsupported detector '{cfg.MODEL.DET}'. The active runtime supports 'AeroLiteDetector' (legacy alias: 'STMDetector')."
        )
    model = build_detection_model(cfg)
    model.to(device)

    # load weight.
    output_dir = cfg.OUTPUT_DIR
    checkpointer = ActionCheckpointer(cfg, model, save_dir=output_dir)
    ckpt_file = os.path.join(output_dir, cfg.MODEL.WEIGHT) if cfg.MODEL.WEIGHT else None
    checkpointer.load(ckpt_file)

    output_folders = [None] * len(cfg.DATA.DATASETS)
    dataset_names = cfg.DATA.DATASETS
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            inf_folder = (
                "inference" if not cfg.TEST.SMALL_OPEN_WORLD else "inference_small"
            )
            output_folder = os.path.join(cfg.OUTPUT_DIR, inf_folder, dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder

    # Do inference.
    data_loaders_test, vocabularies_test, _ = make_data_loader(
        cfg, is_train=False, is_distributed=distributed
    )
    for i, (output_folder, dataset_name, data_loader_test) in enumerate(
        zip(output_folders, dataset_names, data_loaders_test)
    ):
        # set open vocabulary
        if len(vocabularies_test) > 0:
            configure_text_encoder(model, vocabularies_test[i])

        inference(
            model,
            data_loader_test,
            dataset_name,
            output_folder=output_folder,
            metric=cfg.TEST.METRIC,
            use_cache=True,
        )
        synchronize()


if __name__ == "__main__":
    main()
