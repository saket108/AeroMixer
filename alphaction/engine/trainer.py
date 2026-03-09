import datetime
import json
import logging
import os
import time

import torch

from alphaction.engine.inference import inference
from alphaction.utils.metric_logger import MetricLogger
from alphaction.utils.comm import reduce_dict


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_validation_vocabulary(model, vocabulary):
    if vocabulary is None:
        return
    base_model = _unwrap_model(model)
    backbone = getattr(base_model, "backbone", None)
    text_encoder = getattr(backbone, "text_encoder", None)
    if text_encoder is None:
        return
    try:
        text_encoder.set_vocabulary(vocabulary)
    except Exception:
        return


def _coerce_eval_metrics(output):
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], dict):
        return output[0]
    if isinstance(output, dict):
        return output
    return {}


def _metric_value(metrics, key):
    value = metrics.get(key, None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_validation_summary(metrics):
    summary = {}
    for key in [
        "PascalBoxes_Precision/mAP@0.5IOU",
        "PascalBoxes_Precision/mAP@0.5:0.95IOU",
        "Detection/Precision@0.5IOU",
        "Detection/Recall@0.5IOU",
        "SmallObject/AP@0.5IOU",
    ]:
        value = _metric_value(metrics, key)
        if value is not None:
            summary[key] = value
    return summary


def _num_epochs(max_iter, iter_per_epoch):
    iter_per_epoch = max(1, int(iter_per_epoch))
    return max(1, int((int(max_iter) + iter_per_epoch - 1) // iter_per_epoch))


def _init_epoch_summary():
    return {
        "num_batches": 0,
        "loss_sums": {},
        "instance_sum": 0,
        "size_label": "",
        "gpu_mem_gb": 0.0,
    }


def _format_image_size(primary_inputs):
    if primary_inputs is None or primary_inputs.ndim < 4:
        return ""
    height = int(primary_inputs.shape[-2])
    width = int(primary_inputs.shape[-1])
    if height == width:
        return str(height)
    return f"{height}x{width}"


def _update_epoch_summary(epoch_summary, reduced_losses, labels, primary_inputs):
    epoch_summary["num_batches"] += 1
    for key, value in reduced_losses.items():
        try:
            scalar = float(value.item() if isinstance(value, torch.Tensor) else value)
        except Exception:
            continue
        epoch_summary["loss_sums"][key] = (
            epoch_summary["loss_sums"].get(key, 0.0) + scalar
        )

    batch_instances = 0
    for label in labels:
        try:
            batch_instances += int(len(label))
        except Exception:
            continue
    epoch_summary["instance_sum"] += batch_instances

    size_label = _format_image_size(primary_inputs)
    if size_label:
        epoch_summary["size_label"] = size_label

    if isinstance(primary_inputs, torch.Tensor) and primary_inputs.is_cuda:
        try:
            gpu_mem = torch.cuda.max_memory_allocated(primary_inputs.device) / float(
                1024**3
            )
            epoch_summary["gpu_mem_gb"] = max(
                epoch_summary["gpu_mem_gb"], float(gpu_mem)
            )
        except Exception:
            pass


def _avg_epoch_loss(epoch_summary, key):
    count = max(1, int(epoch_summary["num_batches"]))
    return float(epoch_summary["loss_sums"].get(key, 0.0)) / float(count)


def _format_epoch_train_summary(epoch, total_epochs, epoch_summary):
    avg_instances = int(
        round(
            float(epoch_summary["instance_sum"])
            / max(1, int(epoch_summary["num_batches"]))
        )
    )
    return [
        f"{int(epoch):>8}/{int(total_epochs):<3}",
        f"{float(epoch_summary['gpu_mem_gb']):>9.2f}G",
        f"{_avg_epoch_loss(epoch_summary, 'loss_bbox'):>10.4f}",
        f"{_avg_epoch_loss(epoch_summary, 'loss_ce'):>10.4f}",
        f"{_avg_epoch_loss(epoch_summary, 'loss_giou'):>10.4f}",
        f"{avg_instances:>10d}",
        f"{str(epoch_summary.get('size_label', '')):>10}",
    ]


def _dataset_image_instance_counts(data_loader_val):
    dataset = getattr(data_loader_val, "dataset", None)
    if dataset is None:
        return 0, 0

    images = 0
    instances = 0

    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list):
        images = int(len(samples))
        for sample in samples:
            labels = sample.get("labels", None) if isinstance(sample, dict) else None
            try:
                instances += int(len(labels)) if labels is not None else 0
            except Exception:
                continue
        return images, instances

    try:
        images = int(len(dataset))
    except Exception:
        images = 0
    return images, instances


def _format_validation_table_row(summary, data_loader_val):
    images, instances = _dataset_image_instance_counts(data_loader_val)
    return [
        f"{'all':>22}",
        f"{images:>11d}",
        f"{instances:>11d}",
        f"{float(summary.get('Detection/Precision@0.5IOU', 0.0)):>10.4f}",
        f"{float(summary.get('Detection/Recall@0.5IOU', 0.0)):>10.4f}",
        f"{float(summary.get('PascalBoxes_Precision/mAP@0.5IOU', 0.0)):>10.4f}",
        f"{float(summary.get('PascalBoxes_Precision/mAP@0.5:0.95IOU', 0.0)):>10.4f}",
        f"{float(summary.get('SmallObject/AP@0.5IOU', 0.0)):>10.4f}",
    ]


def _run_validation_pass(
    model,
    dataset_names_val,
    data_loaders_val,
    vocabularies_val,
    output_folder,
    metric,
    epoch,
    iteration,
):
    logger = logging.getLogger("alphaction.trainer")
    history_records = []
    if not dataset_names_val or not data_loaders_val:
        return history_records

    base_model = _unwrap_model(model)
    base_model.eval()
    val_root = os.path.join(output_folder, "validation", f"epoch_{int(epoch):03d}")
    os.makedirs(val_root, exist_ok=True)

    try:
        for dataset_idx, (dataset_name, data_loader_val) in enumerate(
            zip(dataset_names_val, data_loaders_val)
        ):
            vocabulary = None
            if dataset_idx < len(vocabularies_val):
                vocabulary = vocabularies_val[dataset_idx]
            _set_validation_vocabulary(model, vocabulary)

            dataset_output = os.path.join(val_root, str(dataset_name))
            eval_output = inference(
                model,
                data_loader_val,
                dataset_name,
                output_folder=dataset_output,
                metric=metric,
            )
            metrics = _coerce_eval_metrics(eval_output)
            summary = _extract_validation_summary(metrics)
            record = {
                "epoch": int(epoch),
                "iteration": int(iteration),
                "dataset": str(dataset_name),
                **summary,
            }
            history_records.append(record)

            if summary:
                logger.info(
                    "                 Class     Images  Instances      Box(P)          R      mAP50  mAP50-95     smallAP"
                )
                logger.info(
                    "".join(_format_validation_table_row(summary, data_loader_val))
                )
    finally:
        base_model.train()
        torch.cuda.empty_cache()

    return history_records


def _build_model_extras(metadata, labels):
    extras = []
    for meta, label in zip(metadata, labels):
        item = {}
        if isinstance(meta, dict):
            item.update(meta)
        item["labels"] = label
        extras.append(item)
    return extras


# ------------------------------------------------------------
# FAST IMAGE TRAINER
# ------------------------------------------------------------
def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    tblogger,
    start_val,
    val_period,
    dataset_names_val,
    data_loaders_val,
    vocabularies_val,
    distributed,
    frozen_backbone_bn,
    output_folder,
    metric="image_ap",
    iter_per_epoch=None,
):

    logger = logging.getLogger("alphaction.trainer")
    logger.info("Start training")

    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    iter_per_epoch = int(iter_per_epoch or max(1, max_iter))
    total_epochs = _num_epochs(max_iter, iter_per_epoch)
    val_history = []
    epoch_summary = _init_epoch_summary()
    printed_epoch_header = False

    model.train()

    logger.info("Image Training Mode Enabled")
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    start_training_time = time.time()
    end = time.time()

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    for iteration, batch in enumerate(data_loader, start_iter):

        data_time = time.time() - end
        iteration += 1
        arguments["iteration"] = iteration

        primary_inputs, secondary_inputs, whwh, boxes, labels, metadata, idx = batch

        primary_inputs = primary_inputs.to(device)
        if secondary_inputs is not None:
            secondary_inputs = secondary_inputs.to(device)
        whwh = whwh.to(device)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        extras = _build_model_extras(metadata, labels)
        loss_dict = model(primary_inputs, secondary_inputs, whwh, boxes, labels, extras)
        optim_loss_terms = [
            value for key, value in loss_dict.items() if str(key).startswith("loss")
        ]
        if not optim_loss_terms:
            raise RuntimeError(
                "Model returned no optimization losses (keys starting with 'loss')."
            )
        losses = sum(optim_loss_terms)

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # scheduler
        if iteration % 10 == 0:
            scheduler.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        loss_dict["total_loss"] = losses.detach()
        loss_dict_reduced = reduce_dict(loss_dict)
        meters.update(**loss_dict_reduced)
        _update_epoch_summary(epoch_summary, loss_dict_reduced, labels, primary_inputs)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            log_fields = [
                f"eta: {eta_string}",
                f"iter: {iteration}/{max_iter}",
                f"loss: {meters.total_loss.avg:.4f}",
                f"lr: {optimizer.param_groups[0]['lr']:.6f}",
            ]
            if "loss_tile_consistency" in meters.meters:
                log_fields.append(
                    f"loss_tile_consistency: {meters.meters['loss_tile_consistency'].avg:.4f}"
                )
            for key in [
                "attn_entropy_avg",
                "attn_diag_avg",
                "attn_tau_mean_avg",
                "refine_l1_avg",
            ]:
                if key in meters.meters:
                    log_fields.append(f"{key}: {meters.meters[key].avg:.4f}")
            logger.info(meters.delimiter.join(log_fields))

        epoch = max(1, int((iteration + iter_per_epoch - 1) // iter_per_epoch))
        epoch_complete = iteration % iter_per_epoch == 0 or iteration == max_iter
        if epoch_complete:
            if not printed_epoch_header:
                logger.info(
                    "      Epoch    GPU_mem   box_loss   cls_loss  giou_loss  Instances       Size"
                )
                printed_epoch_header = True
            logger.info(
                "".join(_format_epoch_train_summary(epoch, total_epochs, epoch_summary))
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------
        if iteration % checkpoint_period == 0:
            checkpointer.save(f"model_{iteration:07d}", **arguments)

        if (
            dataset_names_val
            and data_loaders_val
            and val_period > 0
            and iteration >= start_val
            and iteration % val_period == 0
        ):
            logger.info(
                "Running validation at epoch %d (iter %d/%d).",
                epoch,
                iteration,
                max_iter,
            )
            val_records = _run_validation_pass(
                model=model,
                dataset_names_val=dataset_names_val,
                data_loaders_val=data_loaders_val,
                vocabularies_val=vocabularies_val,
                output_folder=output_folder,
                metric=metric,
                epoch=epoch,
                iteration=iteration,
            )
            if val_records:
                val_history.extend(val_records)
                history_path = os.path.join(output_folder, "val_metrics_history.json")
                with open(history_path, "w") as f:
                    json.dump(val_history, f, indent=2)
                logger.info("Saved validation history to %s", history_path)

        if epoch_complete:
            epoch_summary = _init_epoch_summary()
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    # --------------------------------------------------------
    # END TRAIN
    # --------------------------------------------------------
    total_training_time = time.time() - start_training_time
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            str(datetime.timedelta(seconds=int(total_training_time))),
            total_training_time / max_iter,
        )
    )

    summary_path = os.path.join(output_folder, "train_metrics_final.json")
    summary = {name: meter.global_avg for name, meter in meters.meters.items()}
    summary["max_iter"] = int(max_iter)
    summary["final_iteration"] = int(arguments.get("iteration", max_iter))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training metric summary to {summary_path}")
