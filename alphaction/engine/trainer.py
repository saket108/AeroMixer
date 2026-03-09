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
                log_parts = [
                    f"val_epoch: {int(epoch)}",
                    f"dataset: {dataset_name}",
                ]
                for key in [
                    "PascalBoxes_Precision/mAP@0.5IOU",
                    "PascalBoxes_Precision/mAP@0.5:0.95IOU",
                    "Detection/Precision@0.5IOU",
                    "Detection/Recall@0.5IOU",
                    "SmallObject/AP@0.5IOU",
                ]:
                    if key in summary:
                        short_key = {
                            "PascalBoxes_Precision/mAP@0.5IOU": "mAP@0.5",
                            "PascalBoxes_Precision/mAP@0.5:0.95IOU": "mAP@0.5:0.95",
                            "Detection/Precision@0.5IOU": "precision",
                            "Detection/Recall@0.5IOU": "recall",
                            "SmallObject/AP@0.5IOU": "small_ap",
                        }[key]
                        log_parts.append(f"{short_key}: {summary[key]:.4f}")
                logger.info("  ".join(log_parts))
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
    val_history = []

    model.train()

    logger.info("Image Training Mode Enabled")

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
            epoch = max(1, int((iteration + iter_per_epoch - 1) // iter_per_epoch))
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
