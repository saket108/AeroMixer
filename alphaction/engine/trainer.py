import datetime
import json
import logging
import os
import time
import torch
import torch.nn as nn

from alphaction.utils.metric_logger import MetricLogger
from alphaction.utils.comm import reduce_dict




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
        metric='image_ap'
):

    logger = logging.getLogger("alphaction.trainer")
    logger.info("Start training")

    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]

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
        optim_loss_terms = [value for key, value in loss_dict.items() if str(key).startswith("loss")]
        if not optim_loss_terms:
            raise RuntimeError("Model returned no optimization losses (keys starting with 'loss').")
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
            for key in ["attn_entropy_avg", "attn_diag_avg", "attn_tau_mean_avg", "refine_l1_avg"]:
                if key in meters.meters:
                    log_fields.append(f"{key}: {meters.meters[key].avg:.4f}")
            logger.info(
                meters.delimiter.join(log_fields)
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------
        if iteration % checkpoint_period == 0:
            checkpointer.save(f"model_{iteration:07d}", **arguments)

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
