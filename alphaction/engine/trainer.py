import datetime
import logging
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

    logger.info("🔥 Image Training Mode Enabled")

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
        losses = sum(loss_dict.values())

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
            logger.info(
                meters.delimiter.join([
                    f"eta: {eta_string}",
                    f"iter: {iteration}/{max_iter}",
                    f"loss: {meters.total_loss.avg:.4f}",
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}",
                ])
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
