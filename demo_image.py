import os
import argparse
import torch
import cv2

from alphaction.config import cfg
from alphaction.modeling.detector import build_detection_model
from alphaction.utils.checkpoint import ActionCheckpointer
from alphaction.dataset.datasets.cv2_transform import PreprocessWithBoxes
import alphaction.dataset.datasets.utils as utils
from alphaction.utils.visualize import annotate


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--config-file", required=True, help="config file")
    parser.add_argument("--image", required=True, help="path to input image")
    parser.add_argument("--output", default="output/image_demo.png", help="annotated output path")
    parser.add_argument("--ckpt", default=None, help="checkpoint file (overrides cfg.MODEL.WEIGHT)")
    parser.add_argument("--topk", type=int, default=5, help="top-k action labels to show per person")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("opts", help="Override config options", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # build model and load weights
    model = build_detection_model(cfg)
    model.to(device)
    model.eval()

    if args.ckpt:
        ckpt_file = args.ckpt
    else:
        ckpt_file = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHT) if cfg.MODEL.WEIGHT else None
    if ckpt_file and os.path.exists(ckpt_file):
        checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(ckpt_file)
    else:
        print("Warning: no checkpoint found — running with randomly initialized weights")

    # prepare preprocessing (reuse PreprocessWithBoxes behavior)
    # create a minimal dataset-config object for PreprocessWithBoxes
    from types import SimpleNamespace
    dataset_cfg = SimpleNamespace(
        BGR=False,
        TRAIN_USE_COLOR_AUGMENTATION=False,
        TRAIN_PCA_JITTER_ONLY=False,
        TRAIN_PCA_EIGVAL=[0.225, 0.224, 0.229],
        TRAIN_PCA_EIGVEC=[[0.467, 0.485, 0.456], [0.229, 0.224, 0.225], [0.197, 0.199, 0.201]],
        TEST_FORCE_FLIP=False,
    )
    preproc = PreprocessWithBoxes('test', cfg.DATA, dataset_cfg)

    # load image
    assert os.path.exists(args.image), f"Image not found: {args.image}"
    imgs = utils.retry_load_images([args.image], backend='cv2')

    # preprocess (no boxes provided)
    imgs_proc, _ = preproc.process(imgs, boxes=None)  # imgs_proc: (3, T, H, W)

    # pack pathways and add batch dim
    imgs_packed = utils.pack_pathway_output(cfg, imgs_proc, pathways=cfg.MODEL.BACKBONE.PATHWAYS)
    if cfg.MODEL.BACKBONE.PATHWAYS == 1:
        primary_input = imgs_packed[0][None].to(device)  # (1, C, T, H, W)
        secondary_input = None
    else:
        primary_input = imgs_packed[0][None].to(device)
        secondary_input = imgs_packed[1][None].to(device)

    h, w = primary_input.shape[-2:]
    whwh = torch.tensor([[w, h, w, h]], dtype=torch.float32).to(device)

    # run model
    with torch.no_grad():
        results = model(primary_input, secondary_input, whwh)

    # model returns (action_score_list, box_list) in eval mode
    if isinstance(results, tuple) and len(results) == 2:
        action_score_list, box_list = results
    else:
        raise RuntimeError("Unexpected model output format")

    # prepare readable labels
    # if open-vocab is used, try to get text vocab from backbone; otherwise fallback to numeric labels
    if cfg.DATA.OPEN_VOCABULARY and hasattr(model.backbone, 'text_encoder') and hasattr(model.backbone.text_encoder, 'text_data'):
        try:
            vocab = list(model.backbone.text_encoder.text_data.keys())
        except Exception:
            vocab = None
    else:
        vocab = None

    im_rgb = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
    annotated = im_rgb.copy()

    for i, (scores, boxes_norm) in enumerate(zip(action_score_list, box_list)):
        # boxes_norm: (N, 4) normalized xyxy
        if boxes_norm is None or boxes_norm.numel() == 0:
            continue
        # get top-k predictions per detected person
        topk = min(args.topk, scores.shape[-1])
        probs, ids = torch.topk(torch.softmax(scores, dim=-1), k=topk, dim=-1)
        # create phrases like `class_3 0.87` if no vocab
        phrases = []
        for p, idx in zip(probs.tolist(), ids.tolist()):
            if vocab is not None and idx < len(vocab):
                phrases.append(f"{vocab[idx]} {p:.2f}")
            else:
                phrases.append(f"class_{idx} {p:.2f}")
        # annotate: pass the first (best) phrase as label, show all in console
        best_phrase = phrases[0] if len(phrases) > 0 else ""
        # annotate all boxes for this image (boxes_norm is a tensor of shape (M,4))
        annotated = annotate(annotated, boxes_norm, normalized=True, phrases=[best_phrase], is_xyxy=True, set_text_color='white')
        # print detailed predictions to console
        for j, (b, pr) in enumerate(zip(boxes_norm.tolist(), phrases)):
            print(f"Det {i}-{j}: box={b}, {pr}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, annotated_bgr)
    print(f"Saved annotated image to {args.output}")


if __name__ == '__main__':
    main()
