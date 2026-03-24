import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from box import Box

from configs.base_config import base_config
from datasets import call_load_dataset
from utils.eval_utils import get_prompts
from utils.model import Model

torch.set_float32_matmul_precision('high')


# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run ReSAM inference")

    parser.add_argument("--dataset", type=str, default="NWPU",
                        choices=["NWPU", "WHU", "HRSID"])
    parser.add_argument("--cfg_file", type=str, default="configs/config_nwpu.py")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained ReSAM checkpoint")
    parser.add_argument("--sam_ckpt", type=str,
                        default="/content/ReSAM/pretrain/sam_vit_b_01ec64.pth")
    parser.add_argument("--prompt", type=str, default="point",
                        choices=["point", "box"])
    parser.add_argument("--num_points", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--indices", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


# -----------------------------
# Load Model
# -----------------------------
def load_model(cfg, device, ckpt_path=None):
    model = Model(cfg)
    model.setup()
    model = model.to(device)
    model.eval()

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)

    return model


# -----------------------------
# Utils
# -----------------------------
def combine_instance_masks(pred_mask_tensor, thr=0.5):
    probs = torch.sigmoid(pred_mask_tensor)
    binary = (probs > thr).float()
    merged = (binary.sum(dim=0) > 0).float()
    return merged.cpu().numpy()


def image_from_tensor(img_tensor):
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img_np, 0.0, 1.0)


def split_prompt_points(prompt_tuple):
    point_coords, point_labels = prompt_tuple
    coords = point_coords.detach().cpu().numpy()
    labels = point_labels.detach().cpu().numpy()

    pos = coords[labels == 1].reshape(-1, 2) if np.any(labels == 1) else np.empty((0, 2))
    neg = coords[labels == 0].reshape(-1, 2) if np.any(labels == 0) else np.empty((0, 2))

    return pos, neg


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Build config
    cfg = Box(base_config)
    exec(f"from {args.cfg_file.replace('.py', '').replace('/', '.')} import cfg as dataset_cfg")
    cfg.merge_update(dataset_cfg)

    cfg.dataset = args.dataset
    cfg.prompt = args.prompt
    cfg.num_points = args.num_points

    # Load dataset
    load_datasets = call_load_dataset(cfg)
    _, val_loader, _ = load_datasets(cfg, img_size=args.img_size, return_pt=True)

    print(f"Validation samples: {len(val_loader.dataset)}")

    # Load models
    sam_model = load_model(cfg, device, args.sam_ckpt)
    resam_model = load_model(cfg, device, args.ckpt)

    results = []

    with torch.no_grad():
        for idx in args.indices:
            sample = val_loader.dataset[idx]
            image, bbox, gt_mask, image_path = sample

            images = image.unsqueeze(0).to(device)
            bboxes = (bbox,)
            gt_masks = (gt_mask,)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, sam_pred_masks, _, _ = sam_model(images, prompts)
            _, resam_pred_masks, _, _ = resam_model(images, prompts)

            results.append({
                "idx": idx,
                "img": images[0].cpu(),
                "prompts": prompts[0],
                "sam": sam_pred_masks[0].cpu(),
                "resam": resam_pred_masks[0].cpu(),
            })

    # -----------------------------
    # Visualization
    # -----------------------------
    rows = len(results)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for i, item in enumerate(results):
        img = image_from_tensor(item["img"])
        sam_mask = combine_instance_masks(item["sam"])
        resam_mask = combine_instance_masks(item["resam"])

        ax_main, ax_sam, ax_resam = axes[i]

        ax_main.imshow(img)

        if cfg.prompt == "point":
            pos, neg = split_prompt_points(item["prompts"])
            if len(pos) > 0:
                ax_main.scatter(pos[:, 0], pos[:, 1], c="lime", s=30)
            if len(neg) > 0:
                ax_main.scatter(neg[:, 0], neg[:, 1], c="red", s=30)

        ax_main.set_title(f"Input (idx={item['idx']})")
        ax_main.axis("off")

        ax_sam.imshow(img)
        ax_sam.imshow(sam_mask, alpha=0.4)
        ax_sam.set_title("SAM Output")
        ax_sam.axis("off")

        ax_resam.imshow(img)
        ax_resam.imshow(resam_mask, alpha=0.4)
        ax_resam.set_title("ReSAM Output")
        ax_resam.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()