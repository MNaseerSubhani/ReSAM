import os
import argparse
import numpy as np
import torch
import cv2
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
    parser = argparse.ArgumentParser(description="Run ReSAM inference and save visualizations")

    parser.add_argument("--dataset", type=str, default="NWPU",
                        choices=["NWPU", "WHU", "HRSID"])
    parser.add_argument("--cfg_file", type=str, default="configs/config_nwpu.py")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained ReSAM checkpoint")
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="Optional SAM checkpoint; if omitted, uses cfg.model.checkpoint (pretrained SAM)")
    parser.add_argument("--out_dir", type=str, default="visualizations",
                        help="Directory to save output images")
    parser.add_argument("--prompt", type=str, default="point",
                        choices=["point", "box"])
    parser.add_argument("--num_points", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--indices", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.4)
    parser.add_argument("--show_grid", action="store_true",
                        help="Also display a matplotlib summary grid")

    return parser.parse_args()


# -----------------------------
# Load Model
# -----------------------------
def load_model(cfg, device, ckpt_path=None):
    model = Model(cfg)
    model.setup()
    model = model.to(device)
    model.eval()

    if ckpt_path is not None and os.path.exists(ckpt_path):
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
    return merged.cpu().numpy().astype(bool)


def tensor_to_bgr_uint8(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


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


def draw_prompts_on_image(img_bgr, prompts, prompt_type):
    out = img_bgr.copy()

    if prompt_type == "point":
        pos, neg = split_prompt_points(prompts)
        for x, y in pos:
            cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.circle(out, (int(x), int(y)), 8, (0, 0, 0), 1)
        for x, y in neg:
            cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.circle(out, (int(x), int(y)), 8, (0, 0, 0), 1)
    elif prompt_type == "box":
        boxes = prompts if isinstance(prompts, torch.Tensor) else prompts[0]
        boxes = boxes.detach().cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return out


def overlay_mask(img_bgr, mask_bool, color_bgr, alpha=0.4):
    out = img_bgr.copy().astype(np.float32)
    if not mask_bool.any():
        return out.astype(np.uint8)

    color = np.array(color_bgr, dtype=np.float32)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color
    return out.astype(np.uint8)


def base_name_from_path(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def save_visualizations(out_dir, base_name, img_bgr, prompts, sam_mask, resam_mask,
                        prompt_type, overlay_alpha):
    os.makedirs(out_dir, exist_ok=True)

    points_path = os.path.join(out_dir, f"{base_name}_points.jpg")
    sam_path = os.path.join(out_dir, f"{base_name}_sam_overlay.jpg")
    resam_path = os.path.join(out_dir, f"{base_name}_resam_overlay.jpg")

    points_img = draw_prompts_on_image(img_bgr, prompts, prompt_type)
    cv2.imwrite(points_path, points_img)

    sam_overlay = overlay_mask(img_bgr, sam_mask, color_bgr=(255, 128, 0), alpha=overlay_alpha)
    cv2.imwrite(sam_path, sam_overlay)

    resam_overlay = overlay_mask(img_bgr, resam_mask, color_bgr=(0, 255, 128), alpha=overlay_alpha)
    cv2.imwrite(resam_path, resam_overlay)

    return points_path, sam_path, resam_path


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Build config (same pattern as train_resam.py)
    cfg = Box(base_config)
    cfg_module = args.cfg_file.replace(".py", "").replace("/", ".")
    exec(f"from {cfg_module} import cfg as dataset_cfg")
    cfg.merge_update(dataset_cfg)

    cfg.dataset = args.dataset
    cfg.prompt = args.prompt
    cfg.num_points = args.num_points

    if args.sam_ckpt:
        cfg.model.checkpoint = os.path.dirname(args.sam_ckpt) + os.sep

    # Load dataset (validation set, same as training validation)
    load_datasets = call_load_dataset(cfg)
    _, val_loader, _ = load_datasets(cfg, img_size=args.img_size, return_pt=True)

    print(f"Validation samples: {len(val_loader.dataset)}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Pretrained SAM (LoRA B=0 at init; no ReSAM weights)
    sam_model = load_model(cfg, device, ckpt_path=args.sam_ckpt)
    # Trained ReSAM
    resam_model = load_model(cfg, device, ckpt_path=args.ckpt)

    results = []

    with torch.no_grad():
        for idx in args.indices:
            if idx < 0 or idx >= len(val_loader.dataset):
                print(f"Skipping invalid index {idx}")
                continue

            sample = val_loader.dataset[idx]
            image, bbox, gt_mask, image_path = sample

            images = image.unsqueeze(0).to(device)
            bboxes = (bbox,)
            gt_masks = (gt_mask,)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, sam_pred_masks, _, _ = sam_model(images, prompts)
            _, resam_pred_masks, _, _ = resam_model(images, prompts)

            img_bgr = tensor_to_bgr_uint8(images[0])
            sam_mask = combine_instance_masks(sam_pred_masks[0])
            resam_mask = combine_instance_masks(resam_pred_masks[0])
            base_name = base_name_from_path(image_path)

            paths = save_visualizations(
                args.out_dir,
                base_name,
                img_bgr,
                prompts[0],
                sam_mask,
                resam_mask,
                cfg.prompt,
                args.overlay_alpha,
            )
            print(f"[idx={idx}] {image_path}")
            print(f"  points: {paths[0]}")
            print(f"  sam:    {paths[1]}")
            print(f"  resam:  {paths[2]}")

            results.append({
                "idx": idx,
                "img": images[0].cpu(),
                "img_bgr": img_bgr,
                "image_path": image_path,
                "prompts": prompts[0],
                "sam": sam_mask,
                "resam": resam_mask,
            })

    if args.show_grid and results:
        rows = len(results)
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))

        if rows == 1:
            axes = np.expand_dims(axes, 0)

        for i, item in enumerate(results):
            img = image_from_tensor(item["img"])
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

            ax_sam.imshow(cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB))
            ax_sam.imshow(item["sam"], alpha=args.overlay_alpha, cmap="Blues")
            ax_sam.set_title("Pretrained SAM")
            ax_sam.axis("off")

            ax_resam.imshow(cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB))
            ax_resam.imshow(item["resam"], alpha=args.overlay_alpha, cmap="Greens")
            ax_resam.set_title("ReSAM")
            ax_resam.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
