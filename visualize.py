import os
import argparse
import importlib
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence
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
    parser.add_argument("--cfg", "--cfg_file", dest="cfg_file", type=str,
                        default="configs.config_nwpu",
                        help="Dataset config module (e.g. configs.config_hrsid) or path (configs/config_hrsid.py)")
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
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Dataset indices to visualize (e.g. 0 1 2). If omitted, uses --num_samples.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="How many validation images to show (indices 0 .. num_samples-1) when --indices is not set")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.4)
    parser.add_argument("--show_grid", action="store_true",
                        help="Display matplotlib grid (on by default in Jupyter/Colab)")
    parser.add_argument("--no_display", action="store_true",
                        help="Do not show inline plots (Jupyter/Colab only saves files)")

    return parser.parse_args()


# -----------------------------
# Config
# -----------------------------
def load_dataset_cfg(cfg_file: str) -> Box:
    """Load dataset cfg from a module path or .py file path (same as train_resam --cfg)."""
    module_path = cfg_file.replace(".py", "").replace("\\", "/").replace("/", ".")
    module = importlib.import_module(module_path)
    return module.cfg


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


def resolve_indices(indices: Optional[Sequence[int]], num_samples: int) -> List[int]:
    if indices is not None and len(indices) > 0:
        return list(indices)
    n = max(1, int(num_samples))
    return list(range(n))


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "GoogleColabShell")
    except Exception:
        return False


def should_display(show_grid: bool, no_display: bool) -> bool:
    if no_display:
        return False
    if show_grid:
        return True
    return in_notebook()


def show_comparison_grid(results, cfg, overlay_alpha=0.4):
    """Colab/Jupyter: point prompt | pretrained SAM | ReSAM for each sample."""
    if not results:
        return

    rows = len(results)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    titles = ("Point prompt", "Pretrained SAM", "ReSAM")
    for i, item in enumerate(results):
        img = image_from_tensor(item["img"])
        ax_point, ax_sam, ax_resam = axes[i]

        ax_point.imshow(img)
        if cfg.prompt == "point":
            pos, neg = split_prompt_points(item["prompts"])
            if len(pos) > 0:
                ax_point.scatter(pos[:, 0], pos[:, 1], c="lime", s=80, edgecolors="black", linewidths=0.5)
            if len(neg) > 0:
                ax_point.scatter(neg[:, 0], neg[:, 1], c="red", s=80, edgecolors="black", linewidths=0.5)
        elif cfg.prompt == "box":
            boxes = item["prompts"] if isinstance(item["prompts"], torch.Tensor) else item["prompts"][0]
            boxes = boxes.detach().cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                ax_point.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax_point.set_title(f"{titles[0]} (idx={item['idx']})")
        ax_point.axis("off")

        rgb = cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB)
        ax_sam.imshow(rgb)
        ax_sam.imshow(item["sam"], alpha=overlay_alpha, cmap="Blues")
        ax_sam.set_title(titles[1])
        ax_sam.axis("off")

        ax_resam.imshow(rgb)
        ax_resam.imshow(item["resam"], alpha=overlay_alpha, cmap="Greens")
        ax_resam.set_title(titles[2])
        ax_resam.axis("off")

    plt.tight_layout()
    plt.show()


def run_visualization(
    ckpt: str,
    dataset: str = "NWPU",
    cfg_file: str = "configs.config_nwpu",
    sam_ckpt: Optional[str] = None,
    out_dir: str = "visualizations",
    prompt: str = "point",
    num_points: int = 1,
    img_size: int = 1024,
    indices: Optional[Sequence[int]] = None,
    num_samples: int = 1,
    device: Optional[str] = None,
    overlay_alpha: float = 0.4,
    display: Optional[bool] = None,
):
    """
    Run inference and optionally show inline plots (Colab/Jupyter).

    Example (Colab cell):
        from visualize import run_visualization
        run_visualization(ckpt="best_model.pth", cfg_file="configs.config_hrsid",
                          dataset="HRSID", num_samples=4)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg = Box(base_config)
    cfg.merge_update(load_dataset_cfg(cfg_file))
    cfg.dataset = dataset
    cfg.prompt = prompt
    cfg.num_points = num_points

    if sam_ckpt:
        cfg.model.checkpoint = os.path.dirname(sam_ckpt) + os.sep

    load_datasets = call_load_dataset(cfg)
    _, val_loader, _ = load_datasets(cfg, img_size=img_size, return_pt=True)
    sample_indices = resolve_indices(indices, num_samples)

    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Visualizing indices: {sample_indices}")
    os.makedirs(out_dir, exist_ok=True)

    sam_model = load_model(cfg, device, ckpt_path=sam_ckpt)
    resam_model = load_model(cfg, device, ckpt_path=ckpt)
    results = []

    with torch.no_grad():
        for idx in sample_indices:
            if idx < 0 or idx >= len(val_loader.dataset):
                print(f"Skipping invalid index {idx}")
                continue

            sample = val_loader.dataset[idx]
            image, bbox, gt_mask, image_path = sample

            images = image.unsqueeze(0).to(device)
            bboxes = (bbox.to(device),)
            gt_masks = (gt_mask.to(device),)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, sam_pred_masks, _, _ = sam_model(images, prompts)
            _, resam_pred_masks, _, _ = resam_model(images, prompts)

            img_bgr = tensor_to_bgr_uint8(images[0])
            sam_mask = combine_instance_masks(sam_pred_masks[0])
            resam_mask = combine_instance_masks(resam_pred_masks[0])
            base_name = base_name_from_path(image_path)

            paths = save_visualizations(
                out_dir, base_name, img_bgr, prompts[0], sam_mask, resam_mask,
                cfg.prompt, overlay_alpha,
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

    if display is None:
        display = in_notebook()
    if display and results:
        show_comparison_grid(results, cfg, overlay_alpha)

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    indices = resolve_indices(args.indices, args.num_samples)
    display = should_display(args.show_grid, args.no_display)

    run_visualization(
        ckpt=args.ckpt,
        dataset=args.dataset,
        cfg_file=args.cfg_file,
        sam_ckpt=args.sam_ckpt,
        out_dir=args.out_dir,
        prompt=args.prompt,
        num_points=args.num_points,
        img_size=args.img_size,
        indices=indices,
        num_samples=args.num_samples,
        device=args.device,
        overlay_alpha=args.overlay_alpha,
        display=display,
    )


if __name__ == "__main__":
    main()
