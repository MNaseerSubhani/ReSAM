import random
from collections import deque
from tqdm import tqdm
from box import Box
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import os
from .sample_utils import get_point_prompts

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])

def concatenate_images_with_padding(images, padding=10, color=(255, 255, 255)):
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]

    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)
    
    if len(images[0].shape) == 3:
        new_image = np.full((max_height, total_width, 3), color, dtype=np.uint8)
    else:
        new_image = np.full((max_height, total_width), color[0], dtype=np.uint8)

    x_offset = 0
    for image in images:
        new_image[0:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1] + padding
    
    return new_image

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou

def calc_iou_matrix(mask_list1, mask_list2):
    iou_matrix = torch.zeros((len(mask_list1), len(mask_list2)))
    for i, mask1 in enumerate(mask_list1):
        for j, mask2 in enumerate(mask_list2):
            iou_matrix[i, j] = calculate_iou(mask1, mask2)
    return iou_matrix

def cal_mask_ious(
    cfg,
    model,
    images_weak,
    prompts,
    gt_masks,
):
    with torch.no_grad():
         _, soft_masks, _, _ = model(images_weak, prompts)   

    for i, (soft_mask, gt_mask) in enumerate(zip(soft_masks, gt_masks)):  
        soft_mask = (soft_mask > 0).float()
        mask_ious = calc_iou_matrix(soft_mask, soft_mask)
        indices = torch.arange(mask_ious.size(0))
        mask_ious[indices, indices] = 0.0
    return mask_ious, soft_mask



def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts

def generate_predict_feats(cfg, embed, pseudo_label, gts):
    coords, lbls = gts
    selected_coords = []
    
    num_insts = len(pseudo_label)
    num_points = cfg.num_points
    for coord_grp, lbl_grp in zip(coords, lbls):
        for coord, lbl in zip(coord_grp, lbl_grp):  
            if lbl.item() == 1:  
                selected_coords.append(coord.tolist())

    # Downsample coordinates (SAM's stride is 16)
    coords = [[int(c // 16) for c in pair] for pair in selected_coords]

    embed = embed.permute(1, 2, 0)  # [H, W, C]

    pos_pts = [] 

    for index in range(0, num_insts * num_points, num_points):
        index = random.randint(0, num_points - 1)
        x, y = coords[index]
        pos_pt = embed[x, y]
        pos_pts.append(pos_pt)

    predict_feats = torch.stack(pos_pts, dim=0)

    return predict_feats


def similarity_loss(features, queue, tau=0.07, sim_threshold=0):
    """
    features: [B, D] current batch embeddings (normalized)
    queue: deque of [D] past embeddings (detached)
    tau: temperature for softmax
    sim_threshold: cosine similarity threshold to consider "similar"
    """
    if len(queue) == 0:
        return -1

    # Stack past features from queue
    past_feats = torch.stack(list(queue), dim=0)  # [Q, D]
    features = torch.stack(list(features), dim=0)  # [B, D]

    # Normalize embeddings
    features = F.normalize(features, dim=1)
    past_feats = F.normalize(past_feats, dim=1)

    # Compute cosine similarities
    cos_sim = torch.mm(features, past_feats.t())  # [B, Q]

    # Apply threshold: set values below threshold to 0
    mask = (cos_sim >= sim_threshold).float()
    cos_sim_masked = cos_sim * mask  # [B, Q], below threshold becomes 0

    # Scale by temperature
    logits = cos_sim_masked / tau

    # Softmax over queue dimension
    probs = F.softmax(logits, dim=1)

    # Weighted alignment loss
    loss = ((1 - cos_sim_masked) * probs).sum(dim=1).mean()

    return loss


def get_bbox_feature(embedding_map, bbox, stride=16, pooling='avg'):
    """
    Extract a feature vector from an embedding map given a bounding box.
    
    Args:
        embedding_map (torch.Tensor): Shape (C, H_feat, W_feat) or (B, C, H_feat, W_feat)
        bbox (list or torch.Tensor): [x1, y1, x2, y2] in original image coordinates
        stride (int): Downscaling factor between image and feature map
        pooling (str): 'avg' or 'max' pooling inside the bbox region
        
    Returns:
        torch.Tensor: Feature vector of shape (C,)
    """
    # If batch dimension exists, assume batch size 1
    if embedding_map.dim() == 4:
        embedding_map = embedding_map[0]

    C, H_feat, W_feat = embedding_map.shape
    x1, y1, x2, y2 = bbox

    # Map bbox to feature map coordinates
    fx1 = max(int(x1 / stride), 0)
    fy1 = max(int(y1 / stride), 0)
    fx2 = min(int((x2 + stride - 1) / stride), W_feat)  # ceil division
    fy2 = min(int((y2 + stride - 1) / stride), H_feat)

    # Crop the feature map to bbox region
    region = embedding_map[:, fy1:fy2, fx1:fx2]

    if region.numel() == 0:
        # fallback to global feature if bbox is too small
        region = embedding_map

    # Pool to get a single feature vector
    if pooling == 'avg':
        feature_vec = region.mean(dim=(1,2))
    elif pooling == 'max':
        feature_vec = region.amax(dim=(1,2))
    else:
        raise ValueError("pooling must be 'avg' or 'max'")

    return feature_vec



def create_entropy_mask(entropy_maps, threshold=0.5, device='cuda'):
    """
    Create a mask to reduce learning from high entropy regions.
    
    Args:
        entropy_maps: List of entropy maps for each instance
        threshold: Entropy threshold above which to mask out regions
        device: Device to place the mask on
    
    Returns:
        List of entropy masks (0 for high entropy, 1 for low entropy)
    """
    entropy_masks = []
    
    for entropy_map in entropy_maps:
        # Create binary mask: 1 for low entropy, 0 for high entropy
        entropy_mask = (entropy_map < threshold).float()
        entropy_masks.append(entropy_mask)
    
    return entropy_masks


# from PIL import Image
# import os
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from PIL import Image, ImageDraw

# def save_uncertanity_mask(cfg, model, loader):
#     model.eval()
#     pts = []
#     num_points = cfg.num_points


#     save_dir = "temp_test"
#     # make save directory
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(loader, desc='Generating target prototypes', ncols=100)):
            
#             imgs, boxes, masks, _ = batch
#             prompts = get_prompts(cfg, boxes, masks)

#             embeds, masks_pred, _, _ = model(imgs, prompts) 
#             del _

#             p = masks_pred[0].clamp(1e-6, 1 - 1e-6)

#             print(prompts[0][0].shape)
      

#             entropy_map = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
#             entropy_map = entropy_map.max(dim=0)[0]   # take pixel-wise max across instances → [B, H, W]
#             # print(entropy_map.shape)
#             # -----------------
#             # Save images
#             # -----------------
#             for b in range(imgs.shape[0]):
#                 # Convert image tensor (C,H,W) → (H,W,C)
#                 img_np = (imgs[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

#                 # ===== Ground Truth Mask =====
#                 gt_mask = masks[b].cpu().numpy()   # [N, H, W]
#                 if gt_mask.ndim == 3:
#                     gt_mask = np.max(gt_mask, axis=0) * 255   # merge all instances
#                 gt_mask = gt_mask.astype(np.uint8)

#                 # ===== Prediction Mask =====
#                 pred_mask = masks_pred[b].cpu().numpy()   # [N, H, W]
#                 if pred_mask.ndim == 3:
#                     pred_mask = np.max(pred_mask, axis=0)   
#                 pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
#                 pred_mask = pred_mask.astype(np.uint8)

#                 # ===== Entropy Heatmap =====
#                 entropy_np = entropy_map.cpu().numpy()
#                 entropy_norm = (entropy_np - entropy_np.min()) / (entropy_np.max() - entropy_np.min() + 1e-6)
#                 cmap = cm.get_cmap("viridis")
#                 entropy_color = (cmap(entropy_norm)[:, :, :3] * 255).astype(np.uint8)
                
#                 # -----------------
#                 # Overlay prompts on masks
#                 # -----------------
#                 gt_img = Image.fromarray(gt_mask).convert("RGB")
#                 pred_img = Image.fromarray(pred_mask).convert("RGB")

#                 draw_gt = ImageDraw.Draw(gt_img)
#                 draw_pred = ImageDraw.Draw(pred_img)

#                 # points[b] → torch.Size([num_points, 2, 2])
#                 pts = prompts[0][0].cpu().numpy()  # [num_points, 2, 2]
#                 for pnt in pts:
#                     x, y = pnt[0]   # first [x,y]
#                     label = int(pnt[1][0]) if pnt.shape[0] > 1 else 1  # assume label in second entry
#                     color = "green" if label == 1 else "red"
#                     r = 4
#                     draw_gt.ellipse((x-r, y-r, x+r, y+r), fill=color)
#                     draw_pred.ellipse((x-r, y-r, x+r, y+r), fill=color)

#                 # Save outputs
#                 Image.fromarray(img_np).save(os.path.join(save_dir, f"{i+1}.jpg"))         # original
#                 gt_img.save(os.path.join(save_dir, f"{i+1}_gt.jpg"))                       # GT + points
#                 pred_img.save(os.path.join(save_dir, f"{i+1}_pred.jpg"))                   # Pred + points
#                 Image.fromarray(entropy_color).save(os.path.join(save_dir, f"{i+1}_en.jpg")) # entropy




def save_incremental_by_image_name(out_dir, img_path, suffix, image):
    """
    Saves analyze result using image filename as base.
    If file exists, increments with _2, _3, ...
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # First candidate
    save_path = os.path.join(out_dir, f"{base_name}_{suffix}.jpg")

    # If exists → increment
    counter = 2
    while os.path.exists(save_path):
        save_path = os.path.join(out_dir, f"{base_name}_{suffix}_{counter}.jpg")
        counter += 1

    cv2.imwrite(save_path, image)
    return save_path


def draw_bbox(img, bbox, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    img = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), color, thickness)
    return img


def save_analyze_images(
    img_paths,
    gt_masks,
    pred_stack,
    soft_masks,
    bboxes,
    out_dir,
    index
):
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------
    # Load original image
    # ------------------------------------------
    img = cv2.imread(img_paths[0])
    # img = cv2.flip(img, 1)
    if img is None:
        print("Could not load:", img_paths[0])
        return

    H, W = img.shape[:2]

    # ------------------------------------------
    # Save original
    # ------------------------------------------
    cv2.imwrite(f"{out_dir}/{index}.jpg", img)

    # ------------------------------------------
    # Save GT mask
    # gt_masks = [1,H,W] tensor
    # ------------------------------------------

    merged_gt = gt_masks[0].sum(axis=0)
    gt = (merged_gt.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/{index}_gt.jpg", gt)

    # -----------------------------------------------------
    # Build merged predicted mask from pred_stack
    # pred_stack shape = [N,1,H,W] inside pred_stack[0]
    # -----------------------------------------------------
    preds = pred_stack.detach().cpu().numpy()     # [N,1,H,W]
    preds = preds.squeeze(1)                          # [N,H,W]

    merged_pred = preds.sum(axis=0)
    merged_pred = (merged_pred > 0.5).astype(np.uint8) * 255

    # cv2.imwrite(f"{out_dir}/{index}_pred.jpg", merged_pred)
    # save_incremental_pseudo_mask(out_dir, index, merged_pred, "pred")
    save_incremental_by_image_name(out_dir, img_paths[0], "pred", merged_pred)

    # -----------------------------------------------------
    # Build pseudo mask from soft_masks
    # soft_masks shape = list of N tensors: [1,H,W]
    # -----------------------------------------------------
    # Construct pseudo mask

    soft_masks = torch.sigmoid(torch.stack(soft_masks, dim=0))
    soft_masks = soft_masks[0].detach().cpu().numpy()     # [N,1,H,W]                         # [N,H,W]

    merged_masks = soft_masks.sum(axis=0)
    merged_masks = (merged_masks > 0.5).astype(np.uint8) * 255

    # H, W = merged_masks.shape

    # # Convert mask to 3-channel for drawing boxes
    # merged_color = cv2.cvtColor(merged_masks, cv2.COLOR_GRAY2BGR)

    # # 3) Draw bounding boxes on the mask
    # # bboxes is list of [x1, y1, x2, y2]
    # for (x1, y1, x2, y2) in bboxes:
    #     x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
    #     cv2.rectangle(merged_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 4) Save final image
    # cv2.imwrite(f"{out_dir}/{index}_pseudo_mask.jpg", merged_masks)
    # save_incremental_pseudo_mask(out_dir, index, merged_masks, "pseudo")
    save_incremental_by_image_name(out_dir, img_paths[0], "pseudo_mask", merged_masks)