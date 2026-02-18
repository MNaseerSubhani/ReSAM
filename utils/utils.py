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


# def similarity_loss(features, queue, tau=0.07, sim_threshold=0):
#     """
#     features: [B, D] current batch embeddings (normalized)
#     queue: deque of [D] past embeddings (detached)
#     tau: temperature for softmax
#     sim_threshold: cosine similarity threshold to consider "similar"
#     """
#     if len(queue) == 0:
#         return -1

#     # Stack past features from queue
#     past_feats = torch.stack(list(queue), dim=0)  # [Q, D]
#     features = torch.stack(list(features), dim=0)  # [B, D]

#     # Normalize embeddings
#     features = F.normalize(features, dim=1)
#     past_feats = F.normalize(past_feats, dim=1)

#     # Compute cosine similarities
#     cos_sim = torch.mm(features, past_feats.t())  # [B, Q]

#     # Apply threshold: set values below threshold to 0
#     mask = (cos_sim >= sim_threshold).float()
#     cos_sim_masked = cos_sim * mask  # [B, Q], below threshold becomes 0

#     # Scale by temperature
#     logits = cos_sim_masked / tau

#     # Softmax over queue dimension
#     probs = F.softmax(logits, dim=1)

#     # Weighted alignment loss
#     loss = ((1 - cos_sim_masked) * probs).sum(dim=1).mean()

#     return loss


def similarity_loss(soft_feats, hard_feats, tau=0.07):
    """
    soft_feats: [B, D]
    hard_feats: [B, D]
    Cosine similarity alignment loss with temperature.
    """
    soft_feats = F.normalize(soft_feats, dim=1)
    hard_feats = F.normalize(hard_feats, dim=1)

    cos_sim = (soft_feats * hard_feats).sum(dim=1)

    # Temperature scaling: sharper when tau is small
    loss = ((1 - cos_sim) / tau).mean()

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



def save_incremental_by_image_name(out_dir, img_path, tag, img):
    """
    Saves incrementally:
    <name>_<tag>.jpg
    <name>_<tag>_2.jpg
    <name>_<tag>_3.jpg ...
    """
    base = os.path.splitext(os.path.basename(img_path))[0]

    # base file name
    file_path = os.path.join(out_dir, f"{base}_{tag}.jpg")

    # if exists → find next index
    if os.path.exists(file_path):
        idx = 2
        while True:
            new_path = os.path.join(out_dir, f"{base}_{tag}_{idx}.jpg")
            if not os.path.exists(new_path):
                file_path = new_path
                break
            idx += 1

    # save the image
    cv2.imwrite(file_path, img)



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
    out_dir
):
    os.makedirs(out_dir, exist_ok=True)

    img_path = img_paths[0]
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # ------------------------------------------
    # Load original image
    # ------------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load:", img_path)
        return

    H, W = img.shape[:2]

    # ------------------------------------------
    # Save original image (overwrite, no incremental)
    # ------------------------------------------
    cv2.imwrite(f"{out_dir}/{base_name}_orig.jpg", img)

    # ------------------------------------------
    # Save GT mask (overwrite, no incremental)
    # ------------------------------------------
    if gt_masks.ndim == 4:
        merged_gt = gt_masks[0].sum(dim=0)
    else:
        merged_gt = gt_masks[0]

    gt = (merged_gt.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/{base_name}_gt.jpg", gt)

    # -----------------------------------------------------
    # Save pred_stack merged mask (INCREMENTAL)
    # -----------------------------------------------------
    preds = pred_stack.detach().cpu().numpy()  # [N,1,H,W]
    preds = preds.squeeze(1)                   # [N,H,W]

    merged_pred = (preds.sum(axis=0) > 0.5).astype(np.uint8) * 255
    save_incremental_by_image_name(out_dir, img_path, "pred", merged_pred)

    # -----------------------------------------------------
    # Build pseudo mask from soft_masks (INCREMENTAL)
    # -----------------------------------------------------

    soft_masks = torch.sigmoid(torch.stack(soft_masks, dim=0))
    soft_masks = soft_masks[0].detach().cpu().numpy()     # [N,1,H,W]                         # [N,H,W]

    merged_masks = soft_masks.sum(axis=0)
    merged_masks = (merged_masks > 0.5).astype(np.uint8) * 255

 
    
    save_incremental_by_image_name(out_dir, img_path, "pseudo_mask", merged_masks)
