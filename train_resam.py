

import os
import time
import argparse
import random
# from abc import ABC

import cv2
import numpy as np
import torch
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from matplotlib import cm

from scipy.ndimage import label
import numpy as np

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.fabric import _FabricOptimizer

from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.losses import DiceLoss, FocalLoss, Matching_Loss, cosine_similarity
from utils.eval_utils import AverageMeter, validate, get_prompts, calc_iou
from utils.tools import copy_model, create_csv, reduce_instances
from utils.utils import *

import  csv, copy
import torch
import torch.nn.functional as F
from collections import deque

# vis = False


class LossWatcher:
    def __init__(self, window=100, factor=10.0):
        self.window = window
        self.factor = factor
        self.losses = []
    
    def is_outlier(self, loss):
        if not torch.isfinite(loss):
            return True
        self.losses.append(loss.item())
        if len(self.losses) < self.window:
            return False
        recent_avg = sum(self.losses[-self.window:]) / self.window
        return loss.item() > recent_avg * self.factor





def process_forward(img_tensor, prompt, model):
    with torch.no_grad():
        _, masks_pred, _, _ = model(img_tensor, prompt)
    entropy_maps = []
    pred_ins = []
    eps=1e-8
    for i, mask_p in enumerate( masks_pred[0]):
        mask_p = torch.sigmoid(mask_p)
        p = mask_p.clamp(1e-6, 1 - 1e-6)
        if p.ndim == 2:
            p = p.unsqueeze(0)

        entropy = - (p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        max_ent = torch.log(torch.tensor(2.0, device=mask_p.device))
        entropy_norm = entropy / (max_ent + 1e-8)   # [0, 1]
        entropy_maps.append(entropy_norm)
        pred_ins.append(p)

    return entropy_maps, pred_ins
        

# persistent feature queue
feature_queue = deque(maxlen=32)  # keep up to 512 previous object embeddings


# def train_sam(
#     cfg: Box,
#     fabric: L.Fabric,
#     model: Model,
#     optimizer: _FabricOptimizer,
#     scheduler: _FabricOptimizer,
#     train_dataloader: DataLoader,
#     val_dataloader: DataLoader,
#     init_iou,
# ):

#     watcher = LossWatcher(window=50, factor=4)
#     focal_loss = FocalLoss()
#     dice_loss = DiceLoss()
#     best_ent = init_iou
#     best_state = copy.deepcopy(model.state_dict())
#     no_improve_count = 0
#     max_patience = cfg.get("patience", 3)  # stop if no improvement for X validations
#     match_interval = cfg.match_interval
#     eval_interval = int(len(train_dataloader) * 1)

#     window_size = 30

#     embedding_queue = []
#     iter_mem_usage = []
#     ite_em = 0

#     # Prepare output dirs
#     os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
#     csv_path = os.path.join(cfg.out_dir, "training_log.csv")

#     # Initialize CSV
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Epoch", "Iteration", "Val_ent", "Best_ent", "Status"])

#     fabric.print(f"Training with rollback enabled. Logging to: {csv_path}")

#     entropy_means = deque(maxlen=len(train_dataloader))

#     # overlap_ratios = []

#     eps = 1e-8
#     for epoch in range(1, cfg.num_epochs + 1):
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         focal_losses = AverageMeter()
#         dice_losses = AverageMeter()
#         iou_losses = AverageMeter()
#         total_losses = AverageMeter()
#         match_losses = AverageMeter()
#         end = time.time()
#         sim_losses = AverageMeter()
#         num_iter = len(train_dataloader)
#         entropy_means.clear()



#         for iter, data in enumerate(train_dataloader):
            
#             data_time.update(time.time() - end)
#             images_weak, images_strong, bboxes, gt_masks, img_paths= data
#             del data

            
#             step_size = 50
#             for j in range(0, len(gt_masks[0]), step_size):
                
                
#                 gt_masks_new = gt_masks[0][j:j+step_size].unsqueeze(0)
#                 prompts = get_prompts(cfg, bboxes, gt_masks_new)

#                 batch_size = images_weak.size(0)

#                 entropy_maps, preds = process_forward(images_weak, prompts, model)
                
#                 pred_stack = torch.stack(preds, dim=0)
#                 entropy_maps = torch.stack(entropy_maps, dim=0)

#                 # pred_binary = ((entropy_maps < 0.5) & (pred_stack > 0.5) ).float()
#                 pred_binary = (((1 - entropy_maps) * (pred_stack)) > 0.3) .float()
#                 overlap_count = pred_binary.sum(dim=0)
#                 overlap_map = (overlap_count > 1).float()
#                 invert_overlap_map = 1.0 - overlap_map

#                 bboxes = []
#                 point_list = []
#                 point_labels_list = []
#                 for i,  (pred, ent) in enumerate( zip(pred_binary, entropy_maps)):
#                     point_coords = prompts[0][0][i][:].unsqueeze(0)
#                     point_coords_lab = prompts[0][1][i][:].unsqueeze(0)

#                     pred_w_overlap = ((pred[0]*invert_overlap_map[0]  ) )#    * ((1 - 0.1 * ent[0]))
#                     ys, xs = torch.where(pred_w_overlap > 0.5)
#                     if len(xs) > 0 and len(ys) > 0:
#                         x_min, x_max = xs.min().item(), xs.max().item()
#                         y_min, y_max = ys.min().item(), ys.max().item()

#                         bboxes.append(torch.tensor([x_min, y_min , x_max, y_max], dtype=torch.float32))
          
#                 if len(bboxes) == 0:
#                     continue  # skip if no valid region

#                 bboxes = torch.stack(bboxes)

#                 with torch.no_grad():
#                     embeddings, soft_masks, _, _ = model(images_weak, bboxes.unsqueeze(0))

#                 sof_mask_prob = torch.sigmoid(torch.stack(soft_masks, dim=0))
#                 entropy_sm = - (sof_mask_prob * torch.log(sof_mask_prob + eps) + (1 - sof_mask_prob) * torch.log(1 - sof_mask_prob + eps))

#                 entropy_means.append(entropy_sm.detach().mean().cpu().item())


#                 _, pred_masks, iou_predictions, _= model(images_strong, prompts)
#                 del _

#                 num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
#                 loss_focal = torch.tensor(0., device=fabric.device)
#                 loss_dice = torch.tensor(0., device=fabric.device)
#                 loss_iou = torch.tensor(0., device=fabric.device)
#                 loss_sim = torch.tensor(0., device=fabric.device)


#                 batch_feats = []  # collect all bbox features in current image



#                 for bbox in bboxes:
#                     feat = get_bbox_feature(embeddings, bbox)
#                     batch_feats.append(feat)

#                 if len(batch_feats) > 30:
                 
#                     batch_feats = F.normalize(torch.stack(batch_feats, dim=0), dim=1)
#                     loss_sim = similarity_loss(feature_queue , feature_queue)

#                     if loss_sim == -1:
#                         loss_sim = torch.tensor(0., device=batch_feats.device)
              
#                     # add new features to queue (detach to avoid backprop)
#                     for f in batch_feats:
#                         feature_queue.append(f.detach())
#                 else:
#                     loss_sim = torch.tensor(0., device=embeddings.device)

        

#                 for i, (pred_mask, soft_mask, iou_prediction, bbox) in enumerate(
#                         zip(pred_masks[0], soft_masks[0], iou_predictions[0], bboxes  )
#                     ):
#                         soft_mask = (soft_mask > 0.).float()
#                         # print(overlap_map.shape, pred_mask.shape, soft_mask.shape)
#                         # pred_mask = pred_mask * invert_overlap_map[0]
#                         # soft_mask = soft_mask * invert_overlap_map[0]
                        
#                         # plt.imshow(pred_mask.detach().cpu().numpy(), cmap='viridis')
#                         # plt.show()
#                         # plt.imshow(soft_mask.detach().cpu().numpy(), cmap='viridis')
#                         # plt.show()
#                         # Apply entropy mask to losses
#                         loss_focal += focal_loss(pred_mask, soft_mask)  #, entropy_mask=entropy_mask
#                         loss_dice += dice_loss(pred_mask, soft_mask)   #, entropy_mask=entropy_mask
#                         batch_iou = calc_iou(pred_mask.unsqueeze(0), soft_mask.unsqueeze(0))
#                         loss_iou += F.mse_loss(iou_prediction.view(-1), batch_iou.view(-1), reduction='sum') / num_masks

#                 del  pred_masks, iou_predictions 
#                 del pred_stack, overlap_map, invert_overlap_map
#                 # loss_dist = loss_dist / num_masks
#                 loss_dice = loss_dice / num_masks
#                 loss_focal = loss_focal / num_masks
#                 loss_sim  = loss_sim
             

#                 loss_total =  (20 * loss_focal +  loss_dice  + loss_iou +0.1*loss_sim    )    #
#                 if watcher.is_outlier(loss_total):
#                     continue
#                 fabric.backward(loss_total)

#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#                 torch.cuda.empty_cache()
#                 del  prompts, soft_masks

#                 torch.cuda.synchronize()
#                 curr_mem = torch.cuda.memory_allocated() / 1024**3  # GB
#                 iter_mem_usage.append(curr_mem)

#                 batch_time.update(time.time() - end)
#                 end = time.time()

#                 focal_losses.update(loss_focal.item(), batch_size)
#                 dice_losses.update(loss_dice.item(), batch_size)
#                 iou_losses.update(loss_iou.item(), batch_size)
#                 total_losses.update(loss_total.item(), batch_size)
#                 sim_losses.update(loss_sim.item(), batch_size)
            

#             if (iter+1) % match_interval==0:
         
#                 fabric.print(
#                     f"Epoch [{epoch}] Iter [{iter + 1}/{len(train_dataloader)}] " f"| Time {batch_time.avg:.2f}s "
#                     f"| Focal {focal_losses.avg:.4f} | Dice {dice_losses.avg:.4f} | "
#                     f"IoU {iou_losses.avg:.4f} | Sim_loss {sim_losses.avg:.4f} | Total {total_losses.avg:.4f}"
#                 )
#             if (iter+1) % eval_interval == 0:
#                 avg_mem = sum(iter_mem_usage) / len(iter_mem_usage)
#                 # peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
#                 print(f"Average Memory {avg_mem} GB ")
#                 avg_means, _ = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
#                 # avg_means = sum(entropy_means) / len(entropy_means)
#                 status = ""
#                 if avg_means > 0:  #best_ent
#                     best_ent = avg_means
#                     best_state = copy.deepcopy(model.state_dict())
#                     torch.save(best_state, os.path.join(cfg.out_dir, "save", "best_model.pth"))
#                     status = "Improved → Model Saved"
#                     no_improve_count = 0
#                 else:
#                     model.load_state_dict(best_state)
#                     no_improve_count += 1
#                     status = f"Rollback ({no_improve_count})"

#                 # Write log entry
#                 with open(csv_path, "a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([epoch, iter + 1, avg_means, best_ent, status])

#                 fabric.print(f"Validation IoU={avg_means:.4f} | Best={best_ent:.4f} | {status}")

#                 # Stop if model fails to stabilize
#                 if no_improve_count >= max_patience:
#                     fabric.print(f"Training stopped early after {no_improve_count} failed rollbacks.")
#                     return


analyze = True
def train_sam(cfg: Box, fabric: L.Fabric, model: Model, optimizer: _FabricOptimizer,
              scheduler: _FabricOptimizer, train_dataloader: DataLoader, val_dataloader: DataLoader):

    watcher = LossWatcher(window=50, factor=4)
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    best_state = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    max_patience = cfg.get("patience", 3)
    match_interval = cfg.match_interval
    eval_interval = len(train_dataloader)

    # embedding_queue = []
    iter_mem_usage = []

    os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, "training_log.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Iteration", "Val_IoU", "Status"])

    fabric.print(f"Training enabled. Logging to: {csv_path}")

    eps = 1e-8
    # entropy_means = deque(maxlen=len(train_dataloader))
    step_size = 50
    if analyze:
        analyze_indices = set(random.sample(range(len(train_dataloader.dataset)), 50))
        iou_diff_list = []
    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        sim_losses = AverageMeter()
        end = time.time()



        for iter, data in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, img_paths = data
            del data

            
            
            for j in range(0, len(gt_masks[0]), step_size):
                gt_masks_new = gt_masks[0][j:j+step_size].unsqueeze(0)
                prompts = get_prompts(cfg, bboxes, gt_masks_new)
                batch_size = images_weak.size(0)

                entropy_maps, preds = process_forward(images_weak, prompts, model)
                pred_stack = torch.stack(preds, dim=0)
                entropy_maps = torch.stack(entropy_maps, dim=0)

                pred_binary = (((1 - entropy_maps) * pred_stack) > 0.3).float()
                overlap_map = (pred_binary.sum(dim=0) > 1).float()
                invert_overlap_map = 1.0 - overlap_map

                valid_bboxes = []
                for i, (pred, ent) in enumerate(zip(pred_binary, entropy_maps)):
                    pred_w_overlap = pred[0] * invert_overlap_map[0]
                    ys, xs = torch.where(pred_w_overlap > 0.5)
                    if len(xs) > 0 and len(ys) > 0:
                        valid_bboxes.append(torch.tensor([xs.min().item(), ys.min().item(),
                                                          xs.max().item(), ys.max().item()],
                                                         dtype=torch.float32))
                if not valid_bboxes:
                    continue
                bboxes = torch.stack(valid_bboxes)

                with torch.no_grad():
                    embeddings, soft_masks, _, _ = model(images_weak, bboxes.unsqueeze(0))

                
             

                hard_embeddings, pred_masks, iou_predictions, _ = model(images_strong, prompts)
                del _

                num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
                loss_focal = torch.tensor(0., device=fabric.device)
                loss_dice = torch.tensor(0., device=fabric.device)
                loss_iou = torch.tensor(0., device=fabric.device)
                loss_sim = torch.tensor(0., device=fabric.device)


                batch_feats = [get_bbox_feature(embeddings, bbox) for bbox in bboxes]
                batch_feats_hard = [get_bbox_feature(hard_embeddings, bbox) for bbox in bboxes]
                
                if len(feature_queue) == 32:
                    batch_feats = F.normalize(torch.stack(batch_feats, dim=0), dim=1)
                    batch_feats_hard = F.normalize(torch.stack(batch_feats_hard, dim=0), dim=1)
                    loss_sim = similarity_loss(batch_feats_hard,feature_queue)
                    loss_sim = torch.tensor(0., device=batch_feats.device) if loss_sim == -1 else loss_sim
                    feature_queue.extend([f.detach() for f in batch_feats])
                else:
                    batch_feats = F.normalize(torch.stack(batch_feats, dim=0), dim=1)
                    feature_queue.extend([f.detach() for f in batch_feats])
                    
                    loss_sim = torch.tensor(0., device=fabric.device)

                 
                

                for pred_mask, soft_mask, iou_prediction, bbox in zip(pred_masks[0], soft_masks[0], iou_predictions[0], bboxes):
                    soft_mask = (soft_mask > 0.).float()
                    loss_focal += focal_loss(pred_mask, soft_mask)
                    loss_dice += dice_loss(pred_mask, soft_mask)
                    loss_iou += F.mse_loss(iou_prediction.view(-1), calc_iou(pred_mask.unsqueeze(0), soft_mask.unsqueeze(0)).view(-1),
                                           reduction='sum') / num_masks

                if analyze:
                    
                  
                    gt_masks_bin = (gt_masks_new[0] > 0.5).float()
                    soft_masks_sig = torch.sigmoid(soft_masks[0])
                    soft_masks_sig = (soft_masks_sig > 0.5).float()

                    pred_stack_s  = pred_stack.squeeze(1)
                    pred_masks_sig = (pred_stack_s > 0.5).float()

                    if pred_masks_sig.shape[0] ==soft_masks_sig.shape[0]
                        iou_pred = calculate_iou(gt_masks_bin, pred_masks_sig).item()
                        iou_soft = calculate_iou(gt_masks_bin, soft_masks_sig).item()

                        # Difference: positive if pred_stack improves over soft_mask
                        iou_diff = iou_soft - iou_pred
                        iou_diff_list.append(iou_diff)
                        print(iou_diff)
           


                loss_focal /= num_masks
                loss_dice /= num_masks

                loss_total = 20 * loss_focal + loss_dice + loss_iou + 0.1 * loss_sim
                if watcher.is_outlier(loss_total):
                    continue

                fabric.backward(loss_total)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                curr_mem = torch.cuda.memory_allocated() / 1024**3
                iter_mem_usage.append(curr_mem)

                batch_time.update(time.time() - end)
                end = time.time()

                focal_losses.update(loss_focal.item(), batch_size)
                dice_losses.update(loss_dice.item(), batch_size)
                iou_losses.update(loss_iou.item(), batch_size)
                total_losses.update(loss_total.item(), batch_size)
                sim_losses.update(loss_sim.item(), batch_size)

            if analyze:
                if iter not in analyze_indices:
                    save_analyze_images(
                        img_paths,                    
                        gt_masks_new,  
                        pred_stack, 
                        soft_masks,                     
                        bboxes,                     
                        os.path.join(cfg.out_dir, "analyze"),
                        index=iter
                    )

            if (iter + 1) % match_interval == 0:
                fabric.print(
                    f"Epoch [{epoch}] Iter [{iter + 1}/{len(train_dataloader)}] "
                    f"| Time {batch_time.avg:.2f}s | Focal {focal_losses.avg:.4f} | Dice {dice_losses.avg:.4f} | "
                    f"IoU {iou_losses.avg:.4f} | Sim_loss {sim_losses.avg:.4f} | Total {total_losses.avg:.4f}"
                )

            if (iter + 1) % eval_interval == 0:
                
                avg_means, _ = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(cfg.out_dir, "save", "best_model.pth"))
                status = "Model Saved"
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, iter + 1, avg_means, status])
                avg_mem = sum(iter_mem_usage) / len(iter_mem_usage)
                print(f"Average Memory {avg_mem:.2f} GB")
                fabric.print(f"Validation IoU={avg_means:.4f}  | {status}")

                if analyze:
                    iou_diff_tensor = torch.tensor(iou_diff_list)
                    num_positive = (iou_diff_tensor > 0).sum().item()
                    num_negative = (iou_diff_tensor < 0).sum().item()
                    percent_improved = 100 * num_positive / (num_positive + num_negative + 1e-8)
                    print(f"Percentage of mask improved (pred_stack vs soft_mask): {percent_improved:.2f}%")


            
def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler



def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.out_name = corrupt
        torch.cuda.empty_cache()
        main(cfg)



def main(cfg: Box) -> int:

    gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data, pt_data = load_datasets(cfg, img_size=1024, return_pt = True)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    pt_data = fabric._setup_dataloader(pt_data)
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    # auto_ckpt = None#_find_latest_checkpoint(os.path.join(cfg.out_dir, "save"))

    
    # if auto_ckpt is not None:
    #     full_checkpoint = fabric.load(auto_ckpt)

    #     if isinstance(full_checkpoint, dict) and "model" in full_checkpoint:
    #         model.load_state_dict(full_checkpoint["model"])
    #         if "optimizer" in full_checkpoint:
    #             optimizer.load_state_dict(full_checkpoint["optimizer"])
    #     else:
    #         model.load_state_dict(full_checkpoint)
    #     loaded = True
    #     fabric.print(f"Resumed from explicit checkpoint: {cfg.model.ckpt}")
   

    # print('-'*100)
    # print('\033[92mDirect test on the original SAM.\033[0m') 
    # init_iou, _, = validate(fabric, cfg, model, val_data, name=cfg.name, epoch=0)
    # print('-'*100)
    # del _     

    
    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)

    del model, train_data, val_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--prompt', help='the type of prompt')
    parser.add_argument('--num_points',type=int, help='the number of points')
    parser.add_argument('--out_dir', help='the dir to save logs and models')
    parser.add_argument('--load_type', help='the dir to save logs and models')      
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_args()

    exec(f'from {args.cfg} import cfg')

    # transfer the args to a dict
    args_dict = vars(args)
    cfg.merge_update(args_dict)
    print(cfg.model.backend)

    if cfg.model.backend == 'sam':
        main(cfg)

    torch.cuda.empty_cache()




















