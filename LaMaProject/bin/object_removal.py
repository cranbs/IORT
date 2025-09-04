import os
import yaml
import torch
import numpy as np
import cv2
from omegaconf import OmegaConf
from LaMaProject.saicinpainting.evaluation.utils import move_to_device
from LaMaProject.saicinpainting.training.trainers import load_checkpoint
from LaMaProject.saicinpainting.evaluation.refinement import refine_predict

def load_lama_model(model_path, checkpoint_name="best.ckpt"):
    """
    加载 LaMa 模型
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_config_path = os.path.join(model_path, "config.yaml")
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(model_path, "models", checkpoint_name)
    model = load_checkpoint(train_config, checkpoint_path, 
                            strict=False, map_location=device)
    model.freeze()
    model.to(device)
    return model, train_config

def inpaint(image: np.ndarray, mask: np.ndarray, model, refinement=False, out_key="inpainted"):
    """
    修复单张图片接口
    Args:
        image: HWC, RGB, uint8
        mask:  HWC 或 HW, 单通道二值 (0=保留, 255=需要修复)
        model: 已加载的 LaMa 模型
        device: "cpu" 或 "cuda"
        out_key: 输出关键字
    Returns:
        result: HWC, RGB, uint8
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if mask.ndim == 3:  # 转单通道
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.uint8)

    # 转 tensor 格式 (N, C, H, W)
    img_tensor = torch.from_numpy(image.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    batch = {"image": img_tensor, "mask": mask_tensor}
    batch = move_to_device(batch, device)

    with torch.no_grad():
        if refinement:
            batch = model(batch)
            cur_res = refine_predict(batch, model, gpu_ids=0, modulo=8, n_iters=15, lr=0.002, min_side=512, max_scales=3, px_budget=1800000)
            cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
        else:
            batch = model(batch)
            cur_res = batch[out_key][0].permute(1, 2, 0).detach().cpu().numpy()

    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    return cur_res
