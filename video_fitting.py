import argparse
import yaml
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import skvideo.io
import imageio
import ast

# ====== 导入你的模型（请补全注册自己所有模型）======
from models.fr import get_model as get_fr_model
from models.siren import get_model as get_siren_model
from models.msa import get_model as get_msa_model
from models.fourier import get_model as get_fourier_model
from models.rowdy import get_model as get_rowdy_model

MODEL_FACTORY = {
    'fr': get_fr_model,
    'siren': get_siren_model,
    'msa': get_msa_model,
    'fourier': get_fourier_model,
    'rowdy': get_rowdy_model
    # ...如有其它模型继续添加
}

np.float = float
np.int = int
np.bool = bool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, unknown = parser.parse_known_args()
    cli_kwargs = {}
    key = None
    for item in unknown:
        if item.startswith('--'):
            key = item.lstrip('-')
        elif key:
            val = item
            if isinstance(val, str) and val.startswith("["):
                try:
                    cli_kwargs[key] = ast.literal_eval(val)
                except:
                    cli_kwargs[key] = val
            else:
                try:
                    cli_kwargs[key] = float(val) if '.' in val or 'e' in val or 'E' in val else int(val)
                except:
                    cli_kwargs[key] = val
            key = None
    return args, cli_kwargs

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def psnr(img1, img2, data_range=255):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def get_video_coords(video_path, max_frames=100):
    videodata = skvideo.io.vread(video_path)
    videodata = videodata[:max_frames]
    T, H, W, C = videodata.shape
    coords = []
    for t in range(T):
        for y in range(H):
            for x in range(W):
                coords.append([x / (W-1), y / (H-1), t / (T-1)])
    return np.array(coords, dtype=np.float32), videodata

def reconstruct_video(model, coords, shape, device="cuda", batch_size=20480):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch = torch.tensor(coords[i:i+batch_size], dtype=torch.float32).to(device)
            batch_pred = model(batch).cpu().numpy()
            preds.append(batch_pred)
    preds = np.concatenate(preds, axis=0)
    T, H, W = shape
    preds = preds.reshape((T, H, W, 3))
    preds = np.clip(preds, 0, 1)
    preds = (preds * 255).astype(np.uint8)
    return preds

def evaluate_psnr(gt, pred):
    psnr_list = []
    for i in range(gt.shape[0]):
        gt_frame = gt[i]
        pred_frame = pred[i]
        psnr_val = psnr(gt_frame, pred_frame, data_range=255)
        psnr_list.append(psnr_val)
    psnr_arr = np.array(psnr_list)
    return psnr_arr.mean(), psnr_arr.std(), psnr_arr

def train(model, train_loader, num_epochs=500, lr=1e-3, lr_min=1e-6, save_dir=None, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)
    loss_fn = nn.MSELoss()
    loss_log = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        avg_loss = np.mean(losses)
        loss_log.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch}/{num_epochs} Loss={avg_loss:.6e} LR={optimizer.param_groups[0]['lr']:.2e}")
    # 保存模型和loss
    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "loss.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for ep, v in enumerate(loss_log, 1):
                writer.writerow([ep, v])
    return model, loss_log

def main():
    args, cli_kwargs = parse_args()
    cfg = load_config(args.config)
    merged = {**cfg, **cli_kwargs}
    # 自动把 s_min, s_max 等列表类型转为 list
    for k in merged:
        if isinstance(merged[k], str) and merged[k].startswith("["):
            try:
                merged[k] = ast.literal_eval(merged[k])
            except:
                pass

    model_name = merged["model"]
    video_path = merged.get("video_path", "bbb.mp4")
    max_frames = int(merged.get("max_frames", 100))
    batch_size = int(merged.get("batch_size", 20480))
    device = merged.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_frames = merged.get("save_frames", [24, 49, 74, 99])

    coords, gt_videodata = get_video_coords(video_path, max_frames=max_frames)
    T, H, W, C = gt_videodata.shape
    rgbs = gt_videodata.reshape(-1, 3) / 255.0

    tensor_coords = torch.tensor(coords, dtype=torch.float32)
    tensor_rgbs = torch.tensor(rgbs, dtype=torch.float32)
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(tensor_coords, tensor_rgbs),
        batch_size=batch_size, shuffle=True, num_workers=0
    )

    # 构造模型
    model_func = MODEL_FACTORY[model_name]
    train_keys = {"num_epochs", "lr", "lr_min", "batch_size", "seed", "video_path", "max_frames", "model", "save_frames"}
    model_kwargs = {k: v for k, v in merged.items() if k not in train_keys}
    model = model_func(**model_kwargs)

    if "seed" in merged:
        import random
        torch.manual_seed(merged["seed"])
        np.random.seed(merged["seed"])
        random.seed(merged["seed"])

    result_dir = f"results_video/{os.path.splitext(os.path.basename(video_path))[0]}__{model_name}"
    os.makedirs(result_dir, exist_ok=True)

    # 训练
    model, loss_log = train(
        model, train_loader,
        num_epochs=int(merged.get("num_epochs", 500)),
        lr=float(merged.get("lr", 1e-3)),
        lr_min=float(merged.get("lr_min", 1e-6)),
        save_dir=result_dir,
        device=device
    )

    # --- 重建并评估 ---
    model.load_state_dict(torch.load(os.path.join(result_dir, "model.pt")))
    model = model.to(device)
    preds = reconstruct_video(model, coords, (T, H, W), device=device)
    mean_psnr, std_psnr, psnr_arr = evaluate_psnr(gt_videodata, preds)
    psnr_log_path = os.path.join(result_dir, "psnr_log.csv")
    with open(psnr_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "psnr"])
        for i, val in enumerate(psnr_arr):
            writer.writerow([i, val])
        writer.writerow(["mean", mean_psnr])
        writer.writerow(["std", std_psnr])
    print(f"{model_name}: Mean PSNR={mean_psnr:.2f}, Std={std_psnr:.2f}")

    # 保存指定帧的GT和重建
    frame_dir = os.path.join(result_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for idx in save_frames:
        if idx < T:
            gt_frame = gt_videodata[idx]
            pred_frame = preds[idx]
            imageio.imwrite(os.path.join(frame_dir, f"gt_frame_{idx + 1:03d}.png"), gt_frame)
            imageio.imwrite(os.path.join(frame_dir, f"pred_frame_{idx + 1:03d}.png"), pred_frame)

    # 保存一份config参数快照
    with open(os.path.join(result_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f)

    print(f"Done. Results in {result_dir}")

if __name__ == "__main__":
    main()
