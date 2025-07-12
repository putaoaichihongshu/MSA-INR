import argparse
import yaml
import os
import csv
import numpy as np
from skimage import data
import skimage
print(skimage.__version__)
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import trange
import ast

# ====== 注册你的模型工厂 ======
from models import siren, msa, fourier, rowdy, fr

MODEL_FACTORY = {
    "siren": siren.get_model,
    "msa": msa.get_model,
    "fourier": fourier.get_model,
    "rowdy": rowdy.get_model,
    "fr": fr.get_model,
}

# ====== 数据集读取 ======
def get_images(img_names):
    # 支持多图片名，包括彩色
    img_dict = {
        "camera": img_as_float(data.camera()),
        "coins": img_as_float(data.coins()),
        "moon": img_as_float(data.moon()),
        "clock": img_as_float(data.clock()),
        "phantom": img_as_float(data.shepp_logan_phantom()),
        "astronaut": img_as_float(data.astronaut()),  # 彩色图片
    }
    imgs = [img_dict[n] for n in img_names]
    return imgs, img_names

def image_to_dataset(img):
    H, W = img.shape[:2]
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)


    if img.ndim == 2:  # 灰度
        values = img.flatten()[:, None]
    else:  # 彩色
        values = img.reshape(-1, img.shape[2])
    return coords, values

# ====== 训练与评估 ======
def train(model, x, y, lr=1e-3, epochs=3000, device="cuda", lr_min=5e-7):
    model = model.to(device)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    loss_fn = nn.MSELoss()
    losses = []
    bar = trange(epochs)
    for epoch in bar:
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if epoch % 20 == 0 or epoch == epochs - 1:
            bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    model.eval()
    with torch.no_grad():
        y_hat = model(x).cpu().numpy()
    return y_hat, losses

def psnr_np(img1, img2, data_range=1.0):
    """
    img1, img2: numpy arrays, shape (H,W) or (H,W,C), value in [0,1] or [0,255]
    data_range: 一般为1.0（如果你的像素归一化），或255（若是8位图片）
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)

def evaluate(img_true, img_pred):
    img_true = np.clip(img_true, 0.0, 1.0)
    img_pred = np.clip(img_pred, 0.0, 1.0)


    min_side = min(img_true.shape[0], img_true.shape[1])
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    # 用自定义psnr替代库函数
    psnr_score = psnr_np(img_true, img_pred, data_range=1)
    # 新版ssim只支持channel_axis
    if img_true.ndim == 3:
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(img_true, img_pred, data_range=1, channel_axis=2, win_size=win_size)
    else:
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(img_true, img_pred, data_range=1, win_size=win_size)
    return psnr_score, ssim_score










# ====== 加载yaml+命令行参数，自动类型转换 ======
def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="YAML配置文件")
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

def main():
    args, cli_kwargs = parse_args()
    cfg = load_config(args.config)
    merged = {**cfg, **cli_kwargs}

    # 自动把 s_min, s_max 之类的列表类型从字符串转为列表
    for k in merged:
        if isinstance(merged[k], str) and merged[k].startswith("["):
            try:
                merged[k] = ast.literal_eval(merged[k])
            except:
                pass

    model_name = merged['model']
    img_names = merged.get('images', ['astronaut'])  # 默认彩色

    if isinstance(img_names, str):
        img_names = [img_names]

    imgs, names = get_images(img_names)
    device = merged.get('device', "cuda" if torch.cuda.is_available() else "cpu")

    # 构造模型参数
    model_func = MODEL_FACTORY[model_name]
    train_keys = {'epochs', 'lr', 'lr_min', 'seed', 'images', 'model'}
    model_kwargs = {k: v for k, v in merged.items() if k not in train_keys}

    # 设置随机种子
    if 'seed' in merged:
        import random
        torch.manual_seed(merged['seed'])
        np.random.seed(merged['seed'])
        random.seed(merged['seed'])

    # 结果文件夹
    result_root = "results_imagefit"
    os.makedirs(result_root, exist_ok=True)

    for name, img in zip(names, imgs):
        print(f"Training {model_name} on {name} ...")
        x, y = image_to_dataset(img)
        model = model_func(**model_kwargs)
        y_pred, losses = train(model, x, y, lr=float(merged.get('lr', 1e-3)), epochs=int(merged.get('epochs', 5000)), device=device, lr_min=float(merged.get('lr_min', 1e-7)))
        img_pred = y_pred.reshape(img.shape)
        p, s = evaluate(img, img_pred)

        # 结果唯一目录
        save_dir = os.path.join(result_root, f"{name}__{model_name}")
        os.makedirs(save_dir, exist_ok=True)

        # 保存loss曲线为csv（含epoch列）
        lossfile = os.path.join(save_dir, f"loss.csv")
        with open(lossfile, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for epoch, v in enumerate(losses):
                writer.writerow([epoch, v])

        # 保存最终评估指标
        csvfile = os.path.join(save_dir, f"metrics.csv")
        with open(csvfile, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["PSNR", "SSIM", "FinalLoss"])
            writer.writerow([p, s, losses[-1]])

        # 保存重建图片（自动判断彩色或灰度）
        recon_imgfile = os.path.join(save_dir, f"recon.png")
        if img.ndim == 3:
            img_pred = np.clip(img_pred, 0, 1)
            plt.imsave(recon_imgfile, img_pred)
        else:
            plt.imsave(recon_imgfile, img_pred, cmap="gray")

        # 保存一份参数快照
        yamlfile = os.path.join(save_dir, "config_used.yaml")
        with open(yamlfile, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f)

        print(f"Done: {model_name} {name} | PSNR={p:.2f}, SSIM={s:.4f}, FinalLoss={losses[-1]:.4e}")

    print("所有训练与评估已完成，结果已保存在 results_imagefit 文件夹下。")

if __name__ == "__main__":
    main()
