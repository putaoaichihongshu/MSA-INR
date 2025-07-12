import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from target_curves import TARGET_DICT
from models import msa, siren, fourier, rowdy, fr # 按需添加其它模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FACTORY = {
    'msa': msa.get_model,
    'siren': siren.get_model,
    'rowdy': rowdy.get_model,
    'fourier': fourier.get_model,
    'fr': fr.get_model
    # ... 注册你的其它模型
}

def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def auto_cast(val):
    if isinstance(val, str):
        try:
            if '.' in val or 'e' in val or 'E' in val:
                return float(val)
            return int(val)
        except:
            return val
    return val


def set_seed(seed):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你有多张卡

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    # 只声明 config，剩余参数全部动态解析
    args, unknown = parser.parse_known_args()
    cli_kwargs = {}
    key = None
    for item in unknown:
        if item.startswith('--'):
            key = item.lstrip('-')
        elif key:
            # 自动转数字，否则原字符串
            try:
                cli_kwargs[key] = float(item) if ('.' in item or 'e' in item or 'E' in item) else int(item)
            except:
                cli_kwargs[key] = item
            key = None
    return args, cli_kwargs

def gen_data(target_func, n_train=3000, x_range=(-1, 1)):
    x_train = np.random.uniform(x_range[0], x_range[1], size=(n_train, 1))
    y_train = target_func(x_train)
    return x_train, y_train

def plot_fit_curve(x, y_true, y_pred, save_path, model_name=""):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y_true, label='Ground Truth', linewidth=2)
    plt.plot(x, y_pred, label=f'{model_name} Prediction', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'Fit Curve: {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def train_and_log(task_name,model_name, model, target_func, x_train, y_train, epochs=5000, lr=1e-3, lr_min=1e-7, log_interval=10):
    save_dir = f"results_curvefit/{task_name}__{model_name}/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    criterion = nn.MSELoss()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    log_records = []
    x_vis = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_true_vis = target_func(x_vis)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                x_vis_tensor = torch.tensor(x_vis, dtype=torch.float32, device=DEVICE)
                y_pred_vis = model(x_vis_tensor).cpu().numpy().squeeze()
            record = {'epoch': epoch, 'train_loss': loss.item()}
            log_records.append(record)
            print(f"[{model_name}] Epoch {epoch}/{epochs}  TrainLoss: {loss.item():.6e}")
    # 日志保存
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(os.path.join(save_dir, "train_log.csv"), index=False)
    # 权重和结果保存
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    plot_fit_curve(x_vis, y_true_vis, y_pred_vis, os.path.join(save_dir, "fit_curve.png"), model_name)
    return log_records

def main():
    args, cli_kwargs = parse_args()
    cfg = load_config(args.config)
    # 合并，命令行参数优先级最高
    merged = {**cfg, **cli_kwargs}

    # 随机种子
    if 'seed' in merged:
        set_seed(merged['seed'])

    import ast
    for k in ['s_min', 's_max']:
        if k in merged and isinstance(merged[k], str) and merged[k].startswith('['):
            merged[k] = ast.literal_eval(merged[k])

    model_name = merged['model']
    target_name = merged['target']

    # 构造目标曲线函数
    if target_name not in TARGET_DICT:
        raise ValueError(f"Unknown target curve: {target_name}")
    target_func = TARGET_DICT[target_name]

    # 模型构造参数
    train_keys = {'epochs', 'lr', 'lr_min', 'batch_size', 'seed', 'target', 'model', 'n_train', 'x_range'}
    model_func = MODEL_FACTORY[model_name]
    model_kwargs = {k: v for k, v in merged.items() if k not in train_keys}
    model = model_func(**model_kwargs).to(DEVICE)



    # 训练数据
    x_train, y_train = gen_data(
        target_func,
        n_train=merged.get('n_train', 3000),
        x_range=merged.get('x_range', (-1, 1))
    )

    # 训练
    train_and_log(
        target_name,
        model_name,
        model,
        target_func,
        x_train, y_train,
        lr=auto_cast(merged.get('lr', 1e-3)),
        lr_min = auto_cast(merged.get('lr_min', 1e-7)),
        epochs = auto_cast(merged.get('epochs', 5000))

    )

if __name__ == "__main__":
    main()
