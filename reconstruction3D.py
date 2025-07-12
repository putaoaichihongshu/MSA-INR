import argparse
import yaml
import os
import csv
import numpy as np
from scipy import io, ndimage
import torch
import tqdm, copy, ast
from utlis import get_coords, get_IoU, march_and_save
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

def train_model(model, coords_tensor, sdf_tensor, epochs, batch_size, learning_rate):
    device = next(model.parameters()).device
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lambda x: 0.2**min(x/epochs, 1))

    num_points = coords_tensor.shape[0]
    best_mse = float('inf')
    best_result = None
    losses = []

    im_estim = torch.zeros((num_points, 1), device=device)

    for epoch in tqdm.tqdm(range(epochs)):
        indices = torch.randperm(num_points)
        epoch_loss = 0
        nchunks = 0

        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            batch_indices = indices[start:end]
            batch_coords = coords_tensor[batch_indices]
            batch_sdf = sdf_tensor[batch_indices]

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = criterion(preds, batch_sdf)
            loss.backward()
            optimizer.step()

            im_estim[batch_indices, :] = preds.detach()
            epoch_loss += loss.item()
            nchunks += 1

        epoch_loss /= nchunks
        losses.append(epoch_loss)

        scheduler.step()

        if epoch_loss < best_mse:
            best_mse = epoch_loss
            best_result = copy.deepcopy(im_estim)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    return model, losses, best_result

def main():
    args, cli_kwargs = parse_args()
    cfg = load_config(args.config)
    merged = {**cfg, **cli_kwargs}
    # 自动把列表参数转为list
    for k in merged:
        if isinstance(merged[k], str) and merged[k].startswith("["):
            try:
                merged[k] = ast.literal_eval(merged[k])
            except:
                pass

    data_path = merged.get('data_path', 'data/dragon.mat')
    scale = float(merged.get('scale', 1.0))
    mcubes_thres = float(merged.get('mcubes_thres', 0.5))
    model_name = merged['model']
    epochs = int(merged.get('epochs', 200))
    lr = float(merged.get('lr', 5e-3))
    batch_size = int(merged.get('batch_size', 500000))

    # 结果目录
    base_tag = f"{os.path.splitext(os.path.basename(data_path))[0]}__{model_name}"
    result_dir = os.path.join("results_3drecon", base_tag)
    os.makedirs(result_dir, exist_ok=True)

    # 1. 加载与缩放数据
    data = io.loadmat(data_path)['hypercube'].astype(np.float32)
    data = ndimage.zoom(data / data.max(), [scale, scale, scale], order=0)

    # 2. 裁剪到最小bounding box
    hidx, widx, tidx = np.where(data > 0.99)
    data = data[hidx.min():hidx.max(),
                widx.min():widx.max(),
                tidx.min():tidx.max()]
    H, W, T = data.shape

    # 3. 全量采样
    dataten = torch.tensor(data).cuda().reshape(H * W * T, 1)
    coords = get_coords(H, W, T).cuda()
    # 不做任何采样下标选择，直接用全部点！

    # 4. 构造模型（所有参数来自yaml/命令行）
    model_func = MODEL_FACTORY[model_name]
    train_keys = {'epochs', 'lr', 'batch_size', 'seed', 'data_path', 'scale', 'mcubes_thres', 'model'}
    model_kwargs = {k: v for k, v in merged.items() if k not in train_keys}
    model = model_func(**model_kwargs).cuda()

    if 'seed' in merged:
        import random
        torch.manual_seed(merged['seed'])
        np.random.seed(merged['seed'])
        random.seed(merged['seed'])

    # 5. 训练模型
    trained_model, losses, best_result = train_model(
        model, coords, dataten, epochs=epochs, batch_size=batch_size, learning_rate=lr
    )

    # 6. 还原3D体素并保存
    best_result_np = best_result.reshape(H, W, T).detach().cpu().numpy()
    io.savemat(os.path.join(result_dir, f'{model_name}_result.mat'), {'volume': best_result_np})

    # 7. 保存loss到csv
    loss_csv_path = os.path.join(result_dir, f'{model_name}_loss.csv')
    with open(loss_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'loss'])
        for i, loss in enumerate(losses, 1):
            writer.writerow([i, loss])
    print(f"Losses saved to {loss_csv_path}")

    # 8. 计算IoU并保存
    iou = get_IoU(best_result_np, data, mcubes_thres)
    print(f'Final IoU: {iou:.4f}')
    iou_path = os.path.join(result_dir, f'iou_{model_name}.txt')
    with open(iou_path, 'w') as f:
        f.write(f'IoU: {iou:.6f}\n')

    # 9. Marching Cubes保存.obj
    savename = os.path.join(result_dir, f'{model_name}_reconstruction.obj')
    march_and_save(best_result_np, mcubes_thres, savename, True)

    # 10. 保存本次config快照
    with open(os.path.join(result_dir, 'config_used.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(merged, f)

    print(f"全部结果已保存在 {result_dir}")

if __name__ == '__main__':
    main()
