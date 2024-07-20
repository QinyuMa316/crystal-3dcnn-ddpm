import os
from pymatgen.core import Structure
from model_diff3d_v1 import UNet3D
import torch.optim as optim
from torch import nn
from dataloader_cov3d import CrystalDataset
from torch.utils.data import DataLoader
import torch
import json
import time
def get_struct_list_from_struct_dict(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        entry_dict = json.load(file)
    struct_list = []
    for key, struct_dict in entry_dict.items():
        struct_list.append(Structure.from_dict(struct_dict))
    return struct_list


if __name__ == '__main__':
    # structure = Structure.from_file("str2grid_reverse/P2_NaNiO2.cif")
    # structure.make_supercell([3, 1, 1])
    # structures = [structure for _ in range(100)]  # 替换为实际文件路径

    type_ = 'P2'
    filename = f'{type_}_NaTMO2_2tm_final.json'
    struct_list = get_struct_list_from_struct_dict(filename)
    print(f'Number of structures: {len(struct_list)}')

    structures = []
    for structure in struct_list:
        structure_copy = structure.copy()  # 创建结构的副本
        structure_copy.make_supercell([2, 6, 1]) # [3,1,1]*[2,6,1]=>[6,6,1]
        structures.append(structure_copy)
    del struct_list

    dataset = CrystalDataset(structures)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = UNet3D().to(device)

    model_save_path_old = "best_diff_model.pth"
    model_save_path_new = 'best_diff_model.pth'
    try:
        model.load_state_dict(torch.load(model_save_path_old))
        print("Model loaded successfully, continue training...")
    except FileNotFoundError:
        print("No saved model found, training from scratch...")

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)


    # 定义一个简单的线性噪声调度器
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    # 定义扩散过程的时间步数
    T = 1000
    betas = linear_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)

    # 前向扩散过程（加噪过程）
    def forward_diffusion_sample(x0, t, device=device):
        noise = torch.randn_like(x0).to(device)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).to(device)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).to(device)
        # 调整形状以便广播
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


    # 早停机制参数
    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # 定义训练循环
    epochs = 1000
    print('start training ...')
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            # 随机选择一个时间步
            t = torch.randint(0, T, (x.size(0),)).to(device)
            # 生成加噪样本和对应的噪声
            x_noisy, noise = forward_diffusion_sample(x, t, device)
            # 使用模型预测噪声
            noise_pred = model(x_noisy)
            # x_noisy : torch.Size([32, 1, 32, 32, 32]),
            # noise_pred : torch.Size([32, 1, 32, 32, 32]),
            # noise : torch.Size([32, 1, 32, 32, 32])
            # 计算损失
            epoch_loss = criterion(noise_pred, noise)
            # 反向传播和优化
            epoch_loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_duration = (end_time - start_time)/60
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss.item()}, duration : {epoch_duration} min')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path_new)
            print(f"Model saved at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                early_stop = True
                break

    if not early_stop:
        print('Finished Training')
    else:
        print('Training stopped early due to no improvement')


