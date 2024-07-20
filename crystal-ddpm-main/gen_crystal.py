import torch
from model_diff3d_v1 import UNet3D
from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import MinimumDistanceNN
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")


GRID_SIZE = 64
# NUM_ATOM = 84
# 初始化模型
device = torch.device('cpu')#("cuda" if torch.cuda.is_available() else 'cpu')
model = UNet3D().to(device)

# 加载训练好的模型权重
model_save_path = "best_diff_model.pth"
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# 定义扩散过程的时间步数
T = 1000

# 定义一个简单的线性噪声调度器
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

# 定义反向扩散过程
def reverse_diffusion(x, timesteps):
    for t in reversed(range(timesteps)):
        t = torch.tensor([t]).to(device)
        sqrt_recip_alpha_cumprod_t = sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1, 1).to(device)
        sqrt_recipm1_alpha_cumprod_t = sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1, 1).to(device)
        # 预测噪声
        noise_pred = model(x)
        # 计算无噪声的图像
        x = sqrt_recip_alpha_cumprod_t * (x - sqrt_recipm1_alpha_cumprod_t * noise_pred)
        # 如果不是最后一步，则添加噪声
        if t > 0:
            noise = torch.randn_like(x).to(device)
            x += betas[t].to(device).view(-1, 1, 1, 1, 1) * noise
    return x

# 生成新的数据
def generate_samples(model, num_samples, timesteps):
    model.eval()
    with torch.no_grad():
        # 从随机噪声开始
        x = torch.randn(num_samples, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE).to(device)
        # 通过反向扩散过程生成数据
        generated_samples = reverse_diffusion(x, timesteps)
    return generated_samples

# 生成样本
num_samples = 1  # 生成的样本数量
generated_samples = generate_samples(model, num_samples, T)

# 打印生成的样本
print(generated_samples.shape) # torch.Size([1, 1, 64, 64, 64])
generated_grid = generated_samples.squeeze().numpy()
print(generated_grid.shape)
generated_grid = np.nan_to_num(generated_grid, nan=0)
generated_grid = np.round(generated_grid).astype(int)
print(generated_grid.shape)
# print(generated_samples.shape) # torch.Size([1, 1, 64, 64, 64])

def grid_to_structure(grid, grid_size=GRID_SIZE, lattice=None):  # 使用相同的网格分辨率
    frac_coords = []
    atom_types = []
    x, y, z = np.nonzero(grid)
    print('x,y,z : ',x,y,z)
    for i in range(len(x)):
        frac_coords.append([x[i] / (grid_size - 1), y[i] / (grid_size - 1), z[i] / (grid_size - 1)])
        atom_types.append(Element.from_Z(grid[x[i], y[i], z[i]]))

    if lattice is None:
        lattice = Lattice.cubic(10.0)  # 可以调整晶胞参数
    structure = Structure(lattice, atom_types, frac_coords)
    return structure


type_ = 'P2'
# 读取CIF文件并提取晶体结构
cif_file_path = f"{type_}_NaNiO2.cif"
structure = Structure.from_file(cif_file_path)
if type_ == 'O3':
    structure.make_supercell([3, 2, 1])
if type_ == 'P2':
    structure.make_supercell([3, 1, 1])

structure_from_grid = grid_to_structure(generated_grid, grid_size=GRID_SIZE, lattice=structure.lattice)

# 将结构保存为CIF文件
structure_from_grid.to(filename="generated_p2structure.cif")

