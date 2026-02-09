import os 
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

"""
Navier–Stokes–Darcy (analytic):
域: Ω = [0,1] × [-0.25, 0.75]
Ω_S = [0,1] × [-0, 0]  (Navier–Stokes 区)
Ω_D = [0,1] × [0, 0.75]   (Darcy 区)
纯数据拟合：不加物理约束

  (1) 编码器仅使用空间信息 x,y 与 mask_D（加空间 Fourier），不再接收时间通道；
  (2) 只用 t=0 帧构造 z0，时间相位不进入编码特征；
  (3) Koopman = 多块 2×2 旋转 A=diag(ω_i J)，ω_i 可学习（初始化 2π），严格谐振，天生稳外推；
"""

# ==================== Global ====================
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

FIG_DIR = 'figs_nsd_multiframe96_rotblock'
os.makedirs(FIG_DIR, exist_ok=True)

# ==================== Config ====================
CFG = dict(
    spatial_size=(96, 96),
    T_train=20,                      # train: t∈[0,1]
    T_eval=40,                       # eval: t∈[0,2]
    temporal_domain_train=(0.0, 1.0),
    temporal_domain_eval=(0.0, 2.0),
    batch_size=16,
    epochs=1000,
    lr=5e-4,
    weight_decay=2e-5,

    amp_tempering=False,             # 关掉 e^{-t}

    # Multi-frame encoder（但本版只用 t=0）
    ref_use_all=False,
    ref_K=1,

    # Loss weights（可按需再调）
    w_u1=3.0,
    w_u2=3.0,
    w_p =1.0,
    w_phi=1.0,

    # Model
    patch_size=(16, 16),
    dim=192,
    depth=6,
    heads=6,
    mlp_dim=384,
    z_dim=192,                       # 必须为偶数（旋转块）
    emb_dropout=0.1,

    # Fourier features (space)
    fourier_freqs=(1, 2, 4, 8),

    # Refine head
    refine=True,
    refine_hidden=32,
    refine_blocks=1,

    # Scheduler
    warmup_epochs=10,
)

# ==================== Exact Solution (NS-D Example 106-108) ====================
def exact_solution_ns_d(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    """
    Example:
      phi_D = [2 - π sin(π x)] * [-y + cos(π(1 - y))] * cos(2π t)
      u_S   = [ x^2 y^2 + e^{-y},
                - (2/3) x y^3 + 2 - π sin(π x) ]^T * cos(2π t)
      p_S   = - [2 - π sin(π x)] * cos(2π y) * cos(2π t)
    """
    two_pi_t = 2.0 * math.pi * t
    cos2pt = torch.cos(two_pi_t)

    pi = math.pi
    pix = pi * x
    sin_pix = torch.sin(pix)
    factor = (2.0 - pi * sin_pix)

    phi_D = factor * (-y + torch.cos(pi * (1.0 - y))) * cos2pt

    u_S1 = (x**2 * y**2 + torch.exp(-y)) * cos2pt
    u_S2 = (-(2.0/3.0) * x * y**3 + 2.0 - pi * sin_pix) * cos2pt

    p_S  = - factor * torch.cos(2.0 * pi * y) * cos2pt
    return u_S1, u_S2, p_S, phi_D

# ==================== Dataset ====================
class PhysicsDataset(Dataset):
    def __init__(self,
                 spatial_size=(96,96), temporal_size=20,
                 spatial_domain=((0.0, 1.0), (-0.25, 0.75)),
                 temporal_domain=(0.0,1.0),
                 num_samples=100,
                 mode='train',
                 channel_stats_external=None,
                 amp_temper=False):
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.num_samples = num_samples
        self.mode = mode
        self.amp_temper = amp_temper

        # 基本网格
        self.x = np.linspace(spatial_domain[0][0], spatial_domain[0][1], spatial_size[0], dtype=np.float32)
        self.y = np.linspace(spatial_domain[1][0], spatial_domain[1][1], spatial_size[1], dtype=np.float32)
        self.t = np.linspace(temporal_domain[0], temporal_domain[1], temporal_size, dtype=np.float32)

        if channel_stats_external is None:
            self.compute_normalization_params()
        else:
            self.channel_stats = channel_stats_external

    def compute_normalization_params(self):
        print('[Dataset] Computing normalization parameters (NS-D)...')
        X, Y, T = np.meshgrid(self.x, self.y, self.t, indexing='ij')
        X_t = torch.tensor(X); Y_t = torch.tensor(Y); T_t = torch.tensor(T)
        u1, u2, p, phi = exact_solution_ns_d(X_t, Y_t, T_t)

        # 掩码：子域
        mask_S = (Y < 0.0).astype(np.float32)
        mask_D = (Y >= 0.0).astype(np.float32)

        u1 = u1 * torch.tensor(mask_S)
        u2 = u2 * torch.tensor(mask_S)
        p  = p  * torch.tensor(mask_S)
        phi = phi * torch.tensor(mask_D)

        # 归一化统计（max & std），确保非零
        def _stat(tensor):
            mx = torch.abs(tensor).max().item()
            sd = tensor.std().item()
            return {'max': max(mx, 1e-6), 'std': max(sd, 1e-6)}

        self.channel_stats = {
            'u1': _stat(u1),
            'u2': _stat(u2),
            'p' : _stat(p),
            'phi': _stat(phi),
        }
        for name, st in self.channel_stats.items():
            print(f"    {name:>3}: max={st['max']:.6f}, std={st['std']:.6f}")

    def normalize(self, u1, u2, p, phi):
        u1_norm = u1 / self.channel_stats['u1']['max']
        u2_norm = u2 / self.channel_stats['u2']['max']
        p_norm  = p  / self.channel_stats['p']['max']
        phi_norm= phi/ self.channel_stats['phi']['max']
        return u1_norm, u2_norm, p_norm, phi_norm

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X, Y, T = np.meshgrid(self.x, self.y, self.t, indexing='ij')

        # 子域掩码
        mask_S = (Y < 0.0).astype(np.float32)
        mask_D = (Y >= 0.0).astype(np.float32)

        # 解析解 & 屏蔽无效区域
        X_t = torch.tensor(X); Y_t = torch.tensor(Y); T_t = torch.tensor(T)
        u1, u2, p, phi = exact_solution_ns_d(X_t, Y_t, T_t)
        u1 = u1.numpy(); u2 = u2.numpy(); p = p.numpy(); phi = phi.numpy()
        u1 *= mask_S; u2 *= mask_S; p *= mask_S; phi *= mask_D

        # 归一化
        u1n, u2n, pn, phin = self.normalize(u1, u2, p, phi)

        # 输入特征（为兼容已有管线仍输出 7 通道）：
        #  - 编码器只会用到 [x_norm, y_norm, mask_D] + 空间Fourier
        #  - 其余 t_norm, sin, cos, t_raw 仅保留，便于 dt 计算与兼容可视化
        X_norm = (X - self.spatial_domain[0][0]) / (self.spatial_domain[0][1] - self.spatial_domain[0][0]) * 2 - 1
        Y_norm = (Y - self.spatial_domain[1][0]) / (self.spatial_domain[1][1] - self.spatial_domain[1][0]) * 2 - 1
        T_norm = (T - self.temporal_domain[0]) / (self.temporal_domain[1] - self.temporal_domain[0] + 1e-12) * 2 - 1
        sin2pt = np.sin(2*np.pi*T).astype(np.float32)
        cos2pt = np.cos(2*np.pi*T).astype(np.float32)
        t_raw  = T.astype(np.float32)

        input_features = np.stack([X_norm, Y_norm, T_norm, mask_D, sin2pt, cos2pt, t_raw], axis=0)  # (7,X,Y,T)
        targets = np.stack([u1n, u2n, pn, phin, mask_S, mask_D], axis=0)  # (6,X,Y,T)
        return torch.tensor(input_features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# ==================== Koopman：多块 2x2 旋转（多频谐振） ====================
class KoopmanRotBlocks(nn.Module):
    """
    A = diag(ω_1 J, ω_2 J, ..., ω_{d/2} J),  J=[[0,-1],[1,0]]
    - z_dim 必须为偶数
    - ω_i 为可学习参数（对数域存储，保证 >0），初始化在 2π 附近
    """
    def __init__(self, z_dim: int, init_omega=2*math.pi):
        super().__init__()
        assert z_dim % 2 == 0, "z_dim must be even for 2x2 rotation blocks"
        self.z_dim = z_dim
        # 逐块频率（更泛化）：shape = (d/2,)
        init = torch.full((z_dim // 2,), float(init_omega))
        self.log_omega = nn.Parameter(torch.log(init))

        # 预构造块对角 J
        J = torch.zeros(z_dim, z_dim)
        for i in range(0, z_dim, 2):
            J[i, i+1] = -1.0
            J[i+1, i] = 1.0
        self.register_buffer('J', J)

        # 每个 2×2 块的选择矩阵，用于把 ω_i 展开到对角块
        # 等价于 A = sum_i ω_i * E_i, E_i=diag(0..J..0)（J放在对应2×2位置）
        E_list = []
        for i in range(0, z_dim, 2):
            E = torch.zeros(z_dim, z_dim)
            E[i, i+1] = -1.0
            E[i+1, i] = 1.0
            E_list.append(E)
        self.register_buffer('E_stack', torch.stack(E_list, dim=0))  # (d/2, d, d)

    def forward(self, z0, dt):
        # 组合 A
        omega = torch.exp(self.log_omega)             # (d/2,)
        # A = sum_i omega_i * E_i
        A = torch.tensordot(omega, self.E_stack, dims=1)  # (d,d)

        BT = dt.reshape(-1)                           # (B*T,)
        K = torch.matrix_exp(BT[:, None, None] * A)   # (B*T,d,d)
        B, T = dt.shape
        d = A.shape[0]
        K = K.view(B, T, d, d)                        # (B,T,d,d)
        z0e = z0.unsqueeze(1).expand(B, T, *z0.shape[1:])  # (B,T,N,d)
        zt = torch.matmul(z0e, K.transpose(-1, -2))        # (B,T,N,d)
        return zt

# ==================== Lightweight Refine Head ====================
class ResidualConvBlock(nn.Module):
    def __init__(self, channels=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)

class RefineHead(nn.Module):
    def __init__(self, channels=4, hidden=32, blocks=1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualConvBlock(channels, hidden) for _ in range(blocks)])

    def forward(self, x):  # x: (B,C,X,Y,T)
        B, C, X, Y, T = x.shape
        y = x.permute(0, 4, 1, 2, 3).reshape(B*T, C, X, Y)  # (B*T,C,X,Y)
        y = self.blocks(y)
        y = y.view(B, T, C, X, Y).permute(0, 2, 3, 4, 1)    # (B,C,X,Y,T)
        return y

# ==================== ViT + Spatial-only encoder + Koopman RotBlocks ====================
class ViT_KNO_MultiFrame(nn.Module):
    def __init__(self,
                 spatial_size=(96, 96),
                 patch_size=(16, 16),
                 dim=192,
                 depth=6,
                 heads=6,
                 mlp_dim=384,
                 in_channels_base=7,                 # 兼容输入形状；编码器只用其中 3 个
                 fourier_freqs=(1,2,4,8),
                 z_dim=192,
                 out_channels=4,
                 emb_dropout=0.1,
                 refine=True,
                 refine_hidden=32,
                 refine_blocks=1):
        super().__init__()
        assert z_dim % 2 == 0, "z_dim must be even for 2x2 rotation blocks"
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.fourier_freqs = tuple(fourier_freqs)

        assert spatial_size[0] % patch_size[0] == 0
        assert spatial_size[1] % patch_size[1] == 0

        self.nx = spatial_size[0] // patch_size[0]
        self.ny = spatial_size[1] // patch_size[1]
        self.num_patches = self.nx * self.ny

        # 空间 Fourier 特征：对 x,y 各加 2*len(freqs) 个通道
        fourier_ch = 2 * len(self.fourier_freqs) * 2
        # 仅用 x,y,mask_D 做空间编码
        self.in_channels_space = 3 + fourier_ch

        patch_dim = self.in_channels_space * patch_size[0] * patch_size[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (nx px) (ny py) -> b (nx ny) (px py c)',
                      px=patch_size[0], py=patch_size[1], nx=self.nx, ny=self.ny),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(emb_dropout),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=emb_dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.to_z = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, z_dim),
        )

        # 旋转块 Koopman
        self.koopman = KoopmanRotBlocks(z_dim=z_dim, init_omega=2*math.pi)

        self.decoder = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, patch_size[0] * patch_size[1] * out_channels),
        )

        self.refine = RefineHead(out_channels, refine_hidden, refine_blocks) if refine else None

    @staticmethod
    def _fourier_features(coord_norm: torch.Tensor, freqs):
        outs = []
        for f in freqs:
            ang = 2 * math.pi * f * coord_norm
            outs += [torch.sin(ang), torch.cos(ang)]
        return torch.cat(outs, dim=1) if outs else torch.zeros_like(coord_norm)

    def forward(self, inputs, ref_use_all=False, ref_K=1):
        """
        inputs: (B,7,X,Y,T) = [x_norm,y_norm,t_norm,mask_D,sin,cos,t_raw]
        本版编码器仅使用 x_norm, y_norm, mask_D，并且只取 t=0 帧生成 z0。
        """
        B, C, X, Y, T = inputs.shape
        assert C >= 7, "inputs should have at least 7 channels: x,y,t_norm,mask_D,sin,cos,t_raw"

        # 仅用 t=0
        ti = 0
        base = inputs[:, [0,1,3], :, :, ti]           # (B,3,X,Y) -> x_norm, y_norm, mask_D（仅空间）
        Xn = base[:, 0:1]
        Yn = base[:, 1:2]
        fx = self._fourier_features(Xn, self.fourier_freqs)
        fy = self._fourier_features(Yn, self.fourier_freqs)
        aug = torch.cat([base, fx, fy], dim=1)        # (B, 3+fourier_ch, X, Y)
        tokens = self.to_patch_embedding(aug)         # (B,N,dim)

        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        tokens = self.transformer(tokens)
        z0 = self.to_z(tokens)                        # (B,N,z)

        # Koopman 时间推进 —— 使用 t_raw 计算 dt（真实时间）
        t_raw_grid = inputs[:, 6, 0, 0, :]            # (B,T) 未归一化时间
        t0 = t_raw_grid[:, :1]
        dt = t_raw_grid - t0                          # (B,T)

        zt = self.koopman(z0, dt)                     # (B,T,N,z)
        B_, T_, N_, d_ = zt.shape
        zt_flat = zt.reshape(B_*T_, N_, d_)
        dec = self.decoder(zt_flat)                   # (B*T, N*px*py*out_ch)
        px, py = self.patch_size
        dec = dec.view(B_*T_, N_, px, py, self.out_channels)
        dec = dec.permute(0, 4, 1, 2, 3)
        dec = dec.view(B, T, self.out_channels, self.num_patches, px, py)
        dec = dec.permute(0, 2, 3, 4, 5, 1)
        out = dec.reshape(B, self.out_channels, self.nx*px, self.ny*py, T)

        if self.refine is not None:
            out = self.refine(out)
        return out

# ==================== Loss (分域计算) ====================
class ChannelWeightedLoss(nn.Module):
    def __init__(self, w_u1=2.0, w_u2=5.0, w_p=0.75, w_phi=1.0):
        super().__init__()
        self.weights = [float(w_u1), float(w_u2), float(w_p), float(w_phi)]

    def forward(self, outputs, targets):
        """
        outputs: (B,4,X,Y,T) → 预测的u1/u2/p/phi
        targets: (B,6,X,Y,T) → 前4为物理量，后2为mask_S/mask_D
        """
        total = 0.0
        ch_losses = []

        mask_S = targets[:, 4:5, ...]
        mask_D = targets[:, 5:6, ...]
        domain_masks = torch.cat([mask_S, mask_S, mask_S, mask_D], dim=1)

        for i, w in enumerate(self.weights):
            pred = outputs[:, i:i+1, ...]
            true = targets[:, i:i+1, ...]
            masked = F.mse_loss(pred, true, reduction='none') * domain_masks[:, i:i+1, ...]
            valid_pixels = domain_masks[:, i:i+1, ...].sum() + 1e-12
            avg_loss = masked.sum() / valid_pixels
            total += w * avg_loss
            ch_losses.append(float(avg_loss.item()))
        return total, ch_losses

# ==================== Utils (viz/eval) ====================
def interpolate_error(error, x_range, y_range, interpolation_factor=4):
    x_orig = np.linspace(x_range[0], x_range[1], error.shape[0])
    y_orig = np.linspace(y_range[0], y_range[1], error.shape[1])
    X_orig, Y_orig = np.meshgrid(x_orig, y_orig, indexing='ij')
    x_interp = np.linspace(x_range[0], x_range[1], error.shape[0] * interpolation_factor)
    y_interp = np.linspace(y_range[0], y_range[1], error.shape[1] * interpolation_factor)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp, indexing='ij')
    try:
        points = np.column_stack((X_orig.ravel(), Y_orig.ravel()))
        values = error.ravel()
        error_interp = griddata(points, values, (X_interp, Y_interp), method='cubic', fill_value=0)
        smoothed_error = gaussian_filter(error_interp, sigma=1.0)
        return smoothed_error, X_interp, Y_interp
    except Exception:
        return error, X_orig, Y_orig

def visualize_results(model, device, trainer=None, channel_stats_train=None,
                      amp_temper=False, ref_use_all=True, ref_K=1):
    dataset = PhysicsDataset(
        spatial_size=CFG['spatial_size'],
        temporal_size=CFG['T_eval'],
        spatial_domain=((0.0, 1.0), (-0.25, 0.75)),
        temporal_domain=CFG['temporal_domain_eval'],
        num_samples=1, mode='eval',
        channel_stats_external=channel_stats_train,
        amp_temper=amp_temper
    )
    inputs, targets = dataset[0]
    y_coords = dataset.y
    spatial_size_y = CFG['spatial_size'][1]

    # split 索引
    split_idx = np.where(y_coords >= 0.0)[0][0]
    valid_ranges = {
        'u1': (0, split_idx),
        'u2': (0, split_idx),
        'p':  (0, split_idx),
        'phi':(split_idx, spatial_size_y)
    }
    valid_y_ranges = {
        'u1': (dataset.spatial_domain[1][0], 0.0),
        'u2': (dataset.spatial_domain[1][0], 0.0),
        'p' : (dataset.spatial_domain[1][0], 0.0),
        'phi': (0.0, dataset.spatial_domain[1][1]),
    }

    targets_np = targets[0:4, ...].numpy()
    inputs_gpu = inputs.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs_gpu, ref_use_all=False, ref_K=1)  # 本版恒用 t=0
    outputs_np = outputs.squeeze(0).cpu().numpy()

    # 绘制半张图
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    def t_to_index(t_real):
        t_grid = np.linspace(0.0, 2.0, CFG['T_eval'])
        return int(np.argmin(np.abs(t_grid - t_real)))

    cmap_main = 'viridis'
    cmap_error = 'Blues'

    for t_val in time_points:
        t_idx = t_to_index(t_val)
        fig, axes = plt.subplots(4, 3, figsize=(12, 8.8), dpi=300, constrained_layout=True)
        fig.suptitle(f'NS–Darcy (Valid Subdomains) at t = {t_val:.2f}', fontsize=16, y=1.02)


        titles = ['u1', 'u2', 'p', 'phi']
        for row, title in enumerate(titles):
            y_start, y_end = valid_ranges[title]
            true_data = targets_np[row, :, y_start:y_end, t_idx]
            pred_data = outputs_np[row, :, y_start:y_end, t_idx]
            err_data = np.abs(true_data - pred_data)

            vmin = np.nanmin(true_data)
            vmax = np.nanmax(true_data)
            err_vmax = np.nanmax(err_data) * 0.8
            extent = [dataset.spatial_domain[0][0], dataset.spatial_domain[0][1],
                      valid_y_ranges[title][0], valid_y_ranges[title][1]]

            ax_true = axes[row, 0]
            im_true = ax_true.imshow(true_data.T, origin='lower', cmap=cmap_main, extent=extent, vmin=vmin, vmax=vmax)
            ax_true.set_title(f'True: {title}', fontsize=12)
            ax_true.set_xlabel('x'); ax_true.set_ylabel('y')
            cbar = plt.colorbar(im_true, ax=ax_true, shrink=1); cbar.ax.tick_params(labelsize=8)

            ax_pred = axes[row, 1]
            im_pred = ax_pred.imshow(pred_data.T, origin='lower', cmap=cmap_main, extent=extent, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Predicted: {title}', fontsize=12)
            ax_pred.set_xlabel('x'); ax_pred.set_ylabel('y')
            cbar = plt.colorbar(im_pred, ax=ax_pred, shrink=1); cbar.ax.tick_params(labelsize=8)

            ax_err = axes[row, 2]
            im_err = ax_err.imshow(err_data.T, origin='lower', cmap=cmap_error, extent=extent, vmin=0, vmax=err_vmax)
            ax_err.set_title(f'Error: {title}', fontsize=12)
            ax_err.set_xlabel('x'); ax_err.set_ylabel('y')
            cbar = plt.colorbar(im_err, ax=ax_err, shrink=1); cbar.ax.tick_params(labelsize=8)
            
            # 让每个子图依据网格自适应，不受数据纵横比影响
            for ax in (ax_true, ax_pred, ax_err):
                ax.set_aspect('auto')         # Matplotlib≥3.1通用
    # 如果你的 Matplotlib ≥ 3.3，也可以用下面这句固定“盒子”纵横比：
    # ax.set_box_aspect(0.35)     # 例如 0.35；可按需要微调


        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.25, hspace=0.3)
        save_path = os.path.join(FIG_DIR, f'nsd_halfplot_t_{t_val:.2f}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    # 训练曲线（保持单份）
    if trainer and len(trainer.train_channel_losses) > 0:
        plt.figure(figsize=(16, 10), dpi=600)
        ep = np.arange(len(trainer.train_loss_history))

        colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']
        names   = ['u1', 'u2', 'p', 'phi']

    # 防止对数坐标下出现 log(0)
        eps = 1e-12

    # 自适应标记密度（最多 ~20 个标记）
        me = max(1, len(ep) // 20)

        for i, (n, color, marker) in enumerate(zip(names, colors, markers)):
            tr = [max(eps, float(x[i])) for x in trainer.train_channel_losses]  # 加 eps 防止0
            va = [max(eps, float(x[i])) for x in trainer.val_channel_losses]

            plt.plot(ep, tr, '-',  color=color, marker=marker, markersize=6,
                     markevery=me, linewidth=2.0, alpha=0.85, label=f'Train {n}')
            plt.plot(ep, va, '--', color=color, marker=marker, markersize=6,
                     markevery=me, linewidth=2.0, alpha=0.85, label=f'Val {n}')

        plt.yscale('log')
        plt.grid(True, which='both', ls='-', alpha=0.3)
        plt.title('Training Progress: Per-Channel MSE (masked average)', fontsize=20)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss (log scale)', fontsize=18)
        plt.xticks(fontsize=16); plt.yticks(fontsize=16)
        plt.legend(ncol=2, fontsize=16, framealpha=0.9, loc='upper right')
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, '96_training_progress.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()


def evaluate_prediction_quality(model, device, channel_stats_train=None, amp_temper=False,
                                ref_use_all=False, ref_K=1):
    print('\n[Eval] Quality on t in [0,2] including extrapolation to 2.0:')
    dataset = PhysicsDataset(
        spatial_size=CFG['spatial_size'],
        temporal_size=CFG['T_eval'],
        spatial_domain=((0.0, 1.0), (-0.25, 0.75)),
        temporal_domain=CFG['temporal_domain_eval'],
        num_samples=1, mode='eval',
        channel_stats_external=channel_stats_train,
        amp_temper=amp_temper
    )
    inputs, targets = dataset[0]
    y_coords = dataset.y
    split_idx = np.where(y_coords >= 0.0)[0][0]
    valid_ranges = {
        'u1': (0, split_idx), 'u2': (0, split_idx), 'p': (0, split_idx),
        'phi': (split_idx, CFG['spatial_size'][1])
    }

    inputs_gpu = inputs.unsqueeze(0).to(device)
    targets_np = targets[0:4, ...].numpy()
    with torch.no_grad():
        outputs = model(inputs_gpu, ref_use_all=False, ref_K=1)  # 恒 t=0
    outputs_np = outputs.squeeze(0).cpu().numpy()

    # 真正的 t 轴（0~2）
    T_real = np.linspace(0.0, 2.0, CFG['T_eval'])

    results = []
    titles = ['u1','u2','p','phi']
    for i, title in enumerate(titles):
        y_start, y_end = valid_ranges[title]
        for t_val in [0.0, 0.5, 1.0, 1.5, 2.0]:
            t_idx = int(np.argmin(np.abs(T_real - t_val)))
            true_data = targets_np[i, :, y_start:y_end, t_idx].flatten()
            pred_data = outputs_np[i, :, y_start:y_end, t_idx].flatten()
            valid_mask = ~np.isnan(true_data)
            true_data = true_data[valid_mask]
            pred_data = pred_data[valid_mask]
            if len(true_data) == 0:
                continue
            error = true_data - pred_data
            mse = float(np.mean(error**2))
            mae = float(np.mean(np.abs(error)))
            max_error = float(np.max(np.abs(error)))
            den = float(np.linalg.norm(true_data))
            rel_l2 = float(np.linalg.norm(error) / (den + 1e-12)) if den > 0 else float('nan')
            results.append({'channel': title, 'time': t_val, 'mse': mse, 'mae': mae,
                            'max_error': max_error, 'rel_l2': rel_l2})

    for r in results:
        print(f"{r['channel']:>3} t={r['time']:.1f} | MSE={r['mse']:.8f} | MAE={r['mae']:.8f} | Max={r['max_error']:.8f} | Rel-L2={r['rel_l2']:.6f}")
    return results

# ==================== Trainer ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = ChannelWeightedLoss(
            w_u1=cfg['w_u1'], w_u2=cfg['w_u2'], w_p=cfg['w_p'], w_phi=cfg['w_phi']
        )
        self.optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        self.base_lr = cfg['lr']
        self.warmup_epochs = int(cfg.get('warmup_epochs', 10))
        cosine_T = max(1, cfg['epochs'] - self.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_T, eta_min=1e-6
        )

        self.train_loss_history, self.val_loss_history = [], []
        self.train_channel_losses, self.val_channel_losses = [], []

        # 这里的两个字段不再影响 forward（为了接口兼容保留）
        self.ref_use_all = False
        self.ref_K = 1

    def _forward(self, inputs):
        return self.model(inputs,
                          ref_use_all=False,
                          ref_K=1)

    def _set_warmup_lr(self, epoch):
        lr = self.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def train(self, num_epochs=200):
        print('[Train] start')
        best_val = float('inf')
        self.model.train()
        for ep in range(num_epochs):
            if ep < self.warmup_epochs:
                self._set_warmup_lr(ep)
            if ep % 10 == 0:
                cur_lr = self.optimizer.param_groups[0]['lr']
                print(f"[lr] {cur_lr:.3e}")

            run_loss = 0.0
            run_ch = np.zeros(4)

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._forward(inputs)
                loss, ch_losses = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                run_loss += float(loss.item())
                run_ch += np.array(ch_losses)

            avg_loss = run_loss / len(self.train_loader)
            avg_ch = run_ch / len(self.train_loader)

            val_loss, val_ch = self.validate()

            self.train_loss_history.append(avg_loss)
            self.val_loss_history.append(val_loss)
            self.train_channel_losses.append(avg_ch)
            self.val_channel_losses.append(val_ch)

            print(f"Epoch {ep:03d} | Train {avg_loss:.6e} | Val {val_loss:.6e} | TrainCh {avg_ch}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), os.path.join(FIG_DIR, 'best_model_nsd_multiframe_96_rotblock.pth'))

            # 调度器在 epoch 末尾 step（warmup 后）
            if ep >= self.warmup_epochs:
                self.scheduler.step()

        print(f'[Train] done. Best val = {best_val:.6e}')
        return self.model

    def validate(self):
        self.model.eval()
        tot, tot_ch = 0.0, np.zeros(4)
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._forward(inputs)
                loss, ch_losses = self.criterion(outputs, targets)
                tot += float(loss.item())
                tot_ch += np.array(ch_losses)
        self.model.train()
        return tot/len(self.val_loader), tot_ch/len(self.val_loader)

# ==================== Main ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # 训练数据集（t∈[0,1]）
    dataset_train = PhysicsDataset(
        spatial_size=CFG['spatial_size'],
        temporal_size=CFG['T_train'],
        spatial_domain=((0.0, 1.0), (-0.25, 0.75)),
        temporal_domain=CFG['temporal_domain_train'],
        num_samples=200, mode='train',
        amp_temper=CFG['amp_tempering']
    )
    channel_stats_train = dataset_train.channel_stats

    n_train = int(0.8 * len(dataset_train))
    n_val = len(dataset_train) - n_train
    train_set, val_set = random_split(dataset_train, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CFG['batch_size'], shuffle=False, pin_memory=True)

    # 模型
    model = ViT_KNO_MultiFrame(
        spatial_size=CFG['spatial_size'],
        patch_size=CFG['patch_size'],
        dim=CFG['dim'], depth=CFG['depth'], heads=CFG['heads'], mlp_dim=CFG['mlp_dim'],
        in_channels_base=7,                       # 输入仍是7通道；编码器只用 x,y,mask_D
        fourier_freqs=CFG['fourier_freqs'],
        z_dim=CFG['z_dim'], out_channels=4, emb_dropout=CFG['emb_dropout'],
        refine=CFG['refine'], refine_hidden=CFG['refine_hidden'], refine_blocks=CFG['refine_blocks']
    ).to(device)

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    trainer = Trainer(model, train_loader, val_loader, device, CFG)
    model = trainer.train(num_epochs=CFG['epochs'])

    # 保存
    torch.save(model.state_dict(), os.path.join(FIG_DIR, 'model_nsd_multiframe_96_rotblock_last.pth'))
    print('Saved:', os.path.join(FIG_DIR, 'model_nsd_multiframe_96_rotblock_last.pth'))

    # 可视化
    visualize_results(model, device, trainer, channel_stats_train=channel_stats_train,
                      amp_temper=CFG['amp_tempering'],
                      ref_use_all=False, ref_K=1)

    # 评估（含外推到 t=2）
    results = evaluate_prediction_quality(model, device, channel_stats_train=channel_stats_train,
                                          amp_temper=CFG['amp_tempering'],
                                          ref_use_all=False, ref_K=1)

    # 保存评估结果
    with open(os.path.join(FIG_DIR, 'nsd_multiframe96_rotblock_prediction_evaluation.txt'), 'w') as f:
        f.write("Navier–Stokes–Darcy (Multi-frame, 96x96) Prediction Quality (train on [0,1], eval includes t=2) - RotBlocks\n")
        f.write("Channel | Time | MSE | MAE | Max Error | Relative L2\n")
        f.write("-"*90 + "\n")
        for r in results:
            f.write(f"{r['channel']} t={r['time']:.1f} | MSE: {r['mse']:.8f} | MAE: {r['mae']:.8f} | "
                    f"Max: {r['max_error']:.8f} | Rel-L2: {r['rel_l2']:.6f}\n")

    print("\nAll outputs are in:", os.path.abspath(FIG_DIR))

if __name__ == '__main__':
    main()
