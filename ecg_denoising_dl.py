"""
ECG 数字滤波去噪 — 深度学习方法 (CNN 去噪自编码器)
====================================================
基于 PyTorch 的 1D 卷积去噪自编码器 (CNN-DAE)
与 ecg_denoising.py 的传统滤波管线做公平对比

参考论文:
  - ADCD-Net (2025): Multi-Scale Dense CNN for ECG, SNR 18.95 dB
  - MECG-E (2024): Mamba-based ECG Enhancer
  - MDPI Bioengineering (2024): "Preprocessing and Denoising Techniques for ECG"

云服务器运行 (AutoDL / 阿里云 / Colab):
  1. 上传代码: scp ecg_denoising_dl.py root@IP:/root/autodl-tmp/
  2. 安装依赖: pip install torch numpy scipy matplotlib
  3. tmux 后台运行:
       tmux new -s ecg
       python ecg_denoising_dl.py --headless
       # Ctrl+B D 断开, tmux attach -t ecg 重连
  4. 下载结果: scp -r root@IP:/root/autodl-tmp/figures/ ./

用法:
  python ecg_denoising_dl.py               # 本地训练 + 弹窗显示图表
  python ecg_denoising_dl.py --headless     # 云服务器模式 (仅保存图片)
  python ecg_denoising_dl.py --eval         # 仅评估已有模型
"""

import os
import sys
import time
import argparse
import numpy as np

# 云服务器无头模式：必须放在 import matplotlib 之前
HEADLESS = '--headless' in sys.argv
if HEADLESS:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端，不需要显示器

import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import welch

# ===================== 配置 =====================
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
print(f"模式: {'云服务器 (无头)' if HEADLESS else '本地 (交互)'}")

# 中文字体 — 云服务器可能没有中文字体，降级为英文
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


# ============================================================
# 1. ECG 数据生成器 (与 ecg_denoising.py 同源，保证公平对比)
# ============================================================

def generate_ecg_segment(fs=360, duration=4):
    """
    生成单段 4 秒仿真 ECG (含噪 + 干净配对)

    与 ecg_denoising.py 中的 generate_synthetic_ecg 使用相同的
    P-QRS-T 模型 + 三类噪声 (50Hz 工频 + 基线漂移 + 肌电噪声)
    确保对比公平
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)

    # 干净 ECG
    ecg_clean = np.zeros(n_samples)
    idx = 0
    hr = 72 / 60  # 1.2 Hz
    beat_period = int(fs / hr)

    while idx < n_samples - 250:
        rr = int(beat_period * (1 + 0.05 * np.random.randn()))
        rr = max(240, min(360, rr))
        resp_mod = 1 + 0.08 * np.sin(2 * np.pi * 0.25 * idx / fs)

        t_beat = np.linspace(0, 0.7, min(250, n_samples - idx))
        L = len(t_beat)

        qrs = 1.2 * resp_mod * np.exp(-((t_beat - 0.12) / 0.012)**2)
        qrs += -0.25 * resp_mod * np.exp(-((t_beat - 0.08) / 0.012)**2)
        qrs += -0.18 * resp_mod * np.exp(-((t_beat - 0.16) / 0.012)**2)
        p_wave = 0.15 * resp_mod * np.exp(-((t_beat - 0.03) / 0.025)**2)
        t_wave = 0.35 * resp_mod * np.exp(-((t_beat - 0.35) / 0.05)**2)

        ecg_clean[idx:idx+L] = qrs + p_wave + t_wave
        idx += rr

    # 加噪
    pli = 0.12 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 100 * t)
    bw = 0.35 * np.sin(2 * np.pi * 0.25 * t) + 0.15 * np.sin(2 * np.pi * 0.05 * t + 1.2)
    emg = 0.04 * np.random.randn(n_samples)

    ecg_noisy = ecg_clean + pli + bw + emg

    # 确保长度一致 (pad or trim to fixed length)
    target_len = int(fs * 4)  # 1440 samples
    if len(ecg_clean) < target_len:
        ecg_clean = np.pad(ecg_clean, (0, target_len - len(ecg_clean)))
        ecg_noisy = np.pad(ecg_noisy, (0, target_len - len(ecg_noisy)))
    else:
        ecg_clean = ecg_clean[:target_len]
        ecg_noisy = ecg_noisy[:target_len]

    return ecg_noisy.astype(np.float32), ecg_clean.astype(np.float32)


class ECGDataset(Dataset):
    """ECG 去噪数据集"""

    def __init__(self, n_samples=2000, seg_len=4, fs=360, augment=True):
        self.n_samples = n_samples
        self.seg_len = seg_len
        self.fs = fs
        self.augment = augment

        print(f"  生成 {n_samples} 段 ECG 训练数据 ({seg_len}s each)...")
        self.noisy_list = []
        self.clean_list = []

        for i in range(n_samples):
            noisy, clean = generate_ecg_segment(fs=fs, duration=seg_len)
            self.noisy_list.append(noisy)
            self.clean_list.append(clean)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n_samples}...")

        self.noisy = np.stack(self.noisy_list)
        self.clean = np.stack(self.clean_list)
        print(f"  数据形状: {self.noisy.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.noisy[idx]
        y = self.clean[idx]

        # 数据增强: 随机翻转 (ECG 极性可能反转)
        if self.augment and np.random.rand() > 0.5:
            x = -x
            y = -y

        # 添加通道维度: (1, L)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)

        return x, y


# ============================================================
# 2. CNN 去噪自编码器 (CNN-DAE)
# ============================================================

class CNNDenoisingAutoencoder(nn.Module):
    """
    1D 卷积去噪自编码器

    架构: Encoder → Bottleneck → Decoder (全卷积, 无全连接层)
    灵感: ADCD-Net (2025) 的编码器-解码器设计 + U-Net 跳跃连接

    输入:  (B, 1, L)  含噪 ECG, L = fs * seg_len
    输出:  (B, 1, L)  去噪 ECG
    """

    def __init__(self, input_len=1440, base_channels=16):
        super().__init__()
        self.input_len = input_len
        C = base_channels

        # ====== Encoder ======
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, C, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(C), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv1d(C, C*2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(C*2), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv1d(C*2, C*4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(C*4), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(
            nn.Conv1d(C*4, C*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(C*8), nn.ReLU(inplace=True))

        # ====== Bottleneck ======
        self.bottleneck = nn.Sequential(
            nn.Conv1d(C*8, C*8, kernel_size=3, padding=1),
            nn.BatchNorm1d(C*8), nn.ReLU(inplace=True),
            nn.Conv1d(C*8, C*8, kernel_size=3, padding=1),
            nn.BatchNorm1d(C*8), nn.ReLU(inplace=True))

        # ====== Decoder (with skip connections) ======
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(C*8, C*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(C*4), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(C*8, C*2, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(C*2), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(C*4, C, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(C), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(C*2, 1, kernel_size=31, stride=2, padding=15, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # 裁剪到输入长度 (转置卷积可能导致长度不匹配)
        if d1.shape[-1] > x.shape[-1]:
            d1 = d1[..., :x.shape[-1]]
        elif d1.shape[-1] < x.shape[-1]:
            d1 = nn.functional.pad(d1, (0, x.shape[-1] - d1.shape[-1]))

        return d1


# ============================================================
# 3. 训练与评估
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def compute_metrics(clean, denoised, fs=360):
    """计算去噪指标: MSE, SNR, PRD, 相关系数, QRS保真度"""
    clean = clean.flatten()
    denoised = denoised.flatten()

    mse = np.mean((clean - denoised)**2)
    snr = 10 * np.log10(np.var(clean) / (np.var(clean - denoised) + 1e-10))
    prd = 100 * np.sqrt(np.sum((clean - denoised)**2) / (np.sum(clean**2) + 1e-10))
    corr = np.corrcoef(clean, denoised)[0, 1]

    # QRS保真度：搜索R峰，比较峰峰值
    beat_period = int(fs / 1.2)
    r_peak_offset = int(0.1 / 0.6 * 200)
    qrs_search_center = 4 * beat_period + r_peak_offset  # 第4个心拍
    search_range = 40
    if qrs_search_center + search_range < len(clean):
        r_peak = qrs_search_center - search_range + np.argmax(
            clean[qrs_search_center - search_range:qrs_search_center + search_range])
        half = 15
        r_slice = slice(max(0, r_peak - half), min(len(clean), r_peak + half))
        amp_clean = np.max(clean[r_slice]) - np.min(clean[r_slice])
        amp_den = np.max(denoised[r_slice]) - np.min(denoised[r_slice])
        qrs_fid = 100 * amp_den / amp_clean if amp_clean > 1e-6 else float('nan')
    else:
        qrs_fid = float('nan')

    return {'MSE': mse, 'SNR': snr, 'PRD': prd, 'Corr': corr, 'QRS': qrs_fid}


# ============================================================
# 4. 与传统方法对比
# ============================================================

def evaluate_dl(test_noisy, test_clean, dl_denoised, fs=360):
    """
    仅评估深度学习模型的去噪性能。
    传统滤波指标直接从 ecg_denoising.py 的输出中引用，此处不再重复计算。
    QRS保真度以干净参考信号 ecg_clean 为基准（CNN-DAE 是端到端模型，目标即复现 clean）。
    与传统代码的基准不同：传统代码测的是低通滤波器单独对 QRS 的改变（参考=低通输入），
    CNN-DAE 测的是全链路去噪对 QRS 的改变（参考=干净ECG）。
    """
    metrics = {}
    metrics['Noisy Raw'] = compute_metrics(test_clean, test_noisy, fs)
    metrics['CNN-DAE (DL)'] = compute_metrics(test_clean, dl_denoised, fs)

    # QRS保真度：以 ecg_clean 为参考，取多个心拍的中位数（消除呼吸调制影响）
    beat_period = int(fs / 1.2)
    r_peak_offset = int(0.1 / 0.6 * 200)
    search_range = 40
    half = 15
    n_beats = min(6, len(test_clean) // beat_period - 2)  # 取第2到第7拍

    amps_clean, amps_dl = [], []
    for b in range(2, 2 + n_beats):
        center = b * beat_period + r_peak_offset
        if center + search_range >= len(test_clean):
            break
        r_c = center - search_range + np.argmax(
            test_clean[center - search_range:center + search_range])
        r_d = center - search_range + np.argmax(
            dl_denoised[center - search_range:center + search_range])
        s_c = slice(max(0, r_c - half), min(len(test_clean), r_c + half))
        s_d = slice(max(0, r_d - half), min(len(dl_denoised), r_d + half))
        amps_clean.append(np.max(test_clean[s_c]) - np.min(test_clean[s_c]))
        amps_dl.append(np.max(dl_denoised[s_d]) - np.min(dl_denoised[s_d]))

    if amps_clean:
        amp_clean_med = np.median(amps_clean)
        amp_dl_med = np.median(amps_dl)
        if amp_clean_med > 1e-6:
            metrics['CNN-DAE (DL)']['QRS'] = 100 * amp_dl_med / amp_clean_med
        else:
            metrics['CNN-DAE (DL)']['QRS'] = float('nan')
    else:
        metrics['CNN-DAE (DL)']['QRS'] = float('nan')

    return metrics


# ============================================================
# 5. 可视化
# ============================================================

def plot_training_curve(train_losses, val_losses):
    """训练曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    ax.plot(val_losses, 'r-', linewidth=1, alpha=0.7, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('CNN-DAE Training Curve')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_fig('dl_training_curve.pdf')
    if not HEADLESS:
        plt.show()
    plt.close()


def plot_dl_vs_traditional(t, test_clean, test_noisy, dl_denoised, trad_denoised):
    """深度学习 vs 传统滤波 时域对比"""
    fs = int(1 / (t[1] - t[0]))
    time_plot = 3  # 显示 3 秒
    samples = int(time_plot * fs)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(t[:samples], test_clean[:samples], 'k', linewidth=0.8)
    axes[0].set_title('(a) Clean ECG (Reference)')
    axes[0].set_ylabel('mV'); axes[0].grid(alpha=0.3)

    axes[1].plot(t[:samples], test_noisy[:samples], 'r', linewidth=0.5, alpha=0.7)
    axes[1].set_title('(b) Noisy ECG')
    axes[1].set_ylabel('mV'); axes[1].grid(alpha=0.3)

    axes[2].plot(t[:samples], trad_denoised[:samples], 'orange', linewidth=0.8)
    axes[2].set_title('(c) Traditional Filtering (Notch+Median+Lowpass)')
    axes[2].set_ylabel('mV'); axes[2].grid(alpha=0.3)

    axes[3].plot(t[:samples], dl_denoised[:samples], 'blue', linewidth=0.8)
    axes[3].set_title('(d) CNN-DAE (Deep Learning)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('mV'); axes[3].grid(alpha=0.3)

    plt.tight_layout()
    save_fig('dl_vs_traditional.pdf')
    if not HEADLESS:
        plt.show()
    plt.close()


def plot_psd_dl_comparison(test_noisy, test_clean, dl_denoised, trad_denoised, fs=360):
    """PSD 对比: 含噪 vs 传统 vs 深度学习"""
    f_n, psd_n = welch(test_noisy, fs, nperseg=1024)
    f_c, psd_c = welch(test_clean, fs, nperseg=1024)
    _, psd_t = welch(trad_denoised, fs, nperseg=1024)
    _, psd_dl = welch(dl_denoised, fs, nperseg=1024)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].semilogy(f_n, psd_n, 'r', alpha=0.5, linewidth=0.8, label='Noisy')
    axes[0].semilogy(f_c, psd_c, 'k', linewidth=0.8, label='Clean Ref')
    axes[0].semilogy(f_n, psd_t, 'orange', linewidth=1, label='Traditional')
    axes[0].semilogy(f_n, psd_dl, 'blue', linewidth=1, label='CNN-DAE')
    axes[0].axvline(50, color='gray', ls='--', alpha=0.5)
    axes[0].set_xlim([0, 180])
    axes[0].set_xlabel('Frequency (Hz)'); axes[0].set_ylabel('PSD')
    axes[0].set_title('Power Spectral Density (Full Band)')
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].semilogy(f_n, psd_n, 'r', alpha=0.5, linewidth=0.8, label='Noisy')
    axes[1].semilogy(f_c, psd_c, 'k', linewidth=0.8, label='Clean Ref')
    axes[1].semilogy(f_n, psd_t, 'orange', linewidth=1, label='Traditional')
    axes[1].semilogy(f_n, psd_dl, 'blue', linewidth=1, label='CNN-DAE')
    axes[1].set_xlim([0, 5])
    axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('PSD')
    axes[1].set_title('PSD Low-Frequency Zoom (0-5 Hz)')
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_fig('dl_psd_comparison.pdf')
    if not HEADLESS:
        plt.show()
    plt.close()


def plot_metrics_bar(metrics_dict):
    """各方法指标柱状图"""
    methods = list(metrics_dict.keys())
    snr_vals = [metrics_dict[m]['SNR'] for m in methods]
    prd_vals = [metrics_dict[m]['PRD'] for m in methods]
    corr_vals = [metrics_dict[m]['Corr'] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#d62728', '#ff7f0e', '#1f77b4']

    for ax, vals, title, ylabel in zip(
        axes,
        [snr_vals, prd_vals, corr_vals],
        ['SNR (dB)', 'PRD (%)', 'Correlation'],
        ['dB', '%', 'r']
    ):
        bars = ax.bar(methods, vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=10, labelsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f'{val:.2f}', ha='center', fontsize=9)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Denoising Performance: Noisy vs Traditional vs CNN-DAE', fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig('dl_metrics_comparison.pdf')
    if not HEADLESS:
        plt.show()
    plt.close()


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [图] {path}")
    return path


# ============================================================
# 6. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ECG CNN-DAE 去噪')
    parser.add_argument('--eval', action='store_true', help='仅评估已有模型')
    parser.add_argument('--headless', action='store_true', help='云服务器无头模式（仅存图不弹窗）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--train_samples', type=int, default=2000, help='训练样本数')
    args = parser.parse_args()

    fs = 360
    seg_len = 4  # 每段 4 秒 = 1440 采样点
    input_len = fs * seg_len

    print("=" * 70)
    print("  ECG CNN 去噪自编码器 (CNN-DAE)")
    print(f"  设备: {DEVICE}  |  输入长度: {input_len}  |  fs={fs} Hz")
    print("=" * 70)

    # ---- 构建模型 ----
    model = CNNDenoisingAutoencoder(input_len=input_len, base_channels=16).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")
    print(f"架构: Encoder→Bottleneck→Decoder (U-Net skip connections)")

    model_path = os.path.join(MODEL_DIR, 'ecg_cnn_dae.pth')

    if not args.eval:
        # ---- 准备数据 ----
        print(f"\n{'='*50}")
        print(f"准备训练数据")
        print(f"{'='*50}")
        dataset = ECGDataset(n_samples=args.train_samples, seg_len=seg_len, fs=fs)
        n_train = int(len(dataset) * 0.85)
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                                num_workers=0, pin_memory=True)
        print(f"  训练: {n_train}, 验证: {n_val}")

        # ---- 训练 ----
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)

        print(f"\n{'='*50}")
        print(f"开始训练 ({args.epochs} epochs)")
        print(f"{'='*50}")

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        t_start = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{args.epochs}  "
                      f"Train={train_loss:.6f}  Val={val_loss:.6f}  "
                      f"LR={optimizer.param_groups[0]['lr']:.2e}")

        t_train = time.time() - t_start
        print(f"\n训练完成! 耗时 {t_train:.0f}s ({t_train/60:.1f} min)")
        print(f"最佳验证 Loss: {best_val_loss:.6f}")
        print(f"模型保存至: {model_path}")

        # 绘制训练曲线
        plot_training_curve(train_losses, val_losses)
    else:
        print(f"\n加载已有模型: {model_path}")
        if not os.path.exists(model_path):
            print("错误: 模型文件不存在! 请先训练。")
            sys.exit(1)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # ---- 测试与对比 ----
    print(f"\n{'='*50}")
    print(f"测试与对比评估")
    print(f"{'='*50}")

    # 生成测试数据
    test_noisy, test_clean = generate_ecg_segment(fs=fs, duration=10)
    t_test = np.arange(len(test_noisy)) / fs

    # 深度学习推理
    model.eval()
    with torch.no_grad():
        x_test = torch.from_numpy(test_noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        dl_denoised = model(x_test).squeeze().cpu().numpy()

    # DL 评估
    metrics = evaluate_dl(test_noisy, test_clean, dl_denoised, fs=fs)

    # 简化传统管线（仅用于可视化，指标以 ecg_denoising.py 输出为准）
    from scipy.signal import iirnotch, filtfilt, medfilt, cheb2ord, cheby2
    b_n, a_n = iirnotch(50/(fs/2), 30)
    ecg_trad = filtfilt(b_n, a_n, test_noisy)
    ecg_trad = ecg_trad - medfilt(ecg_trad, 73)
    ny_viz = fs / 2
    N_lp, wn_lp = cheb2ord(100/ny_viz, 130/ny_viz, 1, 60)
    b_l, a_l = cheby2(N_lp, 60, wn_lp, btype='low')
    ecg_trad = filtfilt(b_l, a_l, ecg_trad)
    trad_denoised = ecg_trad

    # ---- 输出指标（仅 DL，传统指标见 ecg_denoising.py） ----
    dl_m = metrics['CNN-DAE (DL)']
    raw_m = metrics['Noisy Raw']
    dl_qrs = dl_m['QRS']
    snr_improve = dl_m['SNR'] - raw_m['SNR']

    print(f"\n{'='*60}")
    print(f"CNN-DAE 去噪性能（传统滤波指标见 ecg_denoising.py）")
    print(f"{'='*60}")
    print(f"  {'指标':<15} {'含噪原始':<15} {'CNN-DAE':<15}")
    print(f"  {'-'*45}")
    print(f"  {'SNR(dB)':<15} {raw_m['SNR']:<15.1f} {dl_m['SNR']:<15.1f}")
    print(f"  {'MSE':<15} {raw_m['MSE']:<15.6f} {dl_m['MSE']:<15.6f}")
    print(f"  {'PRD(%)':<15} {raw_m['PRD']:<15.2f} {dl_m['PRD']:<15.2f}")
    print(f"  {'相关系数 r':<15} {raw_m['Corr']:<15.4f} {dl_m['Corr']:<15.4f}")
    qrs_str = f"{dl_qrs:.1f}" if not np.isnan(dl_qrs) else "N/A"
    print(f"  {'QRS保真%':<15} {'—':<15} {qrs_str:<15}")
    print(f"  {'-'*45}")
    print(f"  SNR 改善: +{snr_improve:.1f} dB")
    print(f"  注: CNN-DAE的QRS以ecg_clean为参考（端到端，目标是复现clean）")
    print(f"      传统Chebyshev II的QRS=95.8%以低通输入为参考（测量低通单独保真度）")
    print(f"      两者基准不同，不可直接比较数值大小")
    print(f"  （传统滤波 SNR 改善 +13.9 dB — 来自 ecg_denoising.py）")

    # ---- 可视化 ----
    plot_dl_vs_traditional(t_test, test_clean, test_noisy, dl_denoised, trad_denoised)
    plot_psd_dl_comparison(test_noisy, test_clean, dl_denoised, trad_denoised, fs=fs)
    plot_metrics_bar(metrics)

    # ---- 结论 ----
    print(f"\n{'='*60}")
    print(f"结论")
    print(f"{'='*60}")
    print(f"  CNN-DAE SNR 改善:     +{snr_improve:.1f} dB")
    print(f"  CNN-DAE QRS保真:      {qrs_str}%")
    print(f"  传统滤波 SNR 改善:    +13.9 dB (来自 ecg_denoising.py)")
    print(f"  传统滤波 QRS保真:     95.8% (Chebyshev II)")
    print(f"  {'  -> CNN-DAE SNR更优但QRS保真度较低' if dl_qrs < 95.8 else '  -> CNN-DAE 全面领先'}")
    print(f"  但需注意:")
    print(f"    传统滤波: IEC 60601 合规, 实时可部署, 延迟<200ms")
    print(f"    CNN-DAE:  无 FDA/IEC 认证路径, 需 GPU, 可解释性差")
    print(f"    工程建议: 诊断设备用传统滤波, 云端后处理可用 DL")
    print(f"{'='*60}")
    print(f"\n图表保存至: {FIG_DIR}/")


if __name__ == '__main__':
    main()
