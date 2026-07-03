# ECG 数字滤波去噪 —— 从工业级数字滤波器到深度学习

数字信号处理课程报告。北京科技大学通信工程专业。

## 项目结构

```
├── ecg_denoising.py            # 传统滤波管线
├── ecg_denoising_dl.py         # 深度学习 CNN-DAE
└── README.md
```

## 传统滤波管线

```
IIR 陷波 (50Hz, Q=30) → 中值滤波基线校正 (200ms) → Chebyshev II 低通 (100Hz, 60dB)
```

核心指标：SNR 改善 **+13.9 dB**，QRS 保真度 **95.8%**，延迟 **101 ms**。

### 运行

```bash
pip install numpy scipy matplotlib
python ecg_denoising.py
```

## 深度学习 CNN-DAE

1D 卷积去噪自编码器，216,817 参数，U-Net 架构。

核心指标：SNR 改善 **+22.7 dB**，QRS 保真度 **91.4%**。

### 运行

```bash
pip install torch numpy scipy matplotlib
python ecg_denoising_dl.py --headless    # 云服务器
python ecg_denoising_dl.py               # 本地
```

## 实验结论

传统滤波链路在 ECG 去噪任务上**整体最优**：QRS 保真度最高、IEC 60601 合规、可实时部署、无需 GPU。CNN-DAE 在噪声抑制指标上领先，但 QRS 保真度较低（MSE Loss 训练的固有代价），仅推荐用于对波形幅值不敏感的非诊断场景。

## 参考教材

- 高西全《数字信号处理（第五版）》
- Oppenheim《离散时间信号处理（第3版）》

## License

MIT
