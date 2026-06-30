# 环境配置 · RTX 5090(Blackwell / sm_120)

> 5090 是 Blackwell 架构(CUDA compute capability **12.0 / sm_120**),非常新。
> **最大的坑不是显存,是软件栈版本**——老版本没有对应 kernel,装上会直接报
> `CUDA error: no kernel image is available for execution on the device`。

## 硬性要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| NVIDIA 驱动 | R570+ | 支持 Blackwell |
| CUDA | **12.8+** | sm_120 从 CUDA 12.8 起被官方 kernel 支持 |
| PyTorch | **2.7+(cu128 构建)** | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| flash-attention | 较新版或源码编译 | Phase 1+ 需要;sm_120 wheel 不一定现成 |
| bitsandbytes / AWQ / GPTQ kernel | 最新 | Phase 7 量化;注意 sm_120 兼容性 |

## 运行环境:AutoDL(云 Linux GPU)

训练在 **AutoDL** 云实例上跑(Linux,坑最少)。链路全部打通后,再尝试搬到 **molab(marimo lab)**
做在线可交互版,延续 RLStudio「浏览器里直接跑」的理念。

### AutoDL 选实例 / 镜像
- **确认实例 GPU 型号**:据此选镜像。若是 5090(sm_120),必须选 **CUDA 12.8 + PyTorch 2.7+** 的镜像;
  若是 4090/A100 等(sm_89/sm_80),cu121/cu124 镜像即可。AutoDL 的官方 PyTorch 镜像已预装好驱动+CUDA+torch,
  省掉本地装 cu128 的麻烦。
- **数据盘**:模型/数据/checkpoint 放 `/root/autodl-tmp`(数据盘,空间大);系统盘小,别往里塞权重。
- **下载加速**:
  - 国内拉 HuggingFace 慢 → `source /etc/network_turbo`(AutoDL 学术加速)或 `export HF_ENDPOINT=https://hf-mirror.com`。
  - **ms-swift 默认走 ModelScope**,国内更稳,Phase 1 起基本无需翻墙。

### 安装顺序

```bash
# 0) 验证镜像自带的 torch 已识别 GPU(AutoDL 镜像通常已装好,跳过自己装 torch)
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 5090 期望: 2.7+  NVIDIA GeForce RTX 5090  (12, 0)
# 若报 'no kernel image' → 镜像 CUDA/torch 版本太老,换 cu128 镜像

# 1) Phase 0 推理依赖(最小集)
pip install transformers accelerate qwen-vl-utils pillow modelscope

# 2) 框架(Phase 1 起)
pip install ms-swift -U
```

## Phase 0 验证目标

在装框架之前,先用最小依赖(步骤 0–1)把 `Qwen2-VL-2B-Instruct` 加载起来跑一次图片推理,
并打印各部件张量形状。这一步通了,说明实例的 CUDA/显存/kernel 全通,后面只剩框架层的坑。
脚本:[`../phase0_architecture/inference_sanity.py`](../phase0_architecture/inference_sanity.py)。
