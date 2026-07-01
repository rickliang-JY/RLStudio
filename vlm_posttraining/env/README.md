# 环境配置 · Blackwell GPU(sm_120)

> 本项目用的 GPU 都是 **Blackwell 架构**(CUDA compute capability **12.0 / sm_120**):
> molab 的 RTX PRO 6000、以及 RTX 5090。非常新,**最大的坑不是显存,是软件栈版本**——
> 老版本没有对应 kernel,装上会直接报
> `CUDA error: no kernel image is available for execution on the device`。

## 硬性要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| NVIDIA 驱动 | R570+ | 支持 Blackwell |
| CUDA | **12.8+** | sm_120 从 CUDA 12.8 起被官方 kernel 支持 |
| PyTorch | **2.7+(cu128 构建)** | molab 已预装 2.12/cu130,**勿重装** |
| flash-attention | 较新版或源码编译 | Phase 1+ 需要;sm_120 wheel 不一定现成,不装则退回 SDPA |
| bitsandbytes / AWQ / GPTQ kernel | 最新 | Phase 7 量化;注意 sm_120 兼容性 |

## 运行环境:molab(主)/ AutoDL(备)

**当前主线在 [molab](https://molab.marimo.io)** 上跑:单张 **RTX PRO 6000 Blackwell(96GB)**,
预装 **torch 2.12 / cu130**,天然过掉 Blackwell 环境坑;`/marimo` 是 7T 持久盘,checkpoint 直接落盘跨 session 复用。
备选是 **AutoDL** 云 Linux 实例(需自己选 cu128 镜像)。`env/common.sh` 会自动探测平台
(有 `/marimo` 用 molab 档,否则用 AutoDL 档),同一份脚本两边通用。

### molab(主路径)

- **Phase 0 引导**:`bash env/setup_molab.sh` —— 只装推理上层依赖,**不碰预装 torch**,走 HuggingFace 直连。
- **持久化**:所有产物落 `/marimo/vlm/`(common.sh 自动指过去),关机重开还在。
- **session 限制**:单次最长 12h、闲置 90min 自动停机。长训练拆成可续跑的小段,或用 `--resume_from_checkpoint`。
- **下载源**:统一走 **ModelScope**(`USE_HF=0`,common.sh 默认)。本项目数据集(coco_caption、RLAIF-V 偏好数据)
  是 ModelScope 独有且带 ms-swift 预处理格式;molab 美区拉 ModelScope 稍慢但能通。模型也从 ModelScope 下。

### AutoDL(备用路径)

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
