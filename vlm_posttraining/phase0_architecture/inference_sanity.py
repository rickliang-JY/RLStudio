"""Phase 0 — 环境 & 架构 sanity check.

在 AutoDL 实例上跑通 Qwen2-VL-2B-Instruct 的一次图片推理,并打印 VLM 三大部件
(ViT 视觉编码器 / connector / LLM)的张量流形状,确认:
  1) GPU 被识别、CUDA kernel 可用(尤其 5090/sm_120 需要 cu128 镜像);
  2) 模型能加载进显存并正确感知图片内容。

用法:
    pip install transformers accelerate qwen-vl-utils pillow modelscope
    python inference_sanity.py

国内下载慢时择一:
    export HF_ENDPOINT=https://hf-mirror.com      # HF 镜像
    或在 AutoDL 上  source /etc/network_turbo      # 学术加速
    或设 USE_MODELSCOPE=1 用 ModelScope 下载(见下)
"""

import os

import torch
from PIL import Image, ImageDraw


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def print_env() -> None:
    print("=" * 60)
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"device           : {torch.cuda.get_device_name(0)}")
        print(f"capability       : sm_{cap[0]}{cap[1]}  (5090 应为 sm_120)")
        print(f"bf16 supported   : {torch.cuda.is_bf16_supported()}")
    print("=" * 60)


def make_test_image(path: str = "sanity_image.png") -> str:
    """生成一张合成图(白底红色圆),不依赖网络即可测真实感知。"""
    img = Image.new("RGB", (336, 336), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([88, 88, 248, 248], fill="red", outline="black", width=4)
    img.save(path)
    return path


def resolve_model_path() -> str:
    """可选用 ModelScope 下载(国内更稳),否则走 HuggingFace。"""
    if os.environ.get("USE_MODELSCOPE") == "1":
        from modelscope import snapshot_download

        return snapshot_download(MODEL_ID)
    return MODEL_ID


def main() -> None:
    print_env()
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU —— 请在 AutoDL GPU 实例上运行。")

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model_path = resolve_model_path()
    print(f"\n加载模型: {model_path} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    image_path = make_test_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "What shape and what color is in this image? Answer briefly."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    # ---- 打印 VLM 三大部件的张量流 ----
    print("\n" + "-" * 60)
    print("[输入张量]")
    print(f"  input_ids        : {tuple(inputs['input_ids'].shape)}  (图文混合 token 序列)")
    print(f"  pixel_values     : {tuple(inputs['pixel_values'].shape)}  (ViT 吃进的 patch)")
    if "image_grid_thw" in inputs:
        thw = inputs["image_grid_thw"][0].tolist()
        n_vis = (thw[0] * thw[1] * thw[2]) // (model.config.vision_config.spatial_merge_size ** 2)
        print(f"  image_grid_thw   : {thw}  (T,H,W of patch grid)")
        print(f"  视觉 token 数    : ~{n_vis}  (connector 投影进 LLM 词向量空间的数量)")
    print("-" * 60)

    print("\n生成回答 ...")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print("\n" + "=" * 60)
    print(f"模型回答: {answer.strip()}")
    print("=" * 60)
    print("\n若回答提到 red / circle,则 ViT→connector→LLM 全链路感知正常。")
    print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
