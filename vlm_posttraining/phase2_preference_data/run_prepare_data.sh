#!/usr/bin/env bash
# Phase 2 — 偏好数据。直接用现成的 RLAIF-V(ms-swift 已注册为 `swift/RLAIF-V-Dataset`)。
# 本脚本只做「下载 + 看清格式 + 统计」,真正喂给训练的是 Phase 3/4 里的 --dataset。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

python - <<'PY'
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('swift/RLAIF-V-Dataset', split='train')
print("样本数:", len(ds))
ex = ds[0]
print("字段:", list(ex.keys()))
for k in ('question', 'chosen', 'rejected'):
    if k in ex:
        v = str(ex[k])
        print(f"\n[{k}] {v[:300]}{'...' if len(v) > 300 else ''}")
print("\n下游用法: Phase 3/4 训练脚本里 --dataset 'swift/RLAIF-V-Dataset#<N>' 即可,无需手动转换。")
PY
echo ">>> Phase 2 完成。偏好数据就绪,进 Phase 3。"
