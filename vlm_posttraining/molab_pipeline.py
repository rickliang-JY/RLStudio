import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # VLM 后训练全流程 · molab 运行台

        在 **molab(RTX PRO 6000 / torch 2.12·cu130)** 上一键跑通 Qwen2-VL-2B 的后训练全生命周期:
        **SFT → 偏好对齐(DPO/ORPO/KTO)→ 奖励模型 → 在线 RL(GRPO)→ 评测 → 量化部署**。

        本 notebook 只做**编排**:每个阶段一个按钮,点一下就调用仓库里已写好的 shell 脚本
        (`vlm_posttraining/**/run_*.sh`),输出打印在按钮下方。真正的逻辑在脚本里,单一事实来源。

        **怎么用:** 从上往下,每一步点一次按钮、等它跑完(下方出现 `[exit 0]`)再点下一步。
        产物全部落在持久盘 `/marimo/vlm/`,关机重开不丢。
        """
    )
    return


@app.cell(hide_code=True)
def _(os, subprocess):
    # 平台探测 + 路径:molab 用 /marimo 持久盘
    DATA_ROOT = "/marimo" if os.path.isdir("/marimo") else os.path.expanduser("~/vlm-data")
    os.makedirs(DATA_ROOT, exist_ok=True)
    REPO = os.path.join(DATA_ROOT, "RLStudio")
    PROJ = os.path.join(REPO, "vlm_posttraining")

    def sh(cmd, cwd=None):
        """跑一条 shell 命令,实时把输出打印到 cell 下方。返回退出码。"""
        print(f"$ {cmd}\n", flush=True)
        p = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in p.stdout:
            print(line, end="", flush=True)
        p.wait()
        print(f"\n[exit {p.returncode}]", flush=True)
        return p.returncode
    return DATA_ROOT, PROJ, REPO, sh


@app.cell(hide_code=True)
def _(DATA_ROOT, PROJ, mo):
    mo.md(
        f"""
        ## ① 环境准备(Phase 0)

        数据盘:`{DATA_ROOT}` · 项目目录:`{PROJ}`
        """
    )
    return


@app.cell
def _(mo):
    clone_btn = mo.ui.run_button(label="① 拉取 / 更新代码")
    clone_btn
    return (clone_btn,)


@app.cell
def _(DATA_ROOT, REPO, clone_btn, mo, os, sh):
    mo.stop(not clone_btn.value, mo.md("_点上方按钮把仓库克隆到持久盘。_"))
    if os.path.isdir(os.path.join(REPO, ".git")):
        sh("git pull", cwd=REPO)
    else:
        sh("git clone https://github.com/rickliang-JY/RLStudio.git", cwd=DATA_ROOT)
    return


@app.cell
def _(mo):
    p0_btn = mo.ui.run_button(label="② Phase 0:验证环境 + 推理 sanity")
    p0_btn
    return (p0_btn,)


@app.cell
def _(PROJ, mo, p0_btn, sh):
    mo.stop(not p0_btn.value, mo.md("_验证 Blackwell 环境并跑通一次 2B 图片推理(不装训练框架、不碰预装 torch)。_"))
    sh("bash env/setup_molab.sh", cwd=PROJ)
    return


@app.cell
def _(mo):
    sw_btn = mo.ui.run_button(label="③ 安装 ms-swift(SFT/DPO/RM 够用)")
    sw_btn
    return (sw_btn,)


@app.cell
def _(PROJ, mo, sh, sw_btn):
    mo.stop(not sw_btn.value, mo.md("_装训练框架 ms-swift。Phase 5/6/7 再补 `full`(见第 ③ 部分)。_"))
    sh("bash env/setup_swift.sh", cwd=PROJ)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ② 训练全流程(Phase 1–4)

        每步产物 merge 成完整模型,供下一步与最终评测使用。命名 `qwen2vl2b-<phase>`。
        """
    )
    return


@app.cell
def _(mo):
    p1_btn = mo.ui.run_button(label="Phase 1 · SFT(视觉指令微调,LoRA)")
    p1_btn
    return (p1_btn,)


@app.cell
def _(PROJ, mo, p1_btn, sh):
    mo.stop(not p1_btn.value, mo.md("_在 coco_caption 上做 LoRA SFT → merge 出 `qwen2vl2b-sft`。_"))
    sh("bash phase1_sft/run_sft.sh", cwd=PROJ)
    return


@app.cell
def _(mo):
    p2_btn = mo.ui.run_button(label="Phase 2 · 查看偏好数据(RLAIF-V)")
    p2_btn
    return (p2_btn,)


@app.cell
def _(PROJ, mo, p2_btn, sh):
    mo.stop(not p2_btn.value, mo.md("_加载 RLAIF-V 偏好数据集,打印统计与样例(不训练)。_"))
    sh("bash phase2_preference_data/run_prepare_data.sh", cwd=PROJ)
    return


@app.cell
def _(mo):
    rlhf_type = mo.ui.dropdown(["dpo", "orpo", "kto"], value="dpo", label="对齐方法")
    p3_btn = mo.ui.run_button(label="Phase 3 · 运行所选对齐")
    mo.hstack([rlhf_type, p3_btn], justify="start")
    return p3_btn, rlhf_type


@app.cell
def _(PROJ, mo, p3_btn, rlhf_type, sh):
    mo.stop(not p3_btn.value, mo.md("_在 SFT 模型上做偏好对齐。三种各跑一次可做对比(依次选 dpo/orpo/kto 各点一遍)。_"))
    sh(f"bash phase3_dpo/run_align.sh {rlhf_type.value}", cwd=PROJ)
    return


@app.cell
def _(mo):
    p4_btn = mo.ui.run_button(label="Phase 4 · 奖励模型(RM)")
    p4_btn
    return (p4_btn,)


@app.cell
def _(PROJ, mo, p4_btn, sh):
    mo.stop(not p4_btn.value, mo.md("_训练打分员模型 `qwen2vl2b-rm`,给在线 RL 用。_"))
    sh("bash phase4_reward_model/run_rm.sh", cwd=PROJ)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ③ 在线 RL / 评测 / 部署(Phase 5–7,需 vLLM)

        这三步依赖 vLLM。**molab 的 torch 2.12/cu130 很新,vllm wheel 可能没现成或想反装 torch** ——
        先点下面的「安装 full」,若报错或 torch 被改动,把输出发我再单独处理。
        """
    )
    return


@app.cell
def _(mo):
    swfull_btn = mo.ui.run_button(label="④ 安装 vLLM + EvalScope(full)")
    swfull_btn
    return (swfull_btn,)


@app.cell
def _(PROJ, mo, sh, swfull_btn):
    mo.stop(not swfull_btn.value, mo.md("_装 vllm + evalscope。留意结尾有没有 torch 被改动的告警。_"))
    sh("bash env/setup_swift.sh full", cwd=PROJ)
    return


@app.cell
def _(mo):
    p5_btn = mo.ui.run_button(label="Phase 5 · 在线 RL(GRPO)")
    p5_btn
    return (p5_btn,)


@app.cell
def _(PROJ, mo, p5_btn, sh):
    mo.stop(not p5_btn.value, mo.md("_用 RM 打分跑 GRPO(vLLM colocate 加速生成)→ `qwen2vl2b-grpo`。_"))
    sh("bash phase5_online_rl/run_grpo.sh", cwd=PROJ)
    return


@app.cell
def _(mo):
    p6_btn = mo.ui.run_button(label="Phase 6 · 评测(横扫各 checkpoint)")
    p6_btn
    return (p6_btn,)


@app.cell
def _(PROJ, mo, p6_btn, sh):
    mo.stop(not p6_btn.value, mo.md("_对 base/sft/dpo/.../grpo 跑 POPE·MMBench·MM-Vet,产出对比表。_"))
    sh("bash phase6_eval/run_eval.sh", cwd=PROJ)
    return


@app.cell
def _(mo):
    quant_action = mo.ui.dropdown(["quant", "deploy", "bench"], value="quant", label="动作")
    p7_btn = mo.ui.run_button(label="Phase 7 · 运行所选")
    mo.hstack([quant_action, p7_btn], justify="start")
    return p7_btn, quant_action


@app.cell
def _(PROJ, mo, p7_btn, quant_action, sh):
    mo.stop(not p7_btn.value, mo.md("_quant=AWQ/GPTQ 量化导出;deploy=vLLM 起服务;bench=吞吐/延迟压测。_"))
    sh(f"bash phase7_quant_deploy/run_quant_deploy.sh {quant_action.value}", cwd=PROJ)
    return


@app.cell
def _():
    import marimo as mo
    import os
    import subprocess

    return mo, os, subprocess


if __name__ == "__main__":
    app.run()
