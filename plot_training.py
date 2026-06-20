"""
学习曲线对比可视化 —— 迁移学习 vs 从头训练的训练/验证指标曲线。

用法：
  # 自动匹配：tl 日志 + scratch 日志（如果都有）
  python plot_training.py

  # 单条曲线
  python plot_training.py --log_tl training_log_tl.json
  python plot_training.py --log_scratch training_log_scratch.json

  # 同时指定两条
  python plot_training.py --log_tl training_log_tl.json --log_scratch training_log_scratch.json

预期 JSON 日志格式（每行一个 epoch）：
  [{"epoch": 1, "loss": 0.6931, "val_acc": 50.0, "fd_acc": 48.0, "of_acc": 52.0}, ...]

输出：plots/training_curves.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 参数
# ==========================================
parser = argparse.ArgumentParser(description="训练学习曲线对比可视化")
parser.add_argument("--log_tl", type=str, default="training_log_tl.json",
                    help="迁移学习训练日志（默认: training_log_tl.json）")
parser.add_argument("--log_scratch", type=str, default="training_log_scratch.json",
                    help="从头训练日志（默认: training_log_scratch.json）")
args = parser.parse_args()


# ==========================================
# 核心函数
# ==========================================
def load_log(log_path: str) -> dict | None:
    """加载训练日志，返回 {epoch, loss, val_acc, fd_acc, of_acc} 的 Numpy 数组字典"""
    if not os.path.exists(log_path):
        print(f"[WARN] 日志文件不存在，跳过: {log_path}")
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not entries:
        print(f"[WARN] 日志为空: {log_path}")
        return None

    return {
        "epoch": np.array([e["epoch"] for e in entries]),
        "loss": np.array([e["loss"] for e in entries]),
        "val_acc": np.array([e["val_acc"] for e in entries]),
        "fd_acc": np.array([e["fd_acc"] for e in entries]),
        "of_acc": np.array([e["of_acc"] for e in entries]),
    }


def plot_learning_curves(log_tl: dict | None, log_scratch: dict | None, output_path: str):
    """双行图：上=Loss 曲线，下=准确率曲线"""

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("VSNet 训练曲线: 迁移学习 vs 从头训练", fontsize=14, fontweight="bold")

    has_tl = log_tl is not None
    has_scratch = log_scratch is not None

    if not has_tl and not has_scratch:
        print("[ERROR] 没有可用的日志文件")
        return

    # ---- 上子图：Loss ----
    if has_tl:
        ax_loss.plot(log_tl["epoch"], log_tl["loss"], color="#2196F3", linewidth=2,
                     marker="o", markersize=4, label="迁移学习 (TL)")
    if has_scratch:
        ax_loss.plot(log_scratch["epoch"], log_scratch["loss"], color="#FF5722", linewidth=2,
                     marker="s", markersize=4, label="从头训练 (Scratch)")

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Training Loss", fontsize=11)
    ax_loss.set_title("训练损失曲线", fontsize=12)
    ax_loss.legend(fontsize=10, loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # ---- 下子图：准确率 ----
    if has_tl:
        ax_acc.plot(log_tl["epoch"], log_tl["val_acc"], color="#2196F3", linewidth=2.5,
                    marker="o", markersize=4, label="整体 Acc (TL)")
        ax_acc.plot(log_tl["epoch"], log_tl["fd_acc"], color="#2196F3", linewidth=1,
                    linestyle="--", marker="x", markersize=3, alpha=0.7, label="FD Acc (TL)")
        ax_acc.plot(log_tl["epoch"], log_tl["of_acc"], color="#2196F3", linewidth=1,
                    linestyle=":", marker="+", markersize=3, alpha=0.7, label="OF Acc (TL)")

    if has_scratch:
        ax_acc.plot(log_scratch["epoch"], log_scratch["val_acc"], color="#FF5722", linewidth=2.5,
                    marker="s", markersize=4, label="整体 Acc (Scratch)")
        ax_acc.plot(log_scratch["epoch"], log_scratch["fd_acc"], color="#FF5722", linewidth=1,
                    linestyle="--", marker="x", markersize=3, alpha=0.7, label="FD Acc (Scratch)")
        ax_acc.plot(log_scratch["epoch"], log_scratch["of_acc"], color="#FF5722", linewidth=1,
                    linestyle=":", marker="+", markersize=3, alpha=0.7, label="OF Acc (Scratch)")

    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_title("验证集准确率曲线", fontsize=12)
    ax_acc.legend(fontsize=8, loc="lower right", ncol=2)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, 105)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 打印汇总
    print(f"\n{'='*50}")
    if has_tl:
        best_tl = log_tl["val_acc"].max()
        best_ep_tl = log_tl["epoch"][log_tl["val_acc"].argmax()]
        print(f"  迁移学习最优 Val Acc: {best_tl:.2f}% (Epoch {best_ep_tl})")
    if has_scratch:
        best_scr = log_scratch["val_acc"].max()
        best_ep_scr = log_scratch["epoch"][log_scratch["val_acc"].argmax()]
        print(f"  从头训练最优 Val Acc: {best_scr:.2f}% (Epoch {best_ep_scr})")
    print(f"{'='*50}")
    print(f"[OK] 已保存: {output_path}")


# ==========================================
# 主入口
# ==========================================
def main():
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plots", "training_curves.png"
    )

    log_tl = load_log(args.log_tl)
    log_scratch = load_log(args.log_scratch)

    plot_learning_curves(log_tl, log_scratch, output_path)


if __name__ == "__main__":
    main()
