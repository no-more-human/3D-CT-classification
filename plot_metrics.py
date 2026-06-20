"""
分类性能评估可视化 —— 混淆矩阵热力图 + ROC-AUC 曲线。

用法：
  # 先跑 classify.py 导出结果
  python classify.py --model best_model_tl.pth --test_split test_split.json --save_json results.json

  # 再画图
  python plot_metrics.py --results results.json

输出：plots/classification_metrics.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ==========================================
# 参数
# ==========================================
parser = argparse.ArgumentParser(description="分类性能评估可视化")
parser.add_argument("--results", type=str, default="results.json",
                    help="classify.py --save_json 输出的结果文件")
args = parser.parse_args()

CLASS_NAMES = {0: "FD", 1: "OF"}


# ==========================================
# 核心函数
# ==========================================
def plot_metrics(results_file: str, output_path: str):
    if not os.path.exists(results_file):
        print(f"[ERROR] 结果文件不存在: {results_file}")
        print(f"  请先运行: python classify.py --model best_model_tl.pth --test_split test_split.json --save_json results.json")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = np.array([r["label"] for r in data])
    preds = np.array([r["pred"] for r in data])
    probs = np.array([r["probs"] for r in data])  # [N, 2]

    # OF 类的概率（ROC 用正类=OF 即 index=1）
    of_probs = probs[:, 1]

    n_fd = int((labels == 0).sum())
    n_of = int((labels == 1).sum())

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # ---- ROC & AUC ----
    fpr, tpr, _ = roc_curve(labels, of_probs)
    auc_score = auc(fpr, tpr)

    # ---- 绘图 ----
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("VSNet 分类性能评估 (测试集)", fontsize=14, fontweight="bold")

    # 左：混淆矩阵热力图
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FD", "OF"], yticklabels=["FD", "OF"],
        ax=ax_cm, cbar_kws={"label": "样本数"},
        linewidths=1, linecolor="white",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
    )
    ax_cm.set_title("混淆矩阵", fontsize=13, fontweight="bold")
    ax_cm.set_xlabel("预测类别", fontsize=11)
    ax_cm.set_ylabel("真实类别", fontsize=11)

    # 右：ROC 曲线
    ax_roc.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
                label=f"AUC = {auc_score:.4f}")
    ax_roc.plot([0, 1], [0, 1], color="gray", linewidth=1,
                linestyle="--", alpha=0.6, label="随机猜测 (AUC=0.50)")
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax_roc.set_title("ROC 曲线 (OF 为正类)", fontsize=13, fontweight="bold")
    ax_roc.legend(fontsize=10, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.05)

    # 汇总信息标注
    acc = 100.0 * (cm[0, 0] + cm[1, 1]) / cm.sum()
    fd_acc = 100.0 * cm[0, 0] / max(cm[0].sum(), 1)
    of_acc = 100.0 * cm[1, 1] / max(cm[1].sum(), 1)
    info = (f"总样本: {len(data)} | FD={n_fd}, OF={n_of}\n"
            f"整体 Acc: {acc:.1f}% | FD Acc: {fd_acc:.1f}% | OF Acc: {of_acc:.1f}%\n"
            f"AUC: {auc_score:.4f}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存: {output_path}")

    # 终端打印汇总
    print(f"\n{'='*50}")
    print(f"  测试集评估汇总")
    print(f"{'='*50}")
    print(f"  样本数: {len(data)} (FD={n_fd}, OF={n_of})")
    print(f"  整体 Acc: {acc:.2f}%")
    print(f"  FD  Acc:  {fd_acc:.2f}%")
    print(f"  OF  Acc:  {of_acc:.2f}%")
    print(f"  AUC:      {auc_score:.4f}")
    print(f"  混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"{'='*50}")


# ==========================================
# 主入口
# ==========================================
def main():
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plots", "classification_metrics.png"
    )
    plot_metrics(args.results, output_path)


if __name__ == "__main__":
    main()
