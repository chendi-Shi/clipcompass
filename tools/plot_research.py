"""Export actual offline results; error bars are across-seed SD, not CI."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(output):
    out = Path(output)
    main = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    joint = json.loads((out / "joint_results.json").read_text(encoding="utf-8"))
    curves = json.loads((out / "curves.json").read_text(encoding="utf-8"))
    names = list(main["summary"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, cohort in zip(axes.flat, ["warm", "cold", "joint"]):
        values = [joint["summary"][n]["ndcg@20"] if cohort == "joint" else main["summary"][n][cohort]["ndcg@20"] for n in names]
        ax.barh(names[::-1], [v["mean"] for v in values][::-1], xerr=[v["std"] for v in values][::-1], color="#4373a9", capsize=3)
        ax.set_title(f"{cohort.title()} observed-exposure ranking")
        ax.set_xlabel("Test NDCG@20 (mean +/- seed SD)")
        ax.grid(axis="x", alpha=.2)
    ax = axes[1, 1]
    for kind in ["bpr_mf", "lightgcn", "content_gate", "content_ssl"]:
        series = [curves[f"{kind}_seed{s}"] for s in main["seeds"]]
        epochs = [r["epoch"] for r in series[0]]
        metrics = np.array([[r["validation_warm_ndcg@20"] for r in rows] for rows in series])
        mean, sd = metrics.mean(0), metrics.std(0, ddof=1)
        ax.plot(epochs, mean, label=kind)
        ax.fill_between(epochs, mean-sd, mean+sd, alpha=.12)
    ax.set_title("Warm validation checkpoint selection")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NDCG@20")
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle("KuaiRec / 3 seeds / fixed protocol / offline metrics only", fontsize=15)
    fig.savefig(out / "recommendation_results.png", dpi=180)
    fig.savefig(out / "recommendation_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/research/kuairec")
    run(p.parse_args().output)
