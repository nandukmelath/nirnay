"""Hero chart for the model card: latency per call vs number of questions (measured on the Z840).
Palette: reference categorical slots 1-3 (validated all-pairs, light surface)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

RES = r"D:\Nirnay\bench\results"
OUTS = [r"D:\Nirnay\hf_release\assets\latency.png", r"D:\Nirnay\assets\latency.png"]   # model card + GitHub README
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua


def nirnay(name):
    d = json.load(open(os.path.join(RES, name)))
    return [min(v["%d_questions" % n]["warm"]["p50_ms"] for v in d["by_threads"].values() if "%d_questions" % n in v)
            for n in (1, 5, 10, 50)]


def laya_fastest():
    eager = json.load(open(os.path.join(RES, "laya_latency.json")))["models"]
    onnx = json.load(open(os.path.join(RES, "laya_latency_onnx.json")))["models"]
    out = []
    for n in (1, 5, 10, 50):
        k = "%d_questions" % n
        xs = [r[k]["p50_ms"] for m in eager.values() for r in m["by_threads"].values() if k in r]
        xs += [r[k]["p50_ms"] for m in onnx.values() for r in m["by_threads"].values() if k in r]
        out.append(min(xs))
    return out


def fmt_ms(v):
    return "%.1f s" % (v / 1000) if v >= 1000 else "%d ms" % round(v)


def main():
    qs = [1, 5, 10, 50]
    series = [("Nirnay v3.1 (default)", nirnay("nirnay_latency_v31.json"), SLOT[0]),
              ("Nirnay v1 (fast)", nirnay("nirnay_latency_v1.json"), SLOT[1]),
              ("Laya, fastest checkpoint + backend", laya_fastest(), SLOT[2])]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for z, (label, ys, color) in zip((5, 4, 3), series):   # Nirnay drawn above Laya where they meet at 1 question
        ax.plot(qs, ys, color=color, linewidth=2.0, solid_capstyle="round", zorder=z, label=label,
                marker="o", markersize=7.5, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2.0)
        # direct label at the line end: name in secondary ink, value in primary ink (never series-coloured text)
        ax.annotate("%s  %s" % (fmt_ms(ys[-1]), label.split(",")[0]), xy=(qs[-1], ys[-1]), xytext=(10, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 50 * 1.12)
    ax.set_ylim(35, 12000)
    ax.xaxis.set_major_locator(FixedLocator(qs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(q) for q in qs])
    yt = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([fmt_ms(v) for v in yt])
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9.5)
    ax.set_xlabel("questions answered in one call", color=INK2, fontsize=10)
    ax.set_ylabel("latency per call (p50, log scale)", color=INK2, fontsize=10)
    fig.text(0.015, 0.965, "Latency per call on a CPU-only workstation — lower is better", fontsize=13,
             fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.015, 0.905, "2x Xeon E5-2630 v3 (2014), warm calls, Laya's own latency workload. "
             "Laya re-reads the text for every question; Nirnay reads it once.", fontsize=9.5, color=INK2,
             ha="left", va="top")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.8)
    fig.subplots_adjust(left=0.10, right=0.74, top=0.82, bottom=0.13)
    for out in OUTS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
    print("wrote", OUTS, [s[1] for s in series])


if __name__ == "__main__":
    main()
