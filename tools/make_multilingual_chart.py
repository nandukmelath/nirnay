"""Model-card chart for Nirnay multilingual: MASSIVE intent accuracy per language, Nirnay vs Laya multilingual.
Palette: reference categorical slots (light surface, validated all-pairs); Laya keeps the aqua it has in the latency
chart, so an entity has one colour across the cards."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = r"D:\Nirnay\bench\results"
OUT = r"D:\Nirnay\hf_release_ml\assets\massive_by_language.png"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NIRNAY, LAYA = "#2a78d6", "#1baf7a"
NAMES = {"af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani", "bn": "Bengali", "cy": "Welsh",
         "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish",
         "fr": "French", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
         "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", "km": "Khmer",
         "kn": "Kannada", "ko": "Korean", "lv": "Latvian", "ml": "Malayalam", "mn": "Mongolian", "ms": "Malay",
         "my": "Burmese", "nb": "Norwegian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
         "ru": "Russian", "sl": "Slovenian", "sq": "Albanian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
         "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese",
         "zh-CN": "Chinese (simpl.)", "zh-TW": "Chinese (trad.)"}


def per_lang(name):
    d = json.load(open(os.path.join(RES, name), encoding="utf-8"))["suites"]
    return {k.split(".", 1)[1]: v["accuracy"] for k, v in d.items() if k.startswith("massive.")}


def main():
    n = per_lang("suites_multi__nirnay__nirnay-multilingual.json")
    l = per_lang("suites_multi__laya__laya-multilingual.json")
    langs = sorted(n, key=lambda x: (n[x], x))
    wins = sum(n[x] > l[x] + 0.005 for x in langs)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.4, 11.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = range(len(langs))
    for y, x in zip(ys, langs):
        ax.plot([l[x], n[x]], [y, y], color=AXIS, linewidth=1.5, zorder=1, solid_capstyle="round")
    ax.scatter([l[x] for x in langs], list(ys), s=46, color=LAYA, edgecolors=SURFACE, linewidths=1.6, zorder=3,
               label="Laya multilingual  (mean %.3f)" % (sum(l.values()) / len(l)))
    ax.scatter([n[x] for x in langs], list(ys), s=46, color=NIRNAY, edgecolors=SURFACE, linewidths=1.6, zorder=4,
               label="Nirnay multilingual  (mean %.3f)" % (sum(n.values()) / len(n)))
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAMES.get(x, x) for x in langs], color=INK2, fontsize=8.6)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.8, len(langs) - 0.2)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_xticklabels(["0", "10%", "20%", "30%", "40%", "50%", "60%", "70%"])
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.6)
    ax.set_xlabel("accuracy: 80 requests per language, all 60 intents as options, no examples", color=INK2, fontsize=9)
    fig.text(0.02, 0.985, "MASSIVE intent detection in 51 languages — higher is better", fontsize=12.5,
             fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.966, "Nirnay multilingual is higher in %d of %d languages. Same cases, same English question, "
             "measured on one CPU workstation." % (wins, len(langs)), fontsize=9, color=INK2, ha="left", va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.4)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.945, bottom=0.05)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print("wrote", OUT, "| wins", wins, "of", len(langs))


if __name__ == "__main__":
    main()
