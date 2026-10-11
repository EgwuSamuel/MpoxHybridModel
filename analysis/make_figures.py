"""Build the manuscript figures and the summary of reported numbers from the saved results.

Usage (from the repository root):
    python analysis/make_figures.py

Reads   results/cross_validation/  and  results/final_model/
Writes  figures/*.png  (600 dpi)   and  analysis/summary_numbers.json
No model is trained here: everything is derived from the stored out-of-fold predictions and tables.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CV, FM, FIG = ROOT / "results" / "cross_validation", ROOT / "results" / "final_model", ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
REFERENCE = "MoViT-SAM"
# One colour and one marker per model family (colour is never the only cue).
FAMILY_COLOUR = {"CNN": "#2a78d6", "Transformer": "#eb6834", "Hybrid": "#1baf7a"}
FAMILY_MARKER = {"CNN": "o", "Transformer": "s", "Hybrid": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DISPLAY = {"MoViT-SAM": "Hybrid + SAM", "Hybrid+Adam": "Hybrid + Adam", "Hybrid+H-SAM": "Hybrid + H-SAM",
           "EfficientNetV2": "EfficientNetV2-B0", "MobileNetV2": "MobileNetV2", "ResNet50": "ResNet50",
           "DenseNet121": "DenseNet121", "ViT-B/16": "ViT-B/16", "DINO ViT-B/16": "DINO ViT-B/16"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight",
                     "legend.frameon": False})


def safe(name):
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", f"figures/{name}.png")


# ----------------------------------------------------------------------------- data
labels = np.load(CV / "labels.npy")
groups = np.load(CV / "groups.npy", allow_pickle=True)
fold_id = np.load(CV / "fold_id.npy")
main = pd.read_csv(CV / "cmp_table_main.csv")
perclass = pd.read_csv(CV / "cmp_table_perclass.csv")
leak = pd.read_csv(CV / "cmp_table_leakage.csv")
pairs = pd.read_csv(CV / "cmp_pairwise_mcnemar.csv")
friedman = pd.read_csv(CV / "cmp_friedman.csv")
folds = pd.read_csv(CV / "cv_per_fold_all_models.csv")
dca = pd.read_csv(CV / "dca.csv")
eff = pd.read_csv(FM / "cmp_table_efficiency.csv")
OOF = {m: np.load(CV / f"oof_{safe(m)}.npy") for m in main.Model}
for m, p in OOF.items():       # the stored predictions must reproduce the stored table
    assert abs((p.argmax(1) == labels).mean() - main.set_index("Model").loc[m, "OOF acc"]) < 5e-5, m
# use the exact pooled accuracy (the CSV is rounded to four decimals) so that figures and text round identically
main["OOF acc"] = [float((OOF[m].argmax(1) == labels).mean()) for m in main.Model]
main["Correct"] = [int((OOF[m].argmax(1) == labels).sum()) for m in main.Model]
main = main.sort_values("OOF acc", ascending=False).reset_index(drop=True)
# the leakage-free rows of the leakage table are the same pooled accuracies
_clean = leak.Protocol.str.startswith("Patient-level")
leak.loc[_clean, "Accuracy"] = [main.set_index("Model").loc[m, "OOF acc"] for m in leak.Model[_clean]]


# ----------------------------------------------------------------------------- patient-level uncertainty
N_BOOT, SEED = 2000, 42
PATIENTS = np.unique(groups)
_rows = {g: np.where(groups == g)[0] for g in PATIENTS}
_rng = np.random.default_rng(SEED)
# the same 2,000 resamples of patients (with all their images) are used for every model
RESAMPLES = [np.concatenate([_rows[g] for g in _rng.choice(PATIENTS, len(PATIENTS))]) for _ in range(N_BOOT)]


def accuracy(y, p):
    return float((p.argmax(1) == y).mean())


def macro_auc(y, p):
    if len(np.unique(y)) < p.shape[1]:
        return float("nan")
    return float(roc_auc_score(np.eye(p.shape[1])[y], p, average="macro", multi_class="ovr"))


def patient_ci(fn, p):
    """95% confidence interval from a bootstrap that resamples patients, not images."""
    vals = [fn(labels[i], p[i]) for i in RESAMPLES]
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


def per_patient(p):
    """One prediction per patient: mean probability over the patient's images (as in notebook 01)."""
    pt = pd.DataFrame(p).groupby(groups).mean()
    y = pd.Series(labels).groupby(groups).first().loc[pt.index].values
    return y, pt.values


def mcnemar(y, pa, pb):
    """Exact McNemar test on paired predictions."""
    a_ok, b_ok = pa == y, pb == y
    n10, n01 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    return n10, n01, (1.0 if n10 + n01 == 0 else float(stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue))


# the table of notebook 01 holds image-level bootstrap intervals; the paper reports patient-level ones
for _m in main.Model:
    _i = main.index[main.Model == _m][0]
    main.loc[_i, ["OOF acc CI low", "OOF acc CI high"]] = patient_ci(accuracy, OOF[_m])
    main.loc[_i, ["AUC CI low", "AUC CI high"]] = patient_ci(macro_auc, OOF[_m])


# ----------------------------------------------------------------------------- late fusion
# Is the gain of the hybrids specific to fusing features in one head? Average the out-of-fold
# probabilities of the two backbones trained separately (no further training) and compare.
FUSION = ("MobileNetV2", "ViT-B/16")
HYBRIDS = ["Hybrid+H-SAM", "Hybrid+Adam", "MoViT-SAM"]
SINGLES = ["EfficientNetV2", "MobileNetV2", "DenseNet121", "ResNet50", "ViT-B/16"]


def late_fusion():
    p = (OOF[FUSION[0]] + OOF[FUSION[1]]) / 2
    py, pp = per_patient(p)
    fold_acc = [accuracy(labels[fold_id == k], p[fold_id == k]) for k in sorted(set(fold_id))]
    out = {"members": list(FUSION), "accuracy": accuracy(labels, p), "correct": int((p.argmax(1) == labels).sum()),
           "accuracy_ci": patient_ci(accuracy, p), "macro_auc": macro_auc(labels, p),
           "macro_auc_ci": patient_ci(macro_auc, p),
           "macro_f1": float(f1_score(labels, p.argmax(1), average="macro", zero_division=0)),
           "patient_accuracy": accuracy(py, pp), "fold_acc_mean": float(np.mean(fold_acc)),
           "fold_acc_sd": float(np.std(fold_acc, ddof=1)), "versus": {}}
    for m in HYBRIDS + list(FUSION):
        qy, qp = per_patient(OOF[m])
        i10, i01, ip = mcnemar(labels, p.argmax(1), OOF[m].argmax(1))
        q10, q01, qpv = mcnemar(py, pp.argmax(1), qp.argmax(1))
        out["versus"][m] = {"image_fusion_only": i10, "image_other_only": i01, "image_p": ip,
                            "patient_fusion_only": q10, "patient_other_only": q01, "patient_p": qpv}
    pairs_ = []
    for i, a in enumerate(SINGLES):
        for b in SINGLES[i + 1:]:
            q = (OOF[a] + OOF[b]) / 2
            pairs_.append({"Model A": a, "Model B": b, "Accuracy of the average": accuracy(labels, q),
                           "Macro-AUC of the average": macro_auc(labels, q),
                           "Accuracy A": accuracy(labels, OOF[a]), "Accuracy B": accuracy(labels, OOF[b])})
    tab = pd.DataFrame(pairs_).sort_values("Accuracy of the average", ascending=False)
    tab.round(4).to_csv(ROOT / "analysis" / "late_fusion_pairs.csv", index=False)
    out["all_pairs"] = tab.to_dict("records")
    print("wrote analysis/late_fusion_pairs.csv")
    return out


LATE = late_fusion()


# ----------------------------------------------------------------------------- calibration helpers
def apply_T(p, T):
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_T(p, y):
    nll = lambda T: -np.mean(np.log(np.clip(apply_T(p, T)[np.arange(len(y)), y], 1e-12, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def ece(p, y, n_bins=10):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, cf = (pred[m] == y[m]).mean(), conf[m].mean()
            e += m.mean() * abs(acc - cf); pts.append((cf, acc))
    return float(e), np.array(pts)


def cross_fitted(p):
    out = np.zeros_like(p); Ts = []
    for k in sorted(set(fold_id)):
        tr, va = fold_id != k, fold_id == k
        T = fit_T(p[tr], labels[tr]); Ts.append(T); out[va] = apply_T(p[va], T)
    return out, Ts


# ----------------------------------------------------------------------------- Fig. 1 study design
def fig_study_design():
    fig, ax = plt.subplots(figsize=(7.0, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#f3f6fb", ec="#2a78d6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.6, fontweight="bold")
        if body:
            ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top", fontsize=7.6, color=MUTED, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED))

    box(3, 82, 44, 15, "MSLD v2.0 original images",
        "755 images, 524 patients, 6 classes\naugmented copies discarded")
    box(53, 82, 44, 15, "Integrity audit",
        "perceptual-hash near-duplicates\npatient identity, skin tone (ITA)")
    arrow(47.6, 89.5, 52.4, 89.5)
    box(3, 60, 94, 15, "Patient-level five-fold stratified-group cross-validation",
        "identical folds for every model; augmentation on training batches only;\n"
        "early stopping on an inner split of the training patients", fc="#f6f5f1", ec="#898781")
    arrow(50, 81.6, 50, 75.6)
    fams = [("CNN backbones", "MobileNetV2\nEfficientNetV2-B0\nResNet50\nDenseNet121", "CNN"),
            ("Transformer backbones", "ViT-B/16\nDINO ViT-B/16", "Transformer"),
            ("Hybrid CNN + transformer", "MobileNetV2 + ViT-B/16 with\nAdam, SAM or H-SAM", "Hybrid")]
    for i, (t, b, f) in enumerate(fams):
        x = 3 + i * 32.2
        box(x, 33, 29.6, 21.5, t, b, fc="#ffffff", ec=FAMILY_COLOUR[f])
        arrow(x + 14.8, 59.6, x + 14.8, 55.1)
        arrow(x + 14.8, 32.6, x + 14.8, 28.6)
    box(3, 15, 94, 13, "Pooled out-of-fold predictions for all 755 images (nine models)", fc="#f6f5f1", ec="#898781")
    outs = ["Comparison\nCIs, Friedman,\nMcNemar + Holm", "Calibration\ncross-fitted\ntemperature",
            "Leakage\nProtocols A, B\nper family", "Efficiency\nsize and\nCPU latency",
            "Reference hybrid\nclinical utility,\nexport, v1.0 check"]
    for i, t in enumerate(outs):
        x = 3 + i * 19.2
        head, _, rest = t.partition("\n")
        ax.text(x + 8.6, 9.6, head, ha="center", va="top", fontsize=8.0, fontweight="bold")
        ax.text(x + 8.6, 6.2, rest, ha="center", va="top", fontsize=7.2, color=MUTED, linespacing=1.3)
        arrow(x + 8.6, 14.6, x + 8.6, 10.6)
    ax.text(50, 19.4, "accuracy, macro-AUC, macro-F1 at image level and patient level",
            ha="center", va="center", fontsize=7.6, color=MUTED)
    save(fig, "fig1_study_design")


# ----------------------------------------------------------------------------- Fig. 3 model comparison
def fig_model_comparison():
    d = main.sort_values("OOF acc").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    specs = [("OOF acc", "OOF acc CI low", "OOF acc CI high", 100, "Pooled out-of-fold accuracy (%)", "{:.1f}"),
             ("Macro-AUC", "AUC CI low", "AUC CI high", 1, "Macro-averaged AUC", "{:.3f}")]
    for ax, (v, lo, hi, k, xl, fmt) in zip(axes, specs):
        for i, r in d.iterrows():
            c, mk = FAMILY_COLOUR[r.Family], FAMILY_MARKER[r.Family]
            ax.plot([r[lo] * k, r[hi] * k], [i, i], color=c, lw=1.6, solid_capstyle="round")
            ax.plot(r[v] * k, i, mk, color=c, ms=6.5, mec="white", mew=0.8)
            ax.text(r[hi] * k, i, "  " + fmt.format(r[v] * k), va="center", ha="left", fontsize=7.6, color=MUTED)
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        span = (d[hi].max() - d[lo].min()) * k
        ax.set_xlim(d[lo].min() * k - 0.04 * span, d[hi].max() * k + 0.22 * span)
    axes[0].set_yticks(range(len(d))); axes[0].set_yticklabels([DISPLAY[m] for m in d.Model], color=INK)
    handles = [plt.Line2D([], [], marker=FAMILY_MARKER[f], color=FAMILY_COLOUR[f], ls="-", lw=1.6, ms=6.5,
                          mec="white", mew=0.8, label=f) for f in ["Hybrid", "CNN", "Transformer"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.56, 1.03))
    for ax, letter in zip(axes, "ab"):
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3_model_comparison")


# ----------------------------------------------------------------------------- Fig. 5 leakage
def fig_leakage():
    order = [REFERENCE, "MobileNetV2", "ViT-B/16"]
    protos = ["Patient-level 5-fold CV (leakage-free)", "A: naive image-level split", "B: augment-before-split"]
    names = ["Leakage-free (patient-level CV)", "Protocol A: image-level split", "Protocol B: augment before split"]
    ramp = ["#9ec5f4", "#2a78d6", "#0d366b"]            # one hue, light to dark = more contamination
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); w = 0.25
    for j, (p, n, c) in enumerate(zip(protos, names, ramp)):
        vals = [leak[(leak.Model == m) & (leak.Protocol == p)].Accuracy.iloc[0] * 100 for m in order]
        xs = np.arange(len(order)) + (j - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=c, label=n, edgecolor="white", linewidth=0.8)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, color=INK)
    for i, m in enumerate(order):
        infl = leak[(leak.Model == m) & (leak.Protocol == protos[2])]["Inflation vs clean (pp)"].iloc[0]
        ax.text(i, 108.5, f"+{infl:.1f} points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fam = main.set_index("Model").Family
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{DISPLAY[m]}\n({fam[m]})" for m in order], color=INK)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.6, handlelength=1.2, columnspacing=1.2)
    save(fig, "fig5_leakage")


# ----------------------------------------------------------------------------- Fig. 4 confusion matrix
def fig_confusion():
    cm = confusion_matrix(labels, OOF[REFERENCE].argmax(1))
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for ax, mat, fmt, letter in zip(axes, [cm, pct], ["{:d}", "{:.0f}"], "ab"):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        for i in range(6):
            for j in range(6):
                v = mat[i, j]
                ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center", fontsize=7.6,
                        color="white" if v > 0.55 * mat.max() else INK)
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=7.6, color=INK)
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASSES if letter == "a" else [], fontsize=7.6, color=INK)
        ax.set_xlabel("Predicted class")
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    axes[0].set_ylabel("True class")
    fig.tight_layout()
    save(fig, "fig4_confusion_matrix")
    return cm


# ----------------------------------------------------------------------------- Fig. 6 calibration
def fig_calibration():
    P = OOF[REFERENCE]; Pc, Ts = cross_fitted(P)
    e0, pts0 = ece(P, labels); e1, pts1 = ece(Pc, labels)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--", label="Perfect calibration")
    ax.plot(pts0[:, 0], pts0[:, 1], "o-", color="#eb6834", lw=1.6, ms=5, mec="white", mew=0.7,
            label=f"Before scaling (ECE {e0:.3f})")
    ax.plot(pts1[:, 0], pts1[:, 1], "s-", color="#2a78d6", lw=1.6, ms=5, mec="white", mew=0.7,
            label=f"After scaling (ECE {e1:.3f})")
    ax.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(loc="upper left", fontsize=7.4)
    ax = axes[1]
    d = main.sort_values("ECE").reset_index(drop=True)
    for i, r in d.iterrows():
        c, mk = FAMILY_COLOUR[r.Family], FAMILY_MARKER[r.Family]
        ax.plot([r["ECE after T"], r["ECE"]], [i, i], color="#c3c2b7", lw=1.4, zorder=1)
        ax.plot(r["ECE"], i, mk, color=c, ms=6.5, mec="white", mew=0.8, zorder=2)
        ax.plot(r["ECE after T"], i, mk, mfc="white", mec=c, ms=6.5, mew=1.4, zorder=2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([DISPLAY[m] for m in d.Model], color=INK, fontsize=7.8)
    ax.set_xlabel("Expected calibration error"); ax.set_xlim(0, max(d.ECE) * 1.12)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    hs = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=6.5, label="Before scaling"),
          plt.Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, mew=1.4, ms=6.5, label="After scaling")]
    hs += [plt.Line2D([], [], marker=FAMILY_MARKER[f], ls="", color=FAMILY_COLOUR[f], ms=6.5, label=f)
           for f in ["Hybrid", "CNN", "Transformer"]]
    ax.legend(handles=hs, loc="lower right", fontsize=7.2, ncol=1)
    for a, letter in zip(axes, "ab"):
        a.text(0.0, 1.03, letter, transform=a.transAxes, fontweight="bold", fontsize=10, va="bottom")
    fig.tight_layout()
    save(fig, "fig6_calibration")
    return e0, e1, Ts


# ----------------------------------------------------------------------------- Fig. 7 decision curve
def fig_dca():
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(dca.threshold, dca.model, color="#2a78d6", lw=2, label="Reference hybrid (calibrated)")
    ax.plot(dca.threshold, dca.treat_all, color="#eb6834", lw=1.6, ls="--", label="Refer all")
    ax.plot(dca.threshold, dca.treat_none, color=INK, lw=1.2, label="Refer none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0, 0.6), ylim=(-0.06, 0.4))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(loc="upper right", fontsize=7.6)
    save(fig, "fig7_decision_curve")


# ----------------------------------------------------------------------------- Fig. 8 efficiency
def fig_efficiency():
    d = eff.merge(main[["Model", "OOF acc"]], on="Model")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for _, r in d.iterrows():
        c, mk = FAMILY_COLOUR[r.Family], FAMILY_MARKER[r.Family]
        ax.scatter(r["CPU latency mean (ms)"], r["OOF acc"] * 100, s=26 + r["TFLite FP32 size (MB)"] * 0.32,
                   marker=mk, color=c, edgecolor="white", linewidth=0.9, alpha=0.95, zorder=2)
    offsets = {"MoViT-SAM": (-30, -17), "Hybrid+Adam": (-70, -3), "Hybrid+H-SAM": (10, -2), "ViT-B/16": (-48, -3),
               "DINO ViT-B/16": (-76, -3), "MobileNetV2": (8, -3), "EfficientNetV2": (8, -3),
               "DenseNet121": (-30, 9), "ResNet50": (9, -3)}
    for _, r in d.iterrows():
        ax.annotate(DISPLAY[r.Model], (r["CPU latency mean (ms)"], r["OOF acc"] * 100),
                    xytext=offsets[r.Model], textcoords="offset points", fontsize=7.6, color=INK)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100, 200, 500]); ax.set_xticklabels([5, 10, 20, 50, 100, 200, 500])
    ax.set_xlim(4.5, 900); ax.set_ylim(62, 80.5); ax.set_yticks(range(62, 81, 2))
    ax.set(xlabel="Mean CPU latency per image (ms, log scale)", ylabel="Pooled out-of-fold accuracy (%)")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    hs = [plt.Line2D([], [], marker=FAMILY_MARKER[f], ls="", color=FAMILY_COLOUR[f], ms=7, label=f)
          for f in ["Hybrid", "CNN", "Transformer"]]
    ax.legend(handles=hs, loc="lower left", fontsize=7.6, title="Marker area: model size", title_fontsize=7.4)
    save(fig, "fig8_efficiency")


# ----------------------------------------------------------------------------- screening AUC interval
def screening_auc_ci():
    """Patient-level bootstrap interval of the binary Mpox screening AUC of the calibrated reference model."""
    score = cross_fitted(OOF[REFERENCE])[0][:, 0]
    yb = (labels == 0).astype(int)
    stored = json.loads((CV / "clinical.json").read_text())["binary_auc"]
    assert abs(roc_auc_score(yb, score) - stored) < 1e-9, "screening AUC mismatch"
    vals = [roc_auc_score(yb[i], score[i]) for i in RESAMPLES if 0 < yb[i].sum() < len(i)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


# ----------------------------------------------------------------------------- summary numbers
def summary(cm, e0, e1, Ts):
    ext = json.loads((FM / "external_validation_results.json").read_text())
    clin = json.loads((CV / "clinical.json").read_text())
    S = {
        "main": main.to_dict("records"),
        "family": main.groupby("Family")[["OOF acc", "Macro-AUC", "Macro-F1"]].agg(["mean", "min", "max"]).round(4)
                      .apply(lambda r: {f"{a}_{b}": v for (a, b), v in r.items()}, axis=1).to_dict(),
        "friedman": {"chi2": float(friedman["Friedman chi2"].iloc[0]), "p": float(friedman["Friedman p"].iloc[0]),
                     "k": int(friedman["k models"].iloc[0]), "n": int(friedman["n folds"].iloc[0]),
                     "mean_rank": dict(zip(friedman.Model, friedman["Mean rank (1 = best)"]))},
        "pairs": pairs.to_dict("records"),
        "n_pairs": int(len(pairs)),
        "n_sig_patient": int((pairs["Patient: p (Holm)"] < 0.05).sum()),
        "n_sig_image": int((pairs["Image: p (Holm)"] < 0.05).sum()),
        "leakage": leak.to_dict("records"),
        "late_fusion": LATE,
        "screening_auc_ci_patient": screening_auc_ci(),
        "ci_method": "bootstrap, 2000 resamples of patients",
        "perclass": perclass.to_dict("records"),
        "reference_perclass": pd.read_csv(CV / "table5_per_class.csv").to_dict("records"),
        "fold_table": folds.to_dict("records"),
        "calibration_reference": {"ece_before": e0, "ece_after": e1, "T_per_fold": Ts, "T_mean": float(np.mean(Ts)),
                                  "stored": json.loads((CV / "calibration.json").read_text())},
        "pooled_reference": json.loads((CV / "pooled_oof_summary.json").read_text()),
        "clinical": clin,
        "ppv_npv": pd.read_csv(CV / "table7_ppv_npv.csv").to_dict("records"),
        "operating_points": pd.read_csv(CV / "screening_operating_points.csv").to_dict("records"),
        "confusion_reference": cm.tolist(),
        "mpox_as_healthy": int(cm[0, 5]), "n_mpox": int(cm[0].sum()),
        "efficiency": eff.merge(main[["Model", "OOF acc", "Macro-AUC"]], on="Model").to_dict("records"),
        "tflite": pd.read_csv(FM / "table8_tflite.csv").to_dict("records"),
        "final_model": pd.read_csv(FM / "table6_final_model.csv").to_dict("records")[0],
        "external": ext,
        "split": pd.read_csv(CV / "table1_split.csv").to_dict("records"),
        "ita": pd.read_csv(CV / "audit_ita_by_class.csv").to_dict("records"),
        "near_duplicates": int(len(pd.read_csv(CV / "audit_near_duplicates.csv"))),
        "hardware": json.loads((CV / "hardware.json").read_text()),
        "n_images": int(len(labels)), "n_patients": int(len(set(groups))),
        "fold_sizes": np.bincount(fold_id)[1:].tolist(),
    }
    stored = S["calibration_reference"]["stored"]
    assert abs(stored["ece_before"] - e0) < 1e-6 and abs(stored["ece_after"] - e1) < 1e-6, "calibration mismatch"
    def clean(o):                  # NaN is not valid JSON
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        return o
    (ROOT / "analysis" / "summary_numbers.json").write_text(json.dumps(clean(S), indent=1, allow_nan=False))
    print("wrote analysis/summary_numbers.json")


if __name__ == "__main__":
    fig_study_design()
    fig_model_comparison()
    cm = fig_confusion()
    fig_leakage()
    e0, e1, Ts = fig_calibration()
    fig_dca()
    fig_efficiency()
    summary(cm, e0, e1, Ts)
