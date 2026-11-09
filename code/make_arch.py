#!/usr/bin/env python3
"""Generate the four methodology diagrams specified in Dengue_Research_Plan.

This deliberately uses Matplotlib rather than Graphviz, so the diagrams remain
reproducible on machines where the Graphviz ``dot`` executable is unavailable.
It also emits equation images retained for the manuscript asset set.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ARCH = "arch"
os.makedirs(ARCH, exist_ok=True)

COLORS = {
    "data": "#D6EAF8",
    "audit": "#FADBD8",
    "novel": "#D5F5E3",
    "model": "#FDEBD0",
    "honesty": "#EBDEF0",
    "output": "#FCF3CF",
    "edge": "#566573",
}


def configure(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, width, height, label, fill, *, fontsize=9, edge="#566573"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (x, y, width, height)


def arrow(ax, start, end, label=None):
    """Connect two normalized coordinates with an unobtrusive directional arrow."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=COLORS["edge"], lw=1.25, shrinkA=2, shrinkB=2),
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#34495E",
        )


def group_label(ax, x, y, width, title, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, 0.055,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0,
            facecolor=color,
            alpha=0.78,
        )
    )
    ax.text(x + width / 2, y + 0.0275, title, ha="center", va="center", fontsize=9, fontweight="bold")


def save(fig, name):
    fig.savefig(f"{ARCH}/{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def overall_architecture():
    fig, ax = plt.subplots(figsize=(13.4, 5.6))
    configure(ax)
    group_label(ax, 0.025, 0.92, 0.205, "1. Data preparation", COLORS["data"])
    group_label(ax, 0.275, 0.92, 0.205, "2. Feature engineering (novel)", COLORS["novel"])
    group_label(ax, 0.525, 0.92, 0.195, "3. Model comparison", COLORS["model"])
    group_label(ax, 0.765, 0.92, 0.21, "4. Honest evaluation (novel)", COLORS["honesty"])

    raw = box(ax, 0.045, 0.68, 0.16, 0.13, "Raw CBC dataset\n1,523 records", COLORS["data"])
    audit = box(ax, 0.045, 0.45, 0.16, 0.13, "Integrity audit\nduplicates, PCT, WBC,\ndifferential checks", COLORS["audit"])
    clean = box(ax, 0.045, 0.20, 0.16, 0.13, "Leakage-safe cleaning\n1,505 records", COLORS["data"])
    arrow(ax, (raw[0] + raw[2] / 2, raw[1]), (audit[0] + audit[2] / 2, audit[1] + audit[3]))
    arrow(ax, (audit[0] + audit[2] / 2, audit[1]), (clean[0] + clean[2] / 2, clean[1] + clean[3]))

    clr = box(ax, 0.295, 0.57, 0.16, 0.13, "CLR transform\nof WBC differential", COLORS["novel"])
    scale = box(ax, 0.295, 0.30, 0.16, 0.13, "Training-fitted\nstandardization", COLORS["novel"])
    arrow(ax, (clean[0] + clean[2], clean[1] + clean[3] / 2), (clr[0], clr[1] + clr[3] / 2))
    arrow(ax, (clr[0] + clr[2] / 2, clr[1]), (scale[0] + scale[2] / 2, scale[1] + scale[3]))

    base = box(ax, 0.545, 0.65, 0.155, 0.12, "Legacy baselines\nLR, RF, XGBoost,\nSVM, MLP", COLORS["model"])
    proposed = box(ax, 0.545, 0.36, 0.155, 0.12, "Proposed CGB-CLR\nGradient boosting +\nisotonic calibration", COLORS["model"], edge="#117A65")
    arrow(ax, (scale[0] + scale[2], scale[1] + scale[3] / 2), (base[0], base[1] + base[3] / 2))
    arrow(ax, (scale[0] + scale[2], scale[1] + scale[3] / 2), (proposed[0], proposed[1] + proposed[3] / 2))

    metrics = box(ax, 0.785, 0.68, 0.165, 0.115, "Discrimination + CIs\nAUC, accuracy, F1", COLORS["honesty"])
    calibration = box(ax, 0.785, 0.50, 0.165, 0.115, "Calibration\nreliability + Brier", COLORS["honesty"])
    utility = box(ax, 0.785, 0.32, 0.165, 0.115, "Clinical utility\nDCA + rule-out", COLORS["honesty"])
    abstain = box(ax, 0.785, 0.14, 0.165, 0.115, "Selective prediction\nconfident output or\ndefer to PCR", COLORS["honesty"])
    for destination in (metrics, calibration, utility, abstain):
        arrow(ax, (proposed[0] + proposed[2], proposed[1] + proposed[3] / 2), (destination[0], destination[1] + destination[3] / 2))
    ax.text(
        0.5, 0.035,
        "Green and purple stages are the framework additions: CLR representation, calibrated probabilities, decision utility, and abstention.",
        ha="center", va="center", fontsize=9, color="#2C3E50",
    )
    save(fig, "arch_overall")


def audit_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    configure(ax)
    raw = box(ax, 0.36, 0.85, 0.28, 0.09, "Raw CBC CSV\n1,523 records × 19 columns", COLORS["data"], fontsize=10)
    defects = [
        (0.05, "Duplicate / conflicting\nfeature-identical records", "12 exact duplicates;\n3 conflict groups"),
        (0.275, "PCT plausibility", "PCT maximum = 234%"),
        (0.50, "Differential coherence", "four-component sum\nrange 71–114"),
        (0.725, "Label-aligned pattern", "15 WBC >8,000 records;\nall are negative"),
    ]
    checks = []
    for x, title, detail in defects:
        checks.append(box(ax, x, 0.54, 0.20, 0.14, f"{title}\n\n{detail}", COLORS["audit"], fontsize=8.4))
    for check in checks:
        arrow(ax, (raw[0] + raw[2] / 2, raw[1]), (check[0] + check[2] / 2, check[1] + check[3]))
    action = box(ax, 0.31, 0.26, 0.38, 0.12, "Pre-specified remediation\nremove duplicate/conflicting records • cap PCT at 1% • retain limitations", COLORS["model"], fontsize=9.2)
    for check in checks:
        arrow(ax, (check[0] + check[2] / 2, check[1]), (action[0] + action[2] / 2, action[1] + action[3]))
    clean = box(ax, 0.36, 0.055, 0.28, 0.09, "Analysis cohort\n1,505 records", COLORS["novel"], fontsize=10)
    arrow(ax, (action[0] + action[2] / 2, action[1]), (clean[0] + clean[2] / 2, clean[1] + clean[3]))
    ax.text(0.5, 0.46, "Audit findings are reported as limitations; they are not silently optimized away.", ha="center", fontsize=9, color="#7B241C")
    save(fig, "arch_audit")


def clr_architecture():
    fig, ax = plt.subplots(figsize=(12.2, 4.15))
    configure(ax)
    recorded = box(ax, 0.035, 0.39, 0.18, 0.22, "Recorded differential\nNeutrophils %\nLymphocytes %\nMonocytes %\nEosinophils %", COLORS["data"], fontsize=9)
    zero = box(ax, 0.275, 0.43, 0.15, 0.14, "Additive\nreplacement\n$c_i + 0.5$", COLORS["novel"], fontsize=9)
    closure = box(ax, 0.485, 0.43, 0.15, 0.14, "Closure\n$x_i=(c_i+0.5)$/Σ$c_j$", COLORS["novel"], fontsize=9)
    clr = box(ax, 0.695, 0.43, 0.15, 0.14, "CLR coordinates\n$z_i=\\log(x_i)$\n− mean$[\\log(x)]$", COLORS["novel"], fontsize=8.3)
    output = box(ax, 0.89, 0.43, 0.08, 0.14, "Four\nmodel-ready\nrelative\nfeatures", COLORS["output"], fontsize=8)
    for left, right in ((recorded, zero), (zero, closure), (closure, clr), (clr, output)):
        arrow(ax, (left[0] + left[2], left[1] + left[3] / 2), (right[0], right[1] + right[3] / 2))
    ax.text(0.5, 0.18, "The closure step represents the relative balance of the four recorded components. It does not claim that the raw values form a complete leukocyte differential.", ha="center", va="center", fontsize=9, wrap=True, color="#2C3E50")
    save(fig, "arch_clr")


def conformal_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 6.1))
    configure(ax)
    training = box(ax, 0.055, 0.78, 0.20, 0.12, "Training split", COLORS["data"])
    fitted = box(ax, 0.35, 0.78, 0.20, 0.12, "Fit gradient-boosting\nmodel", COLORS["model"])
    calibration = box(ax, 0.35, 0.54, 0.20, 0.12, "Calibration split\nclass-wise probability\nnonconformity scores", COLORS["data"])
    score = box(ax, 0.72, 0.54, 0.20, 0.12, "Estimate class-wise\nquantiles at target\ncoverage 1 − α", COLORS["honesty"])
    arrow(ax, (training[0] + training[2], training[1] + training[3] / 2), (fitted[0], fitted[1] + fitted[3] / 2))
    arrow(ax, (fitted[0] + fitted[2] / 2, fitted[1]), (calibration[0] + calibration[2] / 2, calibration[1] + calibration[3]), "probabilities")
    arrow(ax, (calibration[0] + calibration[2], calibration[1] + calibration[3] / 2), (score[0], score[1] + score[3] / 2), "scores")

    patient = box(ax, 0.055, 0.27, 0.20, 0.12, "New patient CBC\nCLR + scaled features", COLORS["data"])
    predset = box(ax, 0.40, 0.27, 0.20, 0.12, "Prediction set\nlabels admissible\nat 1 − α", COLORS["honesty"])
    decision = box(ax, 0.72, 0.27, 0.20, 0.12, "Singleton\nprediction set?", COLORS["honesty"])
    arrow(ax, (patient[0] + patient[2], patient[1] + patient[3] / 2), (predset[0], predset[1] + predset[3] / 2))
    arrow(ax, (score[0] + score[2] / 2, score[1]), (predset[0] + predset[2] / 2, predset[1] + predset[3]), "quantiles")
    arrow(ax, (predset[0] + predset[2], predset[1] + predset[3] / 2), (decision[0], decision[1] + decision[3] / 2))
    decided = box(ax, 0.55, 0.045, 0.20, 0.13, "Yes: output one\nconfident class label", COLORS["novel"])
    defer = box(ax, 0.79, 0.045, 0.18, 0.13, "No: abstain\nand defer to\nconfirmatory test", COLORS["audit"])
    arrow(ax, (decision[0] + decision[2] / 2 - 0.04, decision[1]), (decided[0] + decided[2] / 2, decided[1] + decided[3]), "yes")
    arrow(ax, (decision[0] + decision[2] / 2 + 0.04, decision[1]), (defer[0] + defer[2] / 2, defer[1] + defer[3]), "no")
    ax.text(0.31, 0.125, "At 90% target coverage: 30.5% singleton decisions\nand 69.5% deferred cases (internal test set).", ha="center", fontsize=8.7, color="#2C3E50")
    save(fig, "arch_conformal")


def equation(tex, name, fontsize=20):
    fig = plt.figure(figsize=(6.2, 0.9))
    fig.text(0.02, 0.5, tex, fontsize=fontsize, va="center")
    plt.axis("off")
    fig.savefig(f"{ARCH}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    overall_architecture()
    audit_architecture()
    clr_architecture()
    conformal_architecture()
    equation(r"$\mathrm{CLR}(x)_i=\ln x_i-\dfrac{1}{D}\sum_{j=1}^{D}\ln x_j,\qquad x_i=\dfrac{c_i+0.5}{\sum_j(c_j+0.5)}$", "eq_clr")
    equation(r"$\mathrm{NB}(p_t)=\dfrac{TP}{N}-\dfrac{FP}{N}\cdot\dfrac{p_t}{1-p_t}$", "eq_netbenefit")
    equation(r"$\hat q_{k}=\mathrm{Quantile}_{1-\alpha}(\{s_i:y_i=k\}),\quad s_i=1-\hat p(y_i\mid x_i)$", "eq_conformal")
    equation(r"$\mathrm{Brier}=\dfrac{1}{N}\sum_{i=1}^{N}(\hat p_i-y_i)^2$", "eq_brier")
    equation(r"$\mathrm{AUC}=\int_0^1\mathrm{TPR}(\mathrm{FPR})\,d(\mathrm{FPR})$", "eq_auc")
    equation(r"$F_1=\dfrac{2\,\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}},\quad\mathrm{Recall}=\dfrac{TP}{TP+FN}$", "eq_f1")
    print("Architecture assets:")
    for filename in sorted(os.listdir(ARCH)):
        print(" ", filename)


if __name__ == "__main__":
    main()
