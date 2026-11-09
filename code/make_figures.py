#!/usr/bin/env python3
"""Render every evaluation figure as a print-quality PDF and emit results.json
with the exact numbers used in the paper. Computation is deterministic (seed 42)
so every printed value matches the artifacts."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json, os
from scipy.stats import pointbiserialr
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, recall_score,
                             precision_score, accuracy_score, brier_score_loss,
                             confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- house style
# Validated categorical palette shared with the companion paper (light-safe).
C = ["#3b6fd4", "#d1793b", "#3f9e7c", "#9d5bb5"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.4  # IEEE single-column width, inches


def style(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


RNG = 42; np.random.seed(RNG)
FIG = "figs"; os.makedirs(FIG, exist_ok=True)
CSV = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
DIFF = ['Neutrophils(%)', 'Lymphocytes(%)', 'Monocytes(%)', 'Eosinophils(%)']
R = {}

raw = pd.read_csv(CSV)
# ---- audit numbers ----
s = raw[DIFF].sum(axis=1)
R['n_raw'] = len(raw)
R['exact_dupes'] = int(raw.duplicated().sum())
R['conflict_groups'] = int((raw.groupby(list(raw.drop(columns='Result').columns))
                            ['Result'].nunique() > 1).sum())
R['diff_eq100'] = int((s == 100).sum()); R['diff_min'] = int(s.min()); R['diff_max'] = int(s.max())
R['pct_max'] = float(raw['PCT(%)'].max())
R['rbc_vals'] = sorted(raw['RBC'].unique().tolist())
hi = raw['Total WBC count(/cumm)'] > 8000
R['wbc_hi_n'] = int(hi.sum()); R['wbc_hi_posrate'] = float((raw.loc[hi, 'Result'] == 'positive').mean())
R['n_pos'] = int((raw['Result'] == 'positive').sum()); R['n_neg'] = int((raw['Result'] == 'negative').sum())

# ---- clean ----
df = raw.copy(); df['Gender'] = (df['Gender'] == 'Male').astype(int)
df['y'] = (df['Result'] == 'positive').astype(int)
feat = df.drop(columns=['Result', 'y']); key = feat.astype(str).agg('|'.join, axis=1)
conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
df = df[~conflict].drop_duplicates(subset=list(feat.columns) + ['y'], keep='first')
df['PCT(%)'] = df['PCT(%)'].clip(upper=1.0)
df = df.reset_index(drop=True)
R['n_clean'] = len(df); R['n_removed'] = R['n_raw'] - len(df)

# ---- CLR transform ----
comp = df[DIFF].astype(float).values + 0.5; comp = comp / comp.sum(1, keepdims=True)
logc = np.log(comp); clr = logc - logc.mean(1, keepdims=True)
Xc = df.drop(columns=DIFF + ['Result', 'y']).copy()
for i, c in enumerate(DIFF):
    Xc[f"clr_{c.split('(')[0].strip()}"] = clr[:, i]
y = df['y']; feats = Xc.columns.tolist()

Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.25, stratify=y, random_state=RNG)
sc = StandardScaler().fit(Xtr)
Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats, index=Xtr.index)
Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats, index=Xte.index)
R['base_rate'] = float(yte.mean())


def ci(y_, p_, m, n=2000):
    idx = np.arange(len(y_)); v = []
    for _ in range(n):
        b = np.random.choice(idx, len(idx), replace=True)
        if len(np.unique(y_[b])) < 2:
            continue
        v.append(m(y_[b], p_[b]))
    return np.percentile(v, [2.5, 97.5])


# transparent model's features chosen on the training partition only
cors = {c: abs(pointbiserialr(ytr, Xtr[c])[0]) for c in feats}
top3 = [k for k, _ in sorted(cors.items(), key=lambda z: -z[1])[:3]]
R['top3'] = top3

# ------------------------------------- full common-protocol baseline table
BAL = dict(class_weight='balanced')
models = {
    "Majority": (None, None),
    "Parsimonious LR": (LogisticRegression(max_iter=1000, **BAL), top3),
    "Full LR": (LogisticRegression(max_iter=1000, **BAL), feats),
    "Random Forest": (RandomForestClassifier(random_state=RNG, **BAL), feats),
    "XGBoost": (XGBClassifier(random_state=RNG, eval_metric='logloss',
                              use_label_encoder=False) if HAVE_XGB else
                GradientBoostingClassifier(random_state=RNG), feats),
    "RBF SVM": (SVC(probability=True, random_state=RNG, **BAL), feats),
    "MLP": (MLPClassifier(random_state=RNG, max_iter=800), feats),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=RNG), feats),
}
rows = []; probs = {}
for nm, (mdl, cols) in models.items():
    if nm == "Majority":
        p = np.full(len(yte), ytr.mean())
        pred = np.ones(len(yte), int)
        auc, lo, hi = 0.5, 0.5, 0.5
    else:
        mdl.fit(Xtr_s[cols], ytr); p = mdl.predict_proba(Xte_s[cols])[:, 1]
        probs[nm] = (p, cols, mdl); pred = (p >= .5).astype(int)
        auc = roc_auc_score(yte, p); lo, hi = ci(yte.values, p, roc_auc_score)
    rows.append(dict(model=nm, acc=accuracy_score(yte, pred),
                     prec=precision_score(yte, pred, zero_division=0),
                     rec=recall_score(yte, pred), f1=f1_score(yte, pred),
                     auc=auc, lo=lo, hi=hi,
                     brier=brier_score_loss(yte, p)))

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
aucs = cross_val_score(GradientBoostingClassifier(random_state=RNG),
                       StandardScaler().fit_transform(Xc), y, cv=rcv, scoring='roc_auc')
R['cv_auc_mean'] = float(aucs.mean()); R['cv_auc_std'] = float(aucs.std())
R['cv_auc_min'] = float(aucs.min()); R['cv_auc_max'] = float(aucs.max())

# calibrated GB (the proposed CGB-CLR)
p_raw = probs["Gradient Boosting"][0]
cal = CalibratedClassifierCV(GradientBoostingClassifier(random_state=RNG),
                             method='isotonic', cv=5).fit(Xtr_s, ytr)
p_cal = cal.predict_proba(Xte_s)[:, 1]
R['brier_raw'] = float(brier_score_loss(yte, p_raw))
R['brier_cal'] = float(brier_score_loss(yte, p_cal))
pred_cal = (p_cal >= .5).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred_cal).ravel()
R['cm'] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
rows.append(dict(model="Proposed CGB-CLR", acc=accuracy_score(yte, pred_cal),
                 prec=precision_score(yte, pred_cal), rec=recall_score(yte, pred_cal),
                 f1=f1_score(yte, pred_cal), auc=roc_auc_score(yte, p_cal),
                 lo=ci(yte.values, p_cal, roc_auc_score)[0],
                 hi=ci(yte.values, p_cal, roc_auc_score)[1],
                 brier=R['brier_cal']))
R['table'] = rows

# =========================================================== FIGURES ========
LAB = {"Majority": "Majority", "Parsimonious LR": "3-feat LR", "Full LR": "Full LR",
       "Random Forest": "RF", "XGBoost": "XGBoost", "RBF SVM": "RBF-SVM",
       "MLP": "MLP", "Gradient Boosting": "GB (raw)", "Proposed CGB-CLR": "CGB-CLR"}


# ---- Fig 2 : cohort composition (class balance + core-feature correlation) --
def fig_cohort():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.62),
                                 gridspec_kw={"height_ratios": [0.26, 1.0],
                                              "hspace": 1.05})
    pos, neg = R['n_pos'], R['n_neg']; tot = pos + neg
    a1.barh([0], [pos], color=C[1], height=.6, zorder=3, linewidth=0)
    a1.barh([0], [neg], left=[pos], color=C[0], height=.6, zorder=3, linewidth=0)
    a1.text(pos / 2, 0, f"positive  {pos}  ({pos/tot:.1%})", ha='center',
            va='center', color='white', fontsize=6.6, fontweight='bold')
    a1.text(pos + neg / 2, 0, f"negative  {neg}", ha='center', va='center',
            color='white', fontsize=6.6, fontweight='bold')
    a1.set_xlim(0, tot); a1.set_yticks([]); a1.set_ylim(-0.5, 0.5)
    a1.set_xlabel("Raw-cohort records", labelpad=1.5)
    a1.set_xticks([0, 500, 1000, 1500])
    style(a1, grid="x")

    core = ['Age', 'Hemoglobin(g/dl)', 'HCT(%)', 'RDW-CV(%)',
            'Total WBC count(/cumm)', 'Platelet count(/cumm)', 'PDW(fl)', 'MPV(fl)',
            'clr_Neutrophils', 'clr_Lymphocytes', 'clr_Monocytes', 'clr_Eosinophils']
    short = ['Age', 'Hb', 'HCT', 'RDW', 'WBC', 'PLT', 'PDW', 'MPV',
             'Neu*', 'Lym*', 'Mon*', 'Eos*']
    M = pd.concat([Xc, y.rename('Result')], axis=1)
    order = [c for c in core if c in M.columns] + ['Result']
    lab = short + ['Dengue']
    corr = M[order].corr().values
    im = a2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    a2.set_xticks(range(len(lab))); a2.set_xticklabels(lab, rotation=90, fontsize=5.6)
    a2.set_yticks(range(len(lab))); a2.set_yticklabels(lab, fontsize=5.6)
    a2.tick_params(length=0)
    for sp in a2.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=6, length=0); cb.outline.set_visible(False)
    a2.set_title("Pearson correlation (CLR leukocytes marked *)", fontsize=6.8, pad=3)
    save(fig, "fig_cohort.pdf")


# ---- Fig 3 : model discrimination, forest plot with bootstrap CI -----------
def fig_auc():
    # Read the authoritative common-protocol baselines (same source as the
    # results table) so the figure and the table can never disagree.
    bc = pd.read_csv("baseline_comparison.csv")
    RENAME = {"Majority class": "Majority",
              "Parsimonious LR (3 features)": "Parsimonious LR",
              "Full LR": "Full LR", "Random Forest": "Random Forest",
              "XGBoost": "XGBoost", "RBF SVM": "RBF SVM", "MLP": "MLP",
              "Proposed: calibrated GB + CLR": "Proposed CGB-CLR"}
    bc["model"] = bc["model"].map(RENAME)
    rr = {r["model"]: dict(model=r["model"], auc=r["auc"], lo=r["auc_lo"],
                           hi=r["auc_hi"]) for _, r in bc.iterrows()}
    order = ["Majority", "Parsimonious LR", "Full LR", "MLP", "RBF SVM",
             "XGBoost", "Random Forest", "Proposed CGB-CLR"]
    data = [rr[k] for k in order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    for yi, r in zip(y, data):
        prop = r['model'] == "Proposed CGB-CLR"
        col = C[1] if prop else MUTED
        ax.plot([r['lo'], r['hi']], [yi, yi], color=col, lw=1.6 if prop else 1.1,
                solid_capstyle="round", zorder=3, alpha=0.9)
        ax.scatter([r['auc']], [yi], s=30 if prop else 18, color=col, zorder=4,
                   linewidth=0)
        ax.text(r['hi'] + 0.006, yi, f"{r['auc']:.3f}", va='center', fontsize=6.2,
                color=col, fontweight='bold' if prop else 'normal')
    ax.axvline(0.5, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.annotate("chance", xy=(0.5, y.max() + 0.75), ha='center', fontsize=6,
                color=INK, annotation_clip=False)
    ax.set_yticks(y); ax.set_yticklabels([LAB[k] for k in order], fontsize=6.6)
    ax.set_xlabel("Held-out AUC with 95% bootstrap CI", labelpad=1.5)
    ax.set_xlim(0.45, 0.80); ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_ylim(y.min() - 0.6, y.max() + 1.2)
    style(ax, grid="x")
    save(fig, "fig_auc.pdf")


# ---- Fig 4 : ROC + reliability ---------------------------------------------
def fig_roc_cal():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.55),
                                 gridspec_kw={"wspace": 0.42})
    roc_items = [('3-feat LR', probs['Parsimonious LR'][0], C[3], (0, (3, 2))),
                 ('Full LR', probs['Full LR'][0], C[2], (0, (1, 1.5))),
                 ('CGB-CLR', p_cal, C[1], '-')]
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=2)
    for nm, p, col, ls in roc_items:
        fpr, tpr, _ = roc_curve(yte, p)
        a1.plot(fpr, tpr, ls=ls, color=col, lw=1.7 if nm == 'CGB-CLR' else 1.2,
                zorder=3, label=f"{nm} {roc_auc_score(yte, p):.3f}")
    a1.set_xlabel("False-positive rate", labelpad=1.5)
    a1.set_ylabel("True-positive rate", labelpad=1.5)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.set_xticks([0, 0.5, 1.0]); a1.set_yticks([0, 0.5, 1.0])
    a1.legend(frameon=False, loc="lower right", handlelength=1.2, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4, fontsize=6.0)
    style(a1, grid="both")

    a2.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=2)
    for p_, lbl, col in [(p_raw, 'uncalibrated', C[0]),
                         (p_cal, 'isotonic', C[1])]:
        ft, mp = calibration_curve(yte, p_, n_bins=8, strategy='quantile')
        a2.plot(mp, ft, '-o', color=col, lw=1.4, ms=3.0, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5, label=lbl)
    a2.set_xlabel("Predicted probability", labelpad=1.5)
    a2.set_ylabel("Observed frequency", labelpad=1.5)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    a2.set_xticks([0, 0.5, 1.0]); a2.set_yticks([0, 0.5, 1.0])
    a2.legend(frameon=False, loc="upper left", handlelength=1.0, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4, fontsize=6.2)
    style(a2, grid="both")
    save(fig, "fig_roc_cal.pdf")


# ---- Fig 5 : decision curve + rule-out -------------------------------------
def netben(y_, p_, ts):
    N = len(y_); m = []; a = []
    for pt in ts:
        pred = (p_ >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_ == 1)); fp = np.sum((pred == 1) & (y_ == 0))
        m.append(tp / N - fp / N * (pt / (1 - pt)))
        a.append(np.sum(y_ == 1) / N - np.sum(y_ == 0) / N * (pt / (1 - pt)))
    return np.array(m), np.array(a)


def ruleout(y_, p_, tgt):
    fpr, tpr, thr = roc_curve(y_, p_); spec = 1 - fpr; ok = np.where(tpr >= tgt)[0]
    i = ok[np.argmax(spec[ok])]; t = thr[i]; pred = (p_ >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
    return dict(tgt=tgt, thr=float(t), sens=tp / (tp + fn), spec=tn / (tn + fp),
                npv=tn / (tn + fn) if (tn + fn) else np.nan, saved=(pred == 0).mean())


def fig_dca_ruleout():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.55),
                                 gridspec_kw={"wspace": 0.5})
    ts = np.linspace(0.05, 0.6, 40); nm_, na_ = netben(yte.values, p_cal, ts)
    a1.fill_between(ts, na_, nm_, where=nm_ >= na_, color=C[1], alpha=0.16, zorder=1)
    a1.plot(ts, nm_, lw=1.7, color=C[1], zorder=3, label='CGB-CLR')
    a1.plot(ts, na_, ls=(0, (4, 2)), color=C[0], lw=1.2, label='treat all', zorder=2)
    a1.axhline(0, color=MUTED, lw=0.9, label='treat none', zorder=2)
    a1.set_xlabel("Threshold probability", labelpad=1.5)
    a1.set_ylabel("Net benefit", labelpad=1.5)
    a1.set_ylim(-0.05, nm_.max() * 1.18); a1.set_xticks([0.1, 0.3, 0.5])
    a1.legend(frameon=False, loc="upper right", handlelength=1.1, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4, fontsize=6.0)
    style(a1)

    ro = [ruleout(yte.values, p_cal, t) for t in [0.99, 0.95, 0.90, 0.80]]
    R['ruleout'] = ro
    tg = [r['tgt'] for r in ro]
    a2.plot(tg, [r['spec'] for r in ro], '-o', lw=1.4, ms=3.2, color=C[0],
            markeredgecolor='white', markeredgewidth=0.5, label='specificity')
    a2.plot(tg, [r['saved'] for r in ro], '-D', lw=1.4, ms=3.0, color=C[1],
            markeredgecolor='white', markeredgewidth=0.5, label='tests avoided')
    for x, r in zip(tg, ro):
        a2.text(x, r['saved'] + .022, f"{r['saved']:.0%}", ha='center', fontsize=5.8,
                color=C[1])
    a2.set_xlabel("Required sensitivity", labelpad=1.5)
    a2.set_ylabel("Fraction", labelpad=1.5)
    a2.set_ylim(0, .60); a2.invert_xaxis()
    a2.legend(frameon=False, loc="upper left", handlelength=1.1, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4, fontsize=6.0)
    style(a2)
    save(fig, "fig_dca_ruleout.pdf")


# ---- Fig 6 : conformal decide/defer + permutation importance ---------------
def fig_conf_imp():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.62),
                                 gridspec_kw={"wspace": 0.52, "width_ratios": [1.0, 1.15]})
    Xtr2, Xcal, ytr2, ycal = train_test_split(Xtr_s, ytr, test_size=0.3,
                                               stratify=ytr, random_state=RNG)
    m = GradientBoostingClassifier(random_state=RNG).fit(Xtr2, ytr2)
    pcal = m.predict_proba(Xcal)[:, 1]; ptest = m.predict_proba(Xte_s)[:, 1]
    res = {}
    for alpha in [0.05, 0.10, 0.20]:
        q1 = np.quantile((1 - pcal)[ycal.values == 1], 1 - alpha)
        q0 = np.quantile(pcal[ycal.values == 0], 1 - alpha)
        in1 = (1 - ptest) <= q1; in0 = ptest <= q0
        singleton = in1 ^ in0; abstain = (~(in1 | in0)) | (in1 & in0)
        pred = np.where(in1 & ~in0, 1, 0)
        cov = (((yte.values == 1) & in1) | ((yte.values == 0) & in0)).mean()
        acc = accuracy_score(yte.values[singleton], pred[singleton]) if singleton.any() else np.nan
        res[alpha] = dict(cov=float(cov), singleton=float(singleton.mean()),
                          abstain=float(abstain.mean()), acc=float(acc))
    R['conformal'] = res
    al = sorted(res.keys(), reverse=True); cov = [1 - a for a in al]
    decided = np.array([res[a]['singleton'] for a in al])
    deferred = np.array([res[a]['abstain'] for a in al])
    x = np.arange(len(cov))
    a1.bar(x, decided, color=C[0], width=.62, zorder=3, linewidth=0, label='decide')
    a1.bar(x, deferred, bottom=decided, color=C[1], width=.62, zorder=3,
           linewidth=0, label='defer')
    for i, (d, a) in enumerate(zip(decided, deferred)):
        a1.text(i, d / 2, f"{d:.0%}", ha='center', va='center', color='white',
                fontweight='bold', fontsize=6.2)
        a1.text(i, d + a / 2, f"{a:.0%}", ha='center', va='center', color='white',
                fontweight='bold', fontsize=6.2)
    a1.set_xticks(x); a1.set_xticklabels([f"{c:.0%}" for c in cov])
    a1.set_xlabel("Target coverage", labelpad=1.5)
    a1.set_ylabel("Fraction of patients", labelpad=1.5)
    a1.set_ylim(0, 1.12)
    a1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=0.9, columnspacing=0.9, handletextpad=0.4, fontsize=6.2)
    style(a1, grid="y")

    clf = probs["Gradient Boosting"][2]
    pi = permutation_importance(clf, Xte_s, yte, n_repeats=30, random_state=RNG,
                                scoring='roc_auc')
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    R['imp_top'] = [(k, float(v)) for k, v in imp.head(8).items()]
    R['imp_max'] = float(imp.max())
    plat = [c for c in feats if 'Platelet' in c and 'clr' not in c][0]
    R['platelet_rank'] = int(list(imp.index).index(plat) + 1)
    R['platelet_univar'] = float(cors[plat])
    NICE = {'Total Platelet Count(/cumm)': 'Platelet count', 'clr_Neutrophils': 'Neutrophils*',
            'RDW-CV(%)': 'RDW-CV', 'clr_Lymphocytes': 'Lymphocytes*', 'MPV(fl)': 'MPV',
            'PDW(fl)': 'PDW', 'Hemoglobin(g/dl)': 'Hemoglobin', 'HCT(%)': 'HCT',
            'clr_Monocytes': 'Monocytes*', 'clr_Eosinophils': 'Eosinophils*',
            'Total WBC count(/cumm)': 'WBC', 'PCT(%)': 'PCT', 'RBC': 'RBC', 'Age': 'Age',
            'MCV(fl)': 'MCV', 'MCH(pg)': 'MCH', 'MCHC(g/dl)': 'MCHC', 'RBC ': 'RBC',
            'Gender': 'Gender'}
    top = imp.head(8)[::-1]
    yy = np.arange(len(top))
    cols = [C[1] if name == imp.index[0] else C[0] for name in top.index]
    a2.barh(yy, top.values, 0.66, color=cols, zorder=3, linewidth=0)
    a2.set_yticks(yy)
    a2.set_yticklabels([NICE.get(k, k) for k in top.index], fontsize=6.0)
    a2.set_xlabel("Decrease in held-out AUC", labelpad=1.5)
    a2.set_xlim(0, imp.max() * 1.18)
    style(a2, grid="x")
    save(fig, "fig_conf_imp.pdf")


if __name__ == "__main__":
    fig_cohort()
    fig_auc()
    fig_roc_cal()
    fig_dca_ruleout()
    fig_conf_imp()
    json.dump(R, open("results.json", "w"), indent=2, default=float)
    print("done; figs:", sorted(f for f in os.listdir(FIG) if f.endswith('.pdf')))
    print(json.dumps({k: R[k] for k in ['n_clean', 'base_rate', 'cv_auc_mean',
          'cv_auc_std', 'brier_raw', 'brier_cal', 'platelet_rank', 'imp_max',
          'top3', 'cm']}, indent=2, default=float))
