#!/usr/bin/env python3
"""Revised dengue-paper figures: CV-primary discrimination, calibration, decision
utility, conformal abstention and external validation for the upgraded CGB-CLR
(calibrated LightGBM+XGBoost+CatBoost ensemble on CLR + ratios + interactions)."""
import warnings; warnings.filterwarnings("ignore")
import os, json, zipfile, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.stats import pointbiserialr
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#3b6fd4", "#d1793b", "#3f9e7c", "#9d5bb5"]
INK, MUTED, GRID, GREY = "#1a1a1a", "#5c5c5c", "#d8d8d4", "#9a9a94"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = 3.4
FIG = "figs"; os.makedirs(FIG, exist_ok=True)
RNG = 42; np.random.seed(RNG)
CSV = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
DIFF = ['Neutrophils(%)', 'Lymphocytes(%)', 'Monocytes(%)', 'Eosinophils(%)']
RES = json.load(open("results_final.json")); EXT = json.load(open("external.json"))


def style(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def save(fig, n):
    fig.savefig(os.path.join(FIG, n)); plt.close(fig); print("wrote", n)


# ---- data + model (mirrors dengue_results_final) ----
def load():
    df = pd.read_csv(CSV); df['Gender'] = (df['Gender'] == 'Male').astype(int)
    df['y'] = (df['Result'] == 'positive').astype(int)
    feat = df.drop(columns=['Result', 'y']); key = feat.astype(str).agg('|'.join, axis=1)
    conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
    df = df[~conflict].drop_duplicates(subset=list(feat.columns) + ['y'], keep='first')
    df['PCT(%)'] = df['PCT(%)'].clip(upper=1.0)
    return df.reset_index(drop=True)


def fstd(df):
    comp = df[DIFF].astype(float).values + 0.5; comp = comp / comp.sum(1, keepdims=True)
    clr = np.log(comp) - np.log(comp).mean(1, keepdims=True)
    X = df.drop(columns=DIFF + ['Result', 'y']).copy()
    for i, c in enumerate(DIFF):
        X[f"clr_{c.split('(')[0].strip()}"] = clr[:, i]
    return X


def frich(df, X):
    A = X.copy()
    for c in DIFF:
        A[c.split('(')[0].strip()] = df[c].astype(float)
    neu, lym, mon, eos = [df[c].astype(float) + 0.5 for c in DIFF]
    plt_ = df['Total Platelet Count(/cumm)'].astype(float); wbc = df['Total WBC count(/cumm)'].astype(float)
    hb = df['Hemoglobin(g/dl)'].astype(float); rdw = df['RDW-CV(%)'].astype(float)
    A['NLR'] = neu / lym; A['LMR'] = lym / mon; A['PLR'] = plt_ / lym; A['PLT_WBC'] = plt_ / wbc
    A['MPV_PLT'] = df['MPV(fl)'].astype(float) / plt_ * 1e5; A['RDW_PLT'] = rdw / plt_ * 1e5
    A['Hb_RDW'] = hb / rdw; A['NEU_PLT'] = neu / plt_ * 1e4; A['PLT_LYM_i'] = plt_ * lym / 1e5
    A['NEU_PLT_i'] = neu * plt_ / 1e5; A['WBC_AGE'] = wbc / (df['Age'].astype(float) + 1)
    A['HB_PLT'] = hb * plt_ / 1e5; A['LYM_PLT_i'] = lym * plt_ / 1e5; A['MONO_PLT'] = mon * plt_ / 1e5
    return A


cal = lambda m: CalibratedClassifierCV(m, method="isotonic", cv=5)
LGBM = lambda: LGBMClassifier(random_state=RNG, verbose=-1, n_estimators=700, learning_rate=0.02,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, min_child_samples=30)
XGB = lambda: XGBClassifier(random_state=RNG, eval_metric='logloss', n_estimators=600, learning_rate=0.02,
                            max_depth=3, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0)
CAT = lambda: CatBoostClassifier(random_state=RNG, verbose=0, iterations=700, learning_rate=0.03, depth=4, l2_leaf_reg=6.0)


df = load(); Xs = fstd(df); Xr = frich(df, Xs); y = df['y']
featsR = Xr.columns.tolist()
itr, ite = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y.values[itr], y.values[ite]
scR = StandardScaler().fit(Xr.iloc[itr]); scS = StandardScaler().fit(Xs.iloc[itr])
XtrR = pd.DataFrame(scR.transform(Xr.iloc[itr]), columns=featsR)
XteR = pd.DataFrame(scR.transform(Xr.iloc[ite]), columns=featsR)
XtrS = scS.transform(Xs.iloc[itr]); XteS = scS.transform(Xs.iloc[ite])
prop = VotingClassifier([("l", cal(LGBM())), ("x", cal(XGB())), ("c", cal(CAT()))], voting="soft").fit(XtrR, ytr)
p_prop = prop.predict_proba(XteR)[:, 1]
p_raw = LGBM().fit(XtrR, ytr).predict_proba(XteR)[:, 1]        # uncalibrated core, for reliability
lr3 = LogisticRegression(max_iter=1000, class_weight='balanced')
cors = {i: abs(pointbiserialr(ytr, XtrS[:, i])[0]) for i in range(XtrS.shape[1])}
top3 = [k for k, _ in sorted(cors.items(), key=lambda z: -z[1])[:3]]
p_lr = lr3.fit(XtrS[:, top3], ytr).predict_proba(XteS[:, top3])[:, 1]
p_lgb = LGBMClassifier(random_state=RNG, verbose=-1).fit(XtrS, ytr).predict_proba(XteS)[:, 1]


# ============================ Fig: CV-primary forest =========================
def fig_auc():
    order = sorted(RES["cv"].items(), key=lambda kv: kv[1]["mean"])
    names = [k.replace(" (ours)", "") for k, _ in order]
    mean = [v["mean"] for _, v in order]; std = [v["std"] for _, v in order]
    y_ = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    for i, (k, v) in enumerate(order):
        prop_ = "ours" in k; col = C[1] if prop_ else MUTED
        ax.errorbar(v["mean"], i, xerr=v["std"], fmt='o', color=col, ecolor=col,
                    ms=5 if prop_ else 3.5, elinewidth=1.1, capsize=2, zorder=4)
        ax.text(v["mean"] + v["std"] + 0.004, i, f"{v['mean']:.3f}", va='center',
                fontsize=6.2, color=col, fontweight='bold' if prop_ else 'normal')
    ax.axvline(0.5, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks(y_); ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlabel("Repeated 10×5 CV AUC (mean ± SD)", labelpad=1.5)
    ax.set_xlim(0.48, 0.76); ax.set_xticks([0.5, 0.6, 0.7])
    style(ax, grid="x"); save(fig, "fig_auc.pdf")


# ============================ Fig: ROC + calibration =========================
def fig_roc_cal():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw={"wspace": 0.42})
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 3)))
    for nm, p, col, ls in [("3-feat LR", p_lr, C[3], (0, (3, 2))),
                           ("LightGBM", p_lgb, C[2], (0, (1, 1.5))),
                           ("CGB-CLR", p_prop, C[1], "-")]:
        fpr, tpr, _ = roc_curve(yte, p)
        a1.plot(fpr, tpr, ls=ls, color=col, lw=1.7 if nm == "CGB-CLR" else 1.2,
                label=f"{nm} {roc_auc_score(yte, p):.3f}")
    a1.set_xlabel("False-positive rate", labelpad=1.5); a1.set_ylabel("True-positive rate", labelpad=1.5)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.set_xticks([0, .5, 1]); a1.set_yticks([0, .5, 1])
    a1.legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.2, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4); style(a1, grid="both")
    a2.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 3)))
    for p, lbl, col in [(p_raw, "uncalibrated", C[0]), (p_prop, "calibrated", C[1])]:
        ft, mp = calibration_curve(yte, p, n_bins=8, strategy='quantile')
        a2.plot(mp, ft, '-o', color=col, lw=1.4, ms=3, markeredgecolor="white",
                markeredgewidth=0.5, label=lbl)
    a2.set_xlabel("Predicted probability", labelpad=1.5); a2.set_ylabel("Observed frequency", labelpad=1.5)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.set_xticks([0, .5, 1]); a2.set_yticks([0, .5, 1])
    a2.legend(frameon=False, loc="upper left", fontsize=6.2, handlelength=1.0, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4); style(a2, grid="both")
    save(fig, "fig_roc_cal.pdf")


# ==================== Fig: decision curve + rule-out =========================
def fig_dca_ruleout():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw={"wspace": 0.5})
    ts = np.linspace(0.05, 0.6, 40); N = len(yte)
    nb = []; na = []
    for pt in ts:
        pred = (p_prop >= pt).astype(int); tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
        nb.append(tp / N - fp / N * (pt / (1 - pt)))
        na.append((yte == 1).sum() / N - (yte == 0).sum() / N * (pt / (1 - pt)))
    nb, na = np.array(nb), np.array(na)
    a1.fill_between(ts, na, nb, where=nb >= na, color=C[1], alpha=0.16)
    a1.plot(ts, nb, lw=1.7, color=C[1], label='CGB-CLR'); a1.plot(ts, na, ls=(0, (4, 2)), color=C[0], lw=1.2, label='treat all')
    a1.axhline(0, color=MUTED, lw=0.9, label='treat none')
    a1.set_xlabel("Threshold probability", labelpad=1.5); a1.set_ylabel("Net benefit", labelpad=1.5)
    a1.set_ylim(-0.05, nb.max() * 1.18); a1.set_xticks([0.1, 0.3, 0.5])
    a1.legend(frameon=False, loc="upper right", fontsize=6, handlelength=1.1, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4); style(a1)

    def ro(t):
        fpr, tpr, thr = roc_curve(yte, p_prop); spec = 1 - fpr; ok = np.where(tpr >= t)[0]
        i = ok[np.argmax(spec[ok])]; pred = (p_prop >= thr[i]).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        return spec[i], (pred == 0).mean()
    tg = [0.99, 0.95, 0.90, 0.80]; sp = []; sv = []
    for t in tg:
        s, v = ro(t); sp.append(s); sv.append(v)
    a2.plot(tg, sp, '-o', lw=1.4, ms=3.2, color=C[0], markeredgecolor='white', markeredgewidth=0.5, label='specificity')
    a2.plot(tg, sv, '-D', lw=1.4, ms=3, color=C[1], markeredgecolor='white', markeredgewidth=0.5, label='tests avoided')
    for x, v in zip(tg, sv):
        a2.text(x, v + .022, f"{v:.0%}", ha='center', fontsize=5.8, color=C[1])
    a2.set_xlabel("Required sensitivity", labelpad=1.5); a2.set_ylabel("Fraction", labelpad=1.5)
    a2.set_ylim(0, .62); a2.invert_xaxis()
    a2.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.1, borderpad=0.1,
              labelspacing=0.28, handletextpad=0.4); style(a2)
    save(fig, "fig_dca_ruleout.pdf")


# ==================== Fig: conformal + importance ============================
def fig_conf_imp():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.62),
                                 gridspec_kw={"wspace": 0.52, "width_ratios": [1.0, 1.15]})
    Xt2, Xcal, yt2, ycal = train_test_split(XtrR.values, ytr, test_size=0.3, stratify=ytr, random_state=RNG)
    m = LGBMClassifier(random_state=RNG, verbose=-1, n_estimators=700, learning_rate=0.02,
                       num_leaves=31, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                       min_child_samples=30).fit(Xt2, yt2)
    pc = m.predict_proba(Xcal)[:, 1]; ptest = m.predict_proba(XteR.values)[:, 1]
    al = [0.05, 0.10, 0.20]; cov = [1 - a for a in al]; dec = []; defr = []
    conf = {}
    for a in al:
        q1 = np.quantile((1 - pc)[ycal == 1], 1 - a); q0 = np.quantile(pc[ycal == 0], 1 - a)
        in1 = (1 - ptest) <= q1; in0 = ptest <= q0
        singleton = in1 ^ in0; pred = np.where(in1 & ~in0, 1, 0)
        acc = (pred[singleton] == yte[singleton]).mean() if singleton.any() else np.nan
        dec.append(singleton.mean()); defr.append(1 - singleton.mean())
        conf[a] = dict(singleton=float(singleton.mean()), acc=float(acc))
    x = np.arange(3)
    a1.bar(x, dec, .62, color=C[0], zorder=3, label='decide'); a1.bar(x, defr, .62, bottom=dec, color=C[1], zorder=3, label='defer')
    for i in range(3):
        a1.text(i, dec[i] / 2, f"{dec[i]:.0%}", ha='center', va='center', color='white', fontweight='bold', fontsize=6.2)
        a1.text(i, dec[i] + defr[i] / 2, f"{defr[i]:.0%}", ha='center', va='center', color='white', fontweight='bold', fontsize=6.2)
    a1.set_xticks(x); a1.set_xticklabels([f"{c:.0%}" for c in cov]); a1.set_ylim(0, 1.12)
    a1.set_xlabel("Target coverage", labelpad=1.5); a1.set_ylabel("Fraction of patients", labelpad=1.5)
    a1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=0.9,
              columnspacing=0.9, handletextpad=0.4, fontsize=6.2); style(a1, grid="y")
    pi = permutation_importance(prop, XteR.values, yte, n_repeats=10, random_state=RNG, scoring='roc_auc')
    imp = pd.Series(pi.importances_mean, index=featsR).sort_values(ascending=False).head(8)[::-1]
    NICE = {'Total Platelet Count': 'Platelet', 'clr_Neutrophils': 'Neutrophils*', 'RDW-CV(%)': 'RDW-CV',
            'PLR': 'PLR', 'NLR': 'NLR', 'PLT_WBC': 'PLT/WBC', 'NEU_PLT': 'Neu/PLT', 'clr_Lymphocytes': 'Lymphocytes*',
            'MPV_PLT': 'MPV/PLT', 'HB_PLT': 'Hb·PLT', 'Total WBC count(/cumm)': 'WBC'}
    yy = np.arange(len(imp)); cols = [C[1] if v == imp.index[-1] else C[0] for v in imp.index]
    a2.barh(yy, imp.values, 0.66, color=cols, zorder=3)
    a2.set_yticks(yy); a2.set_yticklabels([NICE.get(k, k) for k in imp.index], fontsize=6.0)
    a2.set_xlabel("Decrease in held-out AUC", labelpad=1.5); a2.set_xlim(0, max(imp.values) * 1.18)
    style(a2, grid="x"); save(fig, "fig_conf_imp.pdf")
    json.dump(conf, open("conf_rev.json", "w"), indent=2)


# ==================== Fig: external validation ===============================
def fig_external():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.5), gridspec_kw={"wspace": 0.5, "width_ratios": [1.05, 1.0]})
    m = EXT["proposed_external"]
    # panel a: internal (CV) vs external metrics
    labels = ['AUC', 'Sens.', 'Spec.', 'NPV']
    intern = [RES["cv"]["CGB-CLR (ours)"]["mean"], RES["holdout"]["CGB-CLR (ours)"]["rec"],
              RES["holdout"]["CGB-CLR (ours)"]["spec"], RES["holdout"]["CGB-CLR (ours)"]["npv"]]
    extern = [m["auc"], m["rec"], m["spec"], m["npv"]]
    x = np.arange(len(labels)); w = 0.38
    a1.bar(x - w / 2, intern, w, color=C[0], label='Jamalpur', zorder=3)
    a1.bar(x + w / 2, extern, w, color=C[1], label='Kalai (ext.)', zorder=3)
    for i, (a, b) in enumerate(zip(intern, extern)):
        a1.text(i - w / 2, a + .02, f"{a:.2f}", ha='center', fontsize=5.4, color=C[0])
        a1.text(i + w / 2, b + .02, f"{b:.2f}", ha='center', fontsize=5.4, color=C[1])
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=6.6); a1.set_ylim(0, 1.12); a1.set_ylabel("Score", labelpad=1.5)
    a1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=6,
              handlelength=0.9, columnspacing=0.9, handletextpad=0.4); style(a1, grid="y")
    # panel b: external ROC
    Xj = load(); yj = Xj['y'].values
    SH = ["Age", "Hemoglobin(g/dl)", "Total WBC count(/cumm)", "Total Platelet Count(/cumm)", "PDW(%)"]
    Xjs = Xj[SH].astype(float).copy(); Xjs['r'] = Xj['Total Platelet Count(/cumm)'] / Xj['Total WBC count(/cumm)']
    z = zipfile.ZipFile("Predictive Clinical Dataset for Dengue Fever Using Vital Signs and Blood Parameters.zip")
    n = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
    dk = pd.read_csv(z.open(n)).dropna(subset=["Age", "Haemoglobin", "WBC Count", "Platelet Count", "PDW", "Final Output"])
    Xk = pd.DataFrame({"Age": dk["Age"], "Hemoglobin(g/dl)": dk["Haemoglobin"], "Total WBC count(/cumm)": dk["WBC Count"],
                       "Total Platelet Count(/cumm)": dk["Platelet Count"], "PDW(%)": dk["PDW"]}).astype(float)
    Xk['r'] = Xk['Total Platelet Count(/cumm)'] / Xk['Total WBC count(/cumm)']; yk = dk["Final Output"].astype(int).values
    sc = StandardScaler().fit(Xjs)
    md = cal(GradientBoostingClassifier(random_state=RNG)).fit(sc.transform(Xjs), yj)
    pk = md.predict_proba(sc.transform(Xk[Xjs.columns]))[:, 1]
    fpr, tpr, _ = roc_curve(yk, pk)
    a2.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 3)))
    a2.plot(fpr, tpr, color=C[1], lw=1.7, label=f"external {roc_auc_score(yk, pk):.3f}")
    a2.set_xlabel("False-positive rate", labelpad=1.5); a2.set_ylabel("True-positive rate", labelpad=1.5)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.set_xticks([0, .5, 1]); a2.set_yticks([0, .5, 1])
    a2.legend(frameon=False, loc="lower right", fontsize=6.2, handlelength=1.1, borderpad=0.1); style(a2, grid="both")
    save(fig, "fig_external.pdf")


if __name__ == "__main__":
    fig_auc(); fig_roc_cal(); fig_dca_ruleout(); fig_conf_imp(); fig_external()
    print("done")
