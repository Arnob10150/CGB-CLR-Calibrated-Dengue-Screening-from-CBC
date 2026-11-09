#!/usr/bin/env python3
"""Reviewer-response analyses for the dengue paper that need no new data:
R4 statistical significance (DeLong paired-AUC test vs the proposed model),
R5 test-set stability (repeated stratified 5x5 CV per model),
R8 advanced models (LightGBM, CatBoost, stacked ensemble),
R9 data-cleaning sensitivity (metrics with vs without the integrity gate).
Saves everything to advanced.json.
"""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, brier_score_loss, confusion_matrix)
from scipy.stats import pointbiserialr
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RNG = 42; np.random.seed(RNG)
CSV = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
DIFF = ['Neutrophils(%)', 'Lymphocytes(%)', 'Monocytes(%)', 'Eosinophils(%)']


# ----------------------------------------------------------------- DeLong test
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float); T2[J] = T
    return T2


def delong_var(y, preds):
    """Fast DeLong. preds: (k, n) array of scores; y binary. Returns AUCs and cov."""
    order = (-y).argsort(kind="mergesort")
    label1 = int(np.sum(y == 1))
    preds = preds[:, order]; m, n = label1, len(y) - label1
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([_compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([_compute_midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_p(y, prob_a, prob_b):
    """Two-sided p-value that AUC_a == AUC_b (paired)."""
    y = np.asarray(y, float)
    preds = np.vstack([np.asarray(prob_a, float), np.asarray(prob_b, float)])
    aucs, cov = delong_var(y, preds)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, float(2 * (1 - stats.norm.cdf(abs(z))))


# ----------------------------------------------------------------- data / clean
def load(clean=True):
    raw = pd.read_csv(CSV)
    df = raw.copy()
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    df['y'] = (df['Result'] == 'positive').astype(int)
    if clean:
        feat = df.drop(columns=['Result', 'y'])
        key = feat.astype(str).agg('|'.join, axis=1)
        conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
        df = df[~conflict].drop_duplicates(subset=list(feat.columns) + ['y'], keep='first')
        df['PCT(%)'] = df['PCT(%)'].clip(upper=1.0)
    df = df.reset_index(drop=True)
    comp = df[DIFF].astype(float).values + 0.5; comp = comp / comp.sum(1, keepdims=True)
    clr = np.log(comp) - np.log(comp).mean(1, keepdims=True)
    Xc = df.drop(columns=DIFF + ['Result', 'y']).copy()
    for i, c in enumerate(DIFF):
        Xc[f"clr_{c.split('(')[0].strip()}"] = clr[:, i]
    return Xc, df['y']


def metrics(y, p, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(acc=accuracy_score(y, pred), prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred), f1=f1_score(y, pred),
                auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                spec=tn / (tn + fp), npv=tn / (tn + fn) if (tn + fn) else np.nan)


def ci(y, p, n=2000):
    idx = np.arange(len(y)); v = []
    for _ in range(n):
        b = np.random.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        v.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


R = {}
Xc, y = load(clean=True)
feats = Xc.columns.tolist()
Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.25, stratify=y, random_state=RNG)
sc = StandardScaler().fit(Xtr)
Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats, index=Xtr.index)
Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats, index=Xte.index)
cors = {c: abs(pointbiserialr(ytr, Xtr[c])[0]) for c in feats}
top3 = [k for k, _ in sorted(cors.items(), key=lambda z: -z[1])[:3]]

BAL = dict(class_weight='balanced')
base = {
    "3-feature LR": (LogisticRegression(max_iter=1000, **BAL), top3),
    "Full LR": (LogisticRegression(max_iter=1000, **BAL), feats),
    "Random Forest": (RandomForestClassifier(random_state=RNG, **BAL), feats),
    "XGBoost": (XGBClassifier(random_state=RNG, eval_metric='logloss'), feats),
    "RBF-SVM": (SVC(probability=True, random_state=RNG, **BAL), feats),
    "MLP": (MLPClassifier(random_state=RNG, max_iter=800), feats),
    "LightGBM": (LGBMClassifier(random_state=RNG, verbose=-1), feats),
    "CatBoost": (CatBoostClassifier(random_state=RNG, verbose=0), feats),
}
probs = {}; rows = {}
for nm, (mdl, cols) in base.items():
    mdl.fit(Xtr_s[cols], ytr); p = mdl.predict_proba(Xte_s[cols])[:, 1]
    probs[nm] = p; pred = (p >= .5).astype(int)
    m = metrics(yte.values, p, pred); m["lo"], m["hi"] = ci(yte.values, p)
    rows[nm] = m

# stacked ensemble (RF + XGB + LGBM -> LR meta)
stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(random_state=RNG, **BAL)),
                ("xgb", XGBClassifier(random_state=RNG, eval_metric='logloss')),
                ("lgbm", LGBMClassifier(random_state=RNG, verbose=-1))],
    final_estimator=LogisticRegression(max_iter=1000), cv=5)
stack.fit(Xtr_s, ytr); p = stack.predict_proba(Xte_s)[:, 1]
probs["Stacked ens."] = p
m = metrics(yte.values, p, (p >= .5).astype(int)); m["lo"], m["hi"] = ci(yte.values, p)
rows["Stacked ens."] = m

# proposed CGB-CLR
cal = CalibratedClassifierCV(GradientBoostingClassifier(random_state=RNG),
                             method='isotonic', cv=5).fit(Xtr_s, ytr)
p_cal = cal.predict_proba(Xte_s)[:, 1]
probs["CGB-CLR"] = p_cal
m = metrics(yte.values, p_cal, (p_cal >= .5).astype(int)); m["lo"], m["hi"] = ci(yte.values, p_cal)
rows["CGB-CLR"] = m
R["table"] = rows

# ------------------------------------------------- R4 DeLong vs proposed
delong = {}
for nm, p in probs.items():
    if nm == "CGB-CLR":
        continue
    aucs, pv = delong_p(yte.values, p_cal, p)
    delong[nm] = dict(auc_other=float(aucs[1]), auc_prop=float(aucs[0]), p=pv)
R["delong"] = delong

# ------------------------------------------------- R5 repeated 5x5 CV AUC
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
Xall = StandardScaler().fit_transform(Xc)
cv = {}
for nm, mdl in [("Random Forest", RandomForestClassifier(random_state=RNG, **BAL)),
                ("XGBoost", XGBClassifier(random_state=RNG, eval_metric='logloss')),
                ("LightGBM", LGBMClassifier(random_state=RNG, verbose=-1)),
                ("CatBoost", CatBoostClassifier(random_state=RNG, verbose=0)),
                ("GB (CGB-CLR core)", GradientBoostingClassifier(random_state=RNG))]:
    s = cross_val_score(mdl, Xall, y, cv=rcv, scoring='roc_auc')
    cv[nm] = dict(mean=float(s.mean()), std=float(s.std()),
                  lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)))
R["cv"] = cv

# ------------------------------------------------- R9 cleaning sensitivity
sens = {}
for tag, cl in (("cleaned", True), ("raw (no gate)", False)):
    Xc2, y2 = load(clean=cl)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xc2, y2, test_size=0.25, stratify=y2,
                                              random_state=RNG)
    sc2 = StandardScaler().fit(Xtr2)
    Xtr2s = sc2.transform(Xtr2); Xte2s = sc2.transform(Xte2)
    cal2 = CalibratedClassifierCV(GradientBoostingClassifier(random_state=RNG),
                                  method='isotonic', cv=5).fit(Xtr2s, ytr2)
    p2 = cal2.predict_proba(Xte2s)[:, 1]
    m = metrics(yte2.values, p2, (p2 >= .5).astype(int))
    m["n"] = int(len(Xc2))
    sens[tag] = m
R["cleaning"] = sens

json.dump(R, open("advanced.json", "w"), indent=2, default=float)
print("== R8 models (test AUC / acc / spec / Brier) ==")
for nm, m in rows.items():
    print(f"  {nm:16s} auc={m['auc']:.3f} [{m['lo']:.3f},{m['hi']:.3f}] "
          f"acc={m['acc']:.3f} rec={m['rec']:.3f} spec={m['spec']:.3f} brier={m['brier']:.3f}")
print("== R4 DeLong vs CGB-CLR (p) ==")
for nm, d in delong.items():
    print(f"  CGB-CLR({d['auc_prop']:.3f}) vs {nm}({d['auc_other']:.3f}): p={d['p']:.3f}")
print("== R5 repeated 5x5 CV AUC ==")
for nm, c in cv.items():
    print(f"  {nm:18s} {c['mean']:.3f}±{c['std']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]")
print("== R9 cleaning sensitivity ==")
for tag, m in sens.items():
    print(f"  {tag:14s} n={m['n']} auc={m['auc']:.3f} acc={m['acc']:.3f} "
          f"rec={m['rec']:.3f} spec={m['spec']:.3f} brier={m['brier']:.3f}")
