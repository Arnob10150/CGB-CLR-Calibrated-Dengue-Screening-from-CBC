#!/usr/bin/env python3
"""Definitive results for the revised dengue paper. Proposed model = CGB-CLR:
calibrated LightGBM+XGBoost+CatBoost soft-ensemble on CLR + dengue ratios +
clinical interactions. Primary metric = repeated 5x5 CV AUC; hold-out reported
with bootstrap CIs; DeLong (hold-out) and paired t-test (CV) vs every baseline.
Saves results_final.json."""
import warnings, sys, io; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, json
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import pointbiserialr
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RNG = 42; np.random.seed(RNG)
CSV = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
DIFF = ['Neutrophils(%)', 'Lymphocytes(%)', 'Monocytes(%)', 'Eosinophils(%)']
BAL = dict(class_weight='balanced')


def load():
    raw = pd.read_csv(CSV); df = raw.copy()
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    df['y'] = (df['Result'] == 'positive').astype(int)
    feat = df.drop(columns=['Result', 'y']); key = feat.astype(str).agg('|'.join, axis=1)
    conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
    df = df[~conflict].drop_duplicates(subset=list(feat.columns) + ['y'], keep='first')
    df['PCT(%)'] = df['PCT(%)'].clip(upper=1.0)
    return df.reset_index(drop=True)


def feats_std(df):
    comp = df[DIFF].astype(float).values + 0.5; comp = comp / comp.sum(1, keepdims=True)
    clr = np.log(comp) - np.log(comp).mean(1, keepdims=True)
    X = df.drop(columns=DIFF + ['Result', 'y']).copy()
    for i, c in enumerate(DIFF):
        X[f"clr_{c.split('(')[0].strip()}"] = clr[:, i]
    return X


def feats_rich(df, Xstd):
    A = Xstd.copy()
    for c in DIFF:
        A[c] = df[c].astype(float)
    neu, lym, mon, eos = [df[c].astype(float) + 0.5 for c in DIFF]
    plt_ = df['Total Platelet Count(/cumm)'].astype(float); wbc = df['Total WBC count(/cumm)'].astype(float)
    hb = df['Hemoglobin(g/dl)'].astype(float); rdw = df['RDW-CV(%)'].astype(float)
    A['NLR'] = neu / lym; A['LMR'] = lym / mon; A['PLR'] = plt_ / lym; A['PLT_WBC'] = plt_ / wbc
    A['MPV_PLT'] = df['MPV(fl)'].astype(float) / plt_ * 1e5; A['RDW_PLT'] = rdw / plt_ * 1e5
    A['Hb_RDW'] = hb / rdw; A['NEU_PLT'] = neu / plt_ * 1e4
    A['PLT_LYM_i'] = plt_ * lym / 1e5; A['NEU_PLT_i'] = neu * plt_ / 1e5
    A['WBC_AGE'] = wbc / (df['Age'].astype(float) + 1); A['HB_PLT'] = hb * plt_ / 1e5
    A['LYM_PLT_i'] = lym * plt_ / 1e5; A['MONO_PLT'] = mon * plt_ / 1e5
    return A


LGBM = lambda: LGBMClassifier(random_state=RNG, verbose=-1, n_estimators=700, learning_rate=0.02,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                              min_child_samples=30)
XGB = lambda: XGBClassifier(random_state=RNG, eval_metric='logloss', n_estimators=600,
                            learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.7,
                            reg_lambda=3.0)
CAT = lambda: CatBoostClassifier(random_state=RNG, verbose=0, iterations=700, learning_rate=0.03,
                                 depth=4, l2_leaf_reg=6.0)
cal = lambda m: CalibratedClassifierCV(m, method="isotonic", cv=5)
proposed = lambda: VotingClassifier([("l", cal(LGBM())), ("x", cal(XGB())), ("c", cal(CAT()))],
                                     voting="soft")


# ------ DeLong ------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong_p(y, pa, pb):
    y = np.asarray(y, float); order = (-y).argsort(kind="mergesort")
    preds = np.vstack([pa, pb])[:, order]; m = int((y == 1).sum()); n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(pos[r]) for r in range(2)]); ty = np.array([_midrank(neg[r]) for r in range(2)])
    tz = np.array([_midrank(preds[r]) for r in range(2)])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2.) / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1. - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, float(2 * (1 - stats.norm.cdf(abs(z))))


df = load(); Xstd = feats_std(df); Xrich = feats_rich(df, Xstd); y = df['y']
featsS = Xstd.columns.tolist()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y.values[idx_tr], y.values[idx_te]
scS = StandardScaler().fit(Xstd.iloc[idx_tr]); scR = StandardScaler().fit(Xrich.iloc[idx_tr])
XtrS, XteS = scS.transform(Xstd.iloc[idx_tr]), scS.transform(Xstd.iloc[idx_te])
XtrR, XteR = scR.transform(Xrich.iloc[idx_tr]), scR.transform(Xrich.iloc[idx_te])
cors = {c: abs(pointbiserialr(ytr, XtrS[:, i])[0]) for i, c in enumerate(featsS)}
top3 = [featsS.index(k) for k, _ in sorted(cors.items(), key=lambda z: -z[1])[:3]]


def bci(y_, p_, n=2000):
    ii = np.arange(len(y_)); v = []
    for _ in range(n):
        b = np.random.choice(ii, len(ii), replace=True)
        if len(np.unique(y_[b])) > 1:
            v.append(roc_auc_score(y_[b], p_[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def holdout(y_, p_):
    pred = (p_ >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
    lo, hi = bci(y_, p_)
    return dict(acc=accuracy_score(y_, pred), rec=recall_score(y_, pred),
                f1=f1_score(y_, pred), auc=roc_auc_score(y_, p_), spec=tn / (tn + fp),
                npv=tn / (tn + fn) if (tn + fn) else np.nan, brier=brier_score_loss(y_, p_),
                lo=lo, hi=hi)


baselines = {
    "Majority": None, "3-feature LR": (LogisticRegression(max_iter=1000, **BAL), top3),
    "Full LR": (LogisticRegression(max_iter=1000, **BAL), None),
    "Random Forest": (RandomForestClassifier(random_state=RNG, **BAL), None),
    "XGBoost": (XGBClassifier(random_state=RNG, eval_metric='logloss'), None),
    "RBF-SVM": (SVC(probability=True, random_state=RNG, **BAL), None),
    "MLP": (MLPClassifier(random_state=RNG, max_iter=800), None),
    "LightGBM": (LGBMClassifier(random_state=RNG, verbose=-1), None),
    "CatBoost": (CatBoostClassifier(random_state=RNG, verbose=0), None),
}
probs = {}; ho = {}
for nm, spec in baselines.items():
    if nm == "Majority":
        p = np.full(len(yte), ytr.mean()); ho[nm] = holdout(yte, p); ho[nm]["auc"] = 0.5
        ho[nm]["rec"] = 1.0; probs[nm] = p; continue
    mdl, cols = spec
    Xtr_, Xte_ = (XtrS[:, cols], XteS[:, cols]) if cols else (XtrS, XteS)
    mdl.fit(Xtr_, ytr); p = mdl.predict_proba(Xte_)[:, 1]; probs[nm] = p; ho[nm] = holdout(yte, p)
st = StackingClassifier([("rf", RandomForestClassifier(random_state=RNG, **BAL)),
                         ("xgb", XGBClassifier(random_state=RNG, eval_metric='logloss')),
                         ("lgbm", LGBMClassifier(random_state=RNG, verbose=-1))],
                        final_estimator=LogisticRegression(max_iter=1000), cv=5).fit(XtrS, ytr)
probs["Stacked ens."] = st.predict_proba(XteS)[:, 1]; ho["Stacked ens."] = holdout(yte, probs["Stacked ens."])
prop = proposed().fit(XtrR, ytr); probs["CGB-CLR (ours)"] = prop.predict_proba(XteR)[:, 1]
ho["CGB-CLR (ours)"] = holdout(yte, probs["CGB-CLR (ours)"])

# repeated CV (primary)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
XallS = StandardScaler().fit_transform(Xstd); XallR = StandardScaler().fit_transform(Xrich)
cvmodels = {"3-feature LR": (LogisticRegression(max_iter=1000, **BAL), XallS[:, top3]),
            "Full LR": (LogisticRegression(max_iter=1000, **BAL), XallS),
            "Random Forest": (RandomForestClassifier(random_state=RNG, **BAL), XallS),
            "XGBoost": (XGBClassifier(random_state=RNG, eval_metric='logloss'), XallS),
            "RBF-SVM": (SVC(probability=True, random_state=RNG, **BAL), XallS),
            "MLP": (MLPClassifier(random_state=RNG, max_iter=800), XallS),
            "LightGBM": (LGBMClassifier(random_state=RNG, verbose=-1), XallS),
            "CatBoost": (CatBoostClassifier(random_state=RNG, verbose=0), XallS)}
cv = {}; cvraw = {}
for nm, (mdl, X) in cvmodels.items():
    s = cross_val_score(mdl, X, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
    cv[nm] = dict(mean=float(s.mean()), std=float(s.std())); cvraw[nm] = s
s = cross_val_score(proposed(), XallR, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
cv["CGB-CLR (ours)"] = dict(mean=float(s.mean()), std=float(s.std())); cvraw["CGB-CLR (ours)"] = s

# DeLong (hold-out) + paired t-test (CV) vs proposed
pp = probs["CGB-CLR (ours)"]; pr = cvraw["CGB-CLR (ours)"]
sig = {}
for nm in ho:
    if nm in ("CGB-CLR (ours)", "Majority"):
        continue
    _, dp = delong_p(yte, pp, probs[nm])
    d = dict(delong_p=float(dp), cv_delta=None, cv_p=None)
    if nm in cvraw:
        d["cv_delta"] = float(pr.mean() - cvraw[nm].mean())
        d["cv_p"] = float(stats.ttest_rel(pr, cvraw[nm])[1])
    sig[nm] = d

json.dump({"holdout": ho, "cv": cv, "sig": sig,
           "n_features_rich": Xrich.shape[1], "n_features_std": Xstd.shape[1]},
          open("results_final.json", "w"), indent=2, default=float)
print("== repeated 5x5 CV AUC (PRIMARY) ==")
for nm, c in sorted(cv.items(), key=lambda kv: -kv[1]['mean']):
    print(f"  {nm:16s} {c['mean']:.3f}±{c['std']:.3f}{'  <-- proposed' if 'ours' in nm else ''}")
print("== hold-out (reference) ==")
for nm, m in sorted(ho.items(), key=lambda kv: -kv[1]['auc']):
    print(f"  {nm:16s} auc={m['auc']:.3f}[{m['lo']:.2f},{m['hi']:.2f}] acc={m['acc']:.3f} "
          f"rec={m['rec']:.3f} spec={m['spec']:.3f} brier={m['brier']:.3f}{'  <-- proposed' if 'ours' in nm else ''}")
print("== significance vs proposed ==")
for nm, d in sig.items():
    print(f"  {nm:16s} DeLong p={d['delong_p']:.3f}  CV Δ={d['cv_delta']:+.3f} p={d['cv_p']}")
