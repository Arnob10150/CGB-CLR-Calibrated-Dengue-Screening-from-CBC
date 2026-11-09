#!/usr/bin/env python3
"""Rerun CGB-CLR exactly as in dengue_results_final.py (same seed/config/split),
then compute precise alternate operating points: Youden's J optimum, and
specificity/rule-out yield at sensitivity >= 0.95/0.90/0.85/0.80, to replace the
paper's qualitative threshold discussion with exact, reproducible numbers.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RNG = 42; np.random.seed(RNG)
CSV = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
DIFF = ['Neutrophils(%)', 'Lymphocytes(%)', 'Monocytes(%)', 'Eosinophils(%)']


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

df = load(); Xstd = feats_std(df); Xrich = feats_rich(df, Xstd); y = df['y']
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y.values[idx_tr], y.values[idx_te]
scR = StandardScaler().fit(Xrich.iloc[idx_tr])
XtrR, XteR = scR.transform(Xrich.iloc[idx_tr]), scR.transform(Xrich.iloc[idx_te])

prop = proposed().fit(XtrR, ytr)
p = prop.predict_proba(XteR)[:, 1]

fpr, tpr, thr = roc_curve(yte, p)
spec = 1 - fpr
youden = tpr - fpr
best_idx = np.argmax(youden)
print(f"Youden-optimal threshold = {thr[best_idx]:.4f}  sens={tpr[best_idx]:.4f}  spec={spec[best_idx]:.4f}")

pred_y = (p >= thr[best_idx]).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred_y).ravel()
acc_y = (tp + tn) / (tp + tn + fp + fn)
npv_y = tn / (tn + fn) if (tn + fn) else float('nan')
print(f"  at Youden thr: acc={acc_y:.4f} npv={npv_y:.4f} TP={tp} TN={tn} FP={fp} FN={fn}")

print("\n== operating points at fixed sensitivity targets ==")
results = {"youden": dict(threshold=float(thr[best_idx]), sens=float(tpr[best_idx]),
                           spec=float(spec[best_idx]), acc=float(acc_y), npv=float(npv_y),
                           tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))}
targets = [0.98, 0.95, 0.90, 0.85, 0.80]
op_points = {}
N = len(yte)
for target in targets:
    # find smallest threshold achieving sens >= target (scan all unique probs)
    cand_idx = np.where(tpr >= target)[0]
    if len(cand_idx) == 0:
        continue
    # among those achieving >= target sens, pick the one with highest spec (i.e. highest threshold)
    best = cand_idx[np.argmax(spec[cand_idx])]
    t_ = thr[best]
    pred_t = (p >= t_).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(yte, pred_t).ravel()
    tests_avoided = tn_ / N
    npv_t = tn_ / (tn_ + fn_) if (tn_ + fn_) else float('nan')
    op_points[target] = dict(threshold=float(t_), sens=float(tpr[best]), spec=float(spec[best]),
                              tests_avoided=float(tests_avoided), npv=float(npv_t),
                              tp=int(tp_), tn=int(tn_), fp=int(fp_), fn=int(fn_))
    print(f"  sens>={target:.2f}: thr={t_:.4f} actual_sens={tpr[best]:.4f} spec={spec[best]:.4f} "
          f"tests_avoided={tests_avoided:.3f} npv={npv_t:.4f} (TP={tp_} TN={tn_} FP={fp_} FN={fn_})")

results["operating_points"] = op_points
json.dump(results, open("threshold_results.json", "w"), indent=2)
print("\nSaved threshold_results.json")
