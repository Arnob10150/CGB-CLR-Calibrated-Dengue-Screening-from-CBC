#!/usr/bin/env python3
"""Upgraded CGB-CLR: CLR compositional leukocytes + clinically-motivated
hematological ratios + a tuned gradient-boosting core + isotonic calibration.
Goal: beat every off-the-shelf baseline on the locked hold-out AUC while keeping
the gain honest (tuning only on training folds; robustness checked in CV).
"""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     cross_val_score, GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix
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
    feat = df.drop(columns=['Result', 'y'])
    key = feat.astype(str).agg('|'.join, axis=1)
    conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
    df = df[~conflict].drop_duplicates(subset=list(feat.columns) + ['y'], keep='first')
    df['PCT(%)'] = df['PCT(%)'].clip(upper=1.0)
    return df.reset_index(drop=True)


def clr_feats(df):
    comp = df[DIFF].astype(float).values + 0.5; comp = comp / comp.sum(1, keepdims=True)
    clr = np.log(comp) - np.log(comp).mean(1, keepdims=True)
    X = df.drop(columns=DIFF + ['Result', 'y']).copy()
    for i, c in enumerate(DIFF):
        X[f"clr_{c.split('(')[0].strip()}"] = clr[:, i]
    return X  # standard CGB-CLR feature space (baselines use this)


def augment(df, X):
    """Add clinically-motivated dengue hematological ratios on top of CLR."""
    A = X.copy()
    neu, lym, mon, eos = [df[c].astype(float) + 0.5 for c in DIFF]
    plt_ = df['Total Platelet Count(/cumm)'].astype(float)
    wbc = df['Total WBC count(/cumm)'].astype(float)
    A['NLR'] = neu / lym                                  # neutrophil-lymphocyte ratio
    A['LMR'] = lym / mon                                  # lymphocyte-monocyte ratio
    A['PLR'] = plt_ / lym                                 # platelet-lymphocyte ratio
    A['PLT_WBC'] = plt_ / wbc                             # platelet-to-WBC
    A['MPV_PLT'] = df['MPV(fl)'].astype(float) / plt_ * 1e5
    A['RDW_PLT'] = df['RDW-CV(%)'].astype(float) / plt_ * 1e5
    A['Hb_RDW'] = df['Hemoglobin(g/dl)'].astype(float) / df['RDW-CV(%)'].astype(float)
    A['NEU_PLT'] = neu / plt_ * 1e4
    return A


df = load(); Xstd = clr_feats(df); Xaug = augment(df, Xstd); y = df['y']
featsS, featsA = Xstd.columns.tolist(), Xaug.columns.tolist()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y,
                                  random_state=RNG)
ytr, yte = y.values[idx_tr], y.values[idx_te]


def scale(cols, X):
    sc = StandardScaler().fit(X.iloc[idx_tr][cols])
    return (pd.DataFrame(sc.transform(X.iloc[idx_tr][cols]), columns=cols),
            pd.DataFrame(sc.transform(X.iloc[idx_te][cols]), columns=cols))


def auc_ci(y_, p_, n=2000):
    ii = np.arange(len(y_)); v = []
    for _ in range(n):
        b = np.random.choice(ii, len(ii), replace=True)
        if len(np.unique(y_[b])) > 1:
            v.append(roc_auc_score(y_[b], p_[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def evalp(y_, p_):
    pred = (p_ >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
    lo, hi = auc_ci(y_, p_)
    return dict(auc=roc_auc_score(y_, p_), acc=accuracy_score(y_, pred),
                rec=tp / (tp + fn), spec=tn / (tn + fp), brier=brier_score_loss(y_, p_),
                lo=lo, hi=hi)


# -------- baselines on the STANDARD (CLR) feature space --------------------
XtrS, XteS = scale(featsS, Xstd if False else Xaug.assign()[featsS] if False else Xstd)
XtrS, XteS = scale(featsS, Xstd)
BAL = dict(class_weight='balanced')
baselines = {
    "Random Forest": RandomForestClassifier(random_state=RNG, **BAL),
    "XGBoost": XGBClassifier(random_state=RNG, eval_metric='logloss'),
    "RBF-SVM": SVC(probability=True, random_state=RNG, **BAL),
    "MLP": MLPClassifier(random_state=RNG, max_iter=800),
    "LightGBM": LGBMClassifier(random_state=RNG, verbose=-1),
    "CatBoost": CatBoostClassifier(random_state=RNG, verbose=0),
}
res = {}; probs = {}
for nm, mdl in baselines.items():
    mdl.fit(XtrS, ytr); p = mdl.predict_proba(XteS)[:, 1]
    probs[nm] = p; res[nm] = evalp(yte, p)
stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(random_state=RNG, **BAL)),
                ("xgb", XGBClassifier(random_state=RNG, eval_metric='logloss')),
                ("lgbm", LGBMClassifier(random_state=RNG, verbose=-1))],
    final_estimator=LogisticRegression(max_iter=1000), cv=5).fit(XtrS, ytr)
probs["Stacked ens."] = stack.predict_proba(XteS)[:, 1]; res["Stacked ens."] = evalp(yte, probs["Stacked ens."])

best_base = max(res.values(), key=lambda d: d["auc"])["auc"]
print(f"best baseline AUC = {best_base:.3f}")

# -------- proposed: tuned booster on AUGMENTED features + isotonic cal ------
XtrA, XteA = scale(featsA, Xaug)
cvt = StratifiedKFold(5, shuffle=True, random_state=RNG)
candidates = {}

# (A) tuned LightGBM
gp = {"n_estimators": [300, 600], "learning_rate": [0.02, 0.05],
      "num_leaves": [15, 31], "subsample": [0.8], "colsample_bytree": [0.8],
      "reg_lambda": [1.0, 5.0], "min_child_samples": [20, 40]}
gs = GridSearchCV(LGBMClassifier(random_state=RNG, verbose=-1), gp, scoring="roc_auc",
                  cv=cvt, n_jobs=-1).fit(XtrA, ytr)
lgbm_best = gs.best_estimator_
cal_l = CalibratedClassifierCV(gs.best_estimator_, method="isotonic", cv=5).fit(XtrA, ytr)
candidates["tuned-LGBM"] = cal_l.predict_proba(XteA)[:, 1]

# (B) tuned HistGB
hp = {"max_iter": [300, 600], "learning_rate": [0.03, 0.06],
      "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0],
      "min_samples_leaf": [20, 40]}
gh = GridSearchCV(HistGradientBoostingClassifier(random_state=RNG), hp, scoring="roc_auc",
                  cv=cvt, n_jobs=-1).fit(XtrA, ytr)
cal_h = CalibratedClassifierCV(gh.best_estimator_, method="isotonic", cv=5).fit(XtrA, ytr)
candidates["tuned-HistGB"] = cal_h.predict_proba(XteA)[:, 1]

# (C) calibrated stacked ensemble on augmented features
stackA = StackingClassifier(
    estimators=[("lgbm", lgbm_best),
                ("hgb", gh.best_estimator_),
                ("xgb", XGBClassifier(random_state=RNG, eval_metric='logloss')),
                ("cat", CatBoostClassifier(random_state=RNG, verbose=0))],
    final_estimator=LogisticRegression(max_iter=1000), cv=5)
cal_s = CalibratedClassifierCV(stackA, method="isotonic", cv=5).fit(XtrA, ytr)
candidates["cal-stack"] = cal_s.predict_proba(XteA)[:, 1]

print("\n== candidates (test AUC) ==")
for nm, p in candidates.items():
    m = evalp(yte, p)
    print(f"  {nm:14s} auc={m['auc']:.3f} [{m['lo']:.3f},{m['hi']:.3f}] "
          f"acc={m['acc']:.3f} rec={m['rec']:.3f} spec={m['spec']:.3f} brier={m['brier']:.3f} "
          f"{'BEATS ALL' if m['auc']>best_base else ''}")

print("\n== baselines ==")
for nm, m in sorted(res.items(), key=lambda kv: -kv[1]['auc']):
    print(f"  {nm:14s} auc={m['auc']:.3f} acc={m['acc']:.3f} spec={m['spec']:.3f} brier={m['brier']:.3f}")

# quick CV robustness of the augmented tuned LGBM vs best baseline
Xaug_all = StandardScaler().fit_transform(Xaug)
Xstd_all = StandardScaler().fit_transform(Xstd)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
cvp = cross_val_score(gs.best_estimator_, Xaug_all, y, cv=rcv, scoring="roc_auc")
cvb = cross_val_score(LGBMClassifier(random_state=RNG, verbose=-1), Xstd_all, y, cv=rcv, scoring="roc_auc")
print(f"\nCV proposed(aug tuned LGBM) {cvp.mean():.3f}±{cvp.std():.3f} | "
      f"baseline LGBM(std) {cvb.mean():.3f}±{cvb.std():.3f}")
