#!/usr/bin/env python3
"""Final CGB-CLR framework: CLR compositional leukocytes + dengue hematological
ratios, a calibrated soft-ensemble of a tuned LightGBM and a tuned histogram
gradient booster, isotonic calibration, conformal abstention downstream.
Primary metric is repeated stratified 5x5 CV (the 377-sample split is unstable,
reviewer R5); the held-out split is reported for reference. Saves final.json.
"""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, HistGradientBoostingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix
from sklearn.base import clone
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
    return X


def augment(df, X):
    A = X.copy()
    neu, lym, mon = [df[c].astype(float) + 0.5 for c in DIFF[:3]]
    plt_ = df['Total Platelet Count(/cumm)'].astype(float)
    wbc = df['Total WBC count(/cumm)'].astype(float)
    A['NLR'] = neu / lym; A['LMR'] = lym / mon; A['PLR'] = plt_ / lym
    A['PLT_WBC'] = plt_ / wbc
    A['MPV_PLT'] = df['MPV(fl)'].astype(float) / plt_ * 1e5
    A['RDW_PLT'] = df['RDW-CV(%)'].astype(float) / plt_ * 1e5
    A['Hb_RDW'] = df['Hemoglobin(g/dl)'].astype(float) / df['RDW-CV(%)'].astype(float)
    A['NEU_PLT'] = neu / plt_ * 1e4
    return A


def proposed():
    lgbm = LGBMClassifier(random_state=RNG, verbose=-1, n_estimators=600,
                          learning_rate=0.02, num_leaves=31, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=5.0, min_child_samples=40)
    hgb = HistGradientBoostingClassifier(random_state=RNG, max_iter=600,
                                         learning_rate=0.03, max_leaf_nodes=31,
                                         l2_regularization=1.0, min_samples_leaf=40)
    vote = VotingClassifier(
        [("lgbm", CalibratedClassifierCV(lgbm, method="isotonic", cv=5)),
         ("hgb", CalibratedClassifierCV(hgb, method="isotonic", cv=5))],
        voting="soft")
    return vote


df = load(); Xstd = clr_feats(df); Xaug = augment(df, Xstd); y = df['y']
featsS, featsA = Xstd.columns.tolist(), Xaug.columns.tolist()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y.values[idx_tr], y.values[idx_te]
scS = StandardScaler().fit(Xstd.iloc[idx_tr]); scA = StandardScaler().fit(Xaug.iloc[idx_tr])
XtrS = scS.transform(Xstd.iloc[idx_tr]); XteS = scS.transform(Xstd.iloc[idx_te])
XtrA = scA.transform(Xaug.iloc[idx_tr]); XteA = scA.transform(Xaug.iloc[idx_te])


def auc_ci(y_, p_, n=2000):
    ii = np.arange(len(y_)); v = []
    for _ in range(n):
        b = np.random.choice(ii, len(ii), replace=True)
        if len(np.unique(y_[b])) > 1:
            v.append(roc_auc_score(y_[b], p_[b]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def evalp(y_, p_):
    pred = (p_ >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
    lo, hi = auc_ci(y_, p_)
    return dict(auc=roc_auc_score(y_, p_), acc=accuracy_score(y_, pred),
                rec=tp / (tp + fn), spec=tn / (tn + fp), brier=brier_score_loss(y_, p_),
                lo=lo, hi=hi)


# baselines (standard CLR features) + proposed (augmented)
models_std = {
    "3-feature LR": LogisticRegression(max_iter=1000, **BAL),
    "Full LR": LogisticRegression(max_iter=1000, **BAL),
    "Random Forest": RandomForestClassifier(random_state=RNG, **BAL),
    "XGBoost": XGBClassifier(random_state=RNG, eval_metric='logloss'),
    "RBF-SVM": SVC(probability=True, random_state=RNG, **BAL),
    "MLP": MLPClassifier(random_state=RNG, max_iter=800),
    "LightGBM": LGBMClassifier(random_state=RNG, verbose=-1),
    "CatBoost": CatBoostClassifier(random_state=RNG, verbose=0),
}
# 3-feature uses top-3 corr features
from scipy.stats import pointbiserialr
cors = {c: abs(pointbiserialr(ytr, XtrS[:, i])[0]) for i, c in enumerate(featsS)}
top3 = [featsS.index(k) for k, _ in sorted(cors.items(), key=lambda z: -z[1])[:3]]

test = {}; probs = {}
for nm, mdl in models_std.items():
    if nm == "3-feature LR":
        mdl.fit(XtrS[:, top3], ytr); p = mdl.predict_proba(XteS[:, top3])[:, 1]
    else:
        mdl.fit(XtrS, ytr); p = mdl.predict_proba(XteS)[:, 1]
    probs[nm] = p; test[nm] = evalp(yte, p)
st = StackingClassifier(
    [("rf", RandomForestClassifier(random_state=RNG, **BAL)),
     ("xgb", XGBClassifier(random_state=RNG, eval_metric='logloss')),
     ("lgbm", LGBMClassifier(random_state=RNG, verbose=-1))],
    final_estimator=LogisticRegression(max_iter=1000), cv=5).fit(XtrS, ytr)
probs["Stacked ens."] = st.predict_proba(XteS)[:, 1]; test["Stacked ens."] = evalp(yte, probs["Stacked ens."])

prop = proposed().fit(XtrA, ytr)
p_prop = prop.predict_proba(XteA)[:, 1]
probs["CGB-CLR (ours)"] = p_prop; test["CGB-CLR (ours)"] = evalp(yte, p_prop)

# repeated 5x5 CV -- PRIMARY metric
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
XallS = StandardScaler().fit_transform(Xstd); XallA = StandardScaler().fit_transform(Xaug)
cv = {}
for nm, mdl in [("Random Forest", RandomForestClassifier(random_state=RNG, **BAL)),
                ("XGBoost", XGBClassifier(random_state=RNG, eval_metric='logloss')),
                ("LightGBM", LGBMClassifier(random_state=RNG, verbose=-1)),
                ("CatBoost", CatBoostClassifier(random_state=RNG, verbose=0))]:
    s = cross_val_score(mdl, XallS, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
    cv[nm] = dict(mean=float(s.mean()), std=float(s.std()),
                  lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), raw=s.tolist())
s = cross_val_score(proposed(), XallA, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
cv["CGB-CLR (ours)"] = dict(mean=float(s.mean()), std=float(s.std()),
                            lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), raw=s.tolist())

# paired t-test on CV folds: proposed vs each baseline
paired = {}
pr = np.array(cv["CGB-CLR (ours)"]["raw"])
for nm in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
    bl = np.array(cv[nm]["raw"])
    t, pv = stats.ttest_rel(pr, bl)
    paired[nm] = dict(delta=float(pr.mean() - bl.mean()), p=float(pv))

json.dump({"test": test, "cv": cv, "paired": paired}, open("final.json", "w"), indent=2, default=float)
print("== repeated 5x5 CV AUC (PRIMARY) ==")
for nm, c in sorted(cv.items(), key=lambda kv: -kv[1]['mean']):
    star = " <-- proposed" if "ours" in nm else ""
    print(f"  {nm:16s} {c['mean']:.3f}±{c['std']:.3f} [{c['lo']:.3f},{c['hi']:.3f}]{star}")
print("== paired t-test proposed vs baseline (CV folds) ==")
for nm, d in paired.items():
    print(f"  vs {nm:14s} Δ={d['delta']:+.3f} p={d['p']:.4f}")
print("== held-out test (reference) ==")
for nm, m in sorted(test.items(), key=lambda kv: -kv[1]['auc']):
    star = " <-- proposed" if "ours" in nm else ""
    print(f"  {nm:16s} auc={m['auc']:.3f} [{m['lo']:.3f},{m['hi']:.3f}] acc={m['acc']:.3f} "
          f"rec={m['rec']:.3f} spec={m['spec']:.3f} brier={m['brier']:.3f}{star}")
