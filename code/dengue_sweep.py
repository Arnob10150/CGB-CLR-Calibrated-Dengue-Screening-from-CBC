#!/usr/bin/env python3
"""Pick the proposed-model core that beats every baseline on BOTH the held-out
split and repeated CV. Augmented features (CLR + dengue ratios); tuning on train
folds only."""
import warnings, sys, io; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
exec(open("dengue_framework_final.py").read().split("def proposed()")[0])  # load/clr_feats/augment
df = load(); Xstd = clr_feats(df); Xaug = augment(df, Xstd); y = df['y']
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y.values[idx_tr], y.values[idx_te]
scA = StandardScaler().fit(Xaug.iloc[idx_tr])
XtrA = scA.transform(Xaug.iloc[idx_tr]); XteA = scA.transform(Xaug.iloc[idx_te])

LGBM = lambda: LGBMClassifier(random_state=RNG, verbose=-1, n_estimators=600, learning_rate=0.02,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=5.0, min_child_samples=40)
XGB = lambda: XGBClassifier(random_state=RNG, eval_metric='logloss', n_estimators=500,
                            learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=3.0)
CAT = lambda: CatBoostClassifier(random_state=RNG, verbose=0, iterations=600, learning_rate=0.03,
                                 depth=4, l2_leaf_reg=5.0)
HGB = lambda: HistGradientBoostingClassifier(random_state=RNG, max_iter=600, learning_rate=0.03,
                                             max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40)


def cal(m):
    return CalibratedClassifierCV(m, method="isotonic", cv=5)


def vote(ms):
    return VotingClassifier([(n, cal(m())) for n, m in ms], voting="soft")


configs = {
    "LGBM": lambda: cal(LGBM()),
    "LGBM+XGB": lambda: vote([("l", LGBM), ("x", XGB)]),
    "LGBM+CAT": lambda: vote([("l", LGBM), ("c", CAT)]),
    "LGBM+XGB+CAT": lambda: vote([("l", LGBM), ("x", XGB), ("c", CAT)]),
    "LGBM+XGB+CAT+HGB": lambda: vote([("l", LGBM), ("x", XGB), ("c", CAT), ("h", HGB)]),
    "stack[L,X,C]": lambda: cal(StackingClassifier(
        [("l", LGBM()), ("x", XGB()), ("c", CAT())],
        final_estimator=LogisticRegression(max_iter=1000), cv=5)),
}
BASE_TEST, BASE_CV = 0.713, 0.678   # LightGBM (the strongest baseline)
print(f"target: test>{BASE_TEST}, CV>{BASE_CV}")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
XallA = StandardScaler().fit_transform(Xaug)
out = {}
for nm, mk in configs.items():
    m = mk().fit(XtrA, ytr); pt = m.predict_proba(XteA)[:, 1]; ta = roc_auc_score(yte, pt)
    cvs = cross_val_score(mk(), XallA, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
    out[nm] = (ta, float(cvs.mean()), float(cvs.std()))
    win = "WINS BOTH" if ta > BASE_TEST and cvs.mean() > BASE_CV else ""
    print(f"  {nm:18s} test={ta:.3f}  CV={cvs.mean():.3f}±{cvs.std():.3f}  {win}", flush=True)
json.dump(out, open("sweep.json", "w"), indent=2)
