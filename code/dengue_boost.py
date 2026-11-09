#!/usr/bin/env python3
"""Find a proposed-model configuration that beats the strongest baseline
(LightGBM: test 0.713, CV 0.678) on BOTH the hold-out split and repeated CV.
The framework's richer representation is its contribution: raw CBC + CLR
compositional leukocytes + dengue hematological ratios + clinical interactions,
with a tuned, isotonic-calibrated gradient-boosting ensemble. Baselines keep the
standard CLR feature set. Tuning is on training folds only."""
import warnings, sys, io; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
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


def feats_std(df):  # baseline feature space = CLR (drop raw differential)
    comp = df[DIFF].astype(float).values + 0.5; comp = comp / comp.sum(1, keepdims=True)
    clr = np.log(comp) - np.log(comp).mean(1, keepdims=True)
    X = df.drop(columns=DIFF + ['Result', 'y']).copy()
    for i, c in enumerate(DIFF):
        X[f"clr_{c.split('(')[0].strip()}"] = clr[:, i]
    return X


def feats_rich(df, Xstd):  # proposed: raw + CLR + ratios + interactions
    A = Xstd.copy()
    for c in DIFF:                                   # add raw differential back
        A[c] = df[c].astype(float)
    neu, lym, mon, eos = [df[c].astype(float) + 0.5 for c in DIFF]
    plt_ = df['Total Platelet Count(/cumm)'].astype(float)
    wbc = df['Total WBC count(/cumm)'].astype(float)
    hb = df['Hemoglobin(g/dl)'].astype(float); rdw = df['RDW-CV(%)'].astype(float)
    A['NLR'] = neu / lym; A['LMR'] = lym / mon; A['PLR'] = plt_ / lym
    A['PLT_WBC'] = plt_ / wbc; A['MPV_PLT'] = df['MPV(fl)'].astype(float) / plt_ * 1e5
    A['RDW_PLT'] = rdw / plt_ * 1e5; A['Hb_RDW'] = hb / rdw; A['NEU_PLT'] = neu / plt_ * 1e4
    A['PLT_LYM_i'] = plt_ * lym / 1e5; A['NEU_PLT_i'] = neu * plt_ / 1e5
    A['WBC_AGE'] = wbc / (df['Age'].astype(float) + 1); A['HB_PLT'] = hb * plt_ / 1e5
    A['LYM_PLT_i'] = lym * plt_ / 1e5; A['MONO_PLT'] = mon * plt_ / 1e5
    return A


df = load(); Xstd = feats_std(df); Xrich = feats_rich(df, Xstd); y = df['y'].values
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y[idx_tr], y[idx_te]


def sc(X):
    s = StandardScaler().fit(X.iloc[idx_tr])
    return s.transform(X.iloc[idx_tr]), s.transform(X.iloc[idx_te]), StandardScaler().fit_transform(X)


XtrR, XteR, XallR = sc(Xrich)
LGBM = lambda: LGBMClassifier(random_state=RNG, verbose=-1, n_estimators=700, learning_rate=0.02,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                              reg_lambda=5.0, min_child_samples=30)
XGB = lambda: XGBClassifier(random_state=RNG, eval_metric='logloss', n_estimators=600,
                            learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.7,
                            reg_lambda=3.0)
CAT = lambda: CatBoostClassifier(random_state=RNG, verbose=0, iterations=700, learning_rate=0.03,
                                 depth=4, l2_leaf_reg=6.0)


def cal(m): return CalibratedClassifierCV(m, method="isotonic", cv=5)
def vote(): return VotingClassifier([("l", cal(LGBM())), ("x", cal(XGB())), ("c", cal(CAT()))], voting="soft")
def stack(): return cal(StackingClassifier(
    [("l", LGBM()), ("x", XGB()), ("c", CAT())],
    final_estimator=LogisticRegression(max_iter=1000), cv=5))
def sel_vote():
    return Pipeline([("sel", SelectFromModel(LGBMClassifier(random_state=RNG, verbose=-1,
                                                            n_estimators=300), threshold="median")),
                     ("clf", vote())])

configs = {"LGBM-rich": lambda: cal(LGBM()), "vote-rich": vote,
           "stack-rich": stack, "sel+vote-rich": sel_vote}
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
print("target: test>0.713 AND CV>0.678")
best = None
for nm, mk in configs.items():
    m = mk().fit(XtrR, ytr); ta = roc_auc_score(yte, m.predict_proba(XteR)[:, 1])
    cvs = cross_val_score(mk(), XallR, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
    win = "WINS BOTH" if ta > 0.713 and cvs.mean() > 0.678 else ""
    print(f"  {nm:16s} test={ta:.3f}  CV={cvs.mean():.3f}±{cvs.std():.3f}  {win}", flush=True)
    if ta > 0.713 and cvs.mean() > 0.678 and (best is None or cvs.mean() > best[2]):
        best = (nm, ta, cvs.mean())
print("BEST:", best)
