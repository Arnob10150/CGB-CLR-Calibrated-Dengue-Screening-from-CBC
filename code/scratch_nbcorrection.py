#!/usr/bin/env python3
"""Rerun the definitive CV (identical to dengue_results_final.py, same seed/config)
but additionally save raw per-fold scores and compute the Nadeau-Bengio (2003)
corrected resampled paired t-test against every baseline, to replace the
uncorrected paired t-test claim in the paper with an honest, actually-computed one.
"""
import warnings, sys, io; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, json
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
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

df = load(); Xstd = feats_std(df); Xrich = feats_rich(df, Xstd); y = df['y']
featsS = Xstd.columns.tolist()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=RNG)
ytr = y.values[idx_tr]
scS = StandardScaler().fit(Xstd.iloc[idx_tr])
XtrS = scS.transform(Xstd.iloc[idx_tr])
cors = {c: abs(pointbiserialr(ytr, XtrS[:, i])[0]) for i, c in enumerate(featsS)}
top3 = [featsS.index(k) for k, _ in sorted(cors.items(), key=lambda z: -z[1])[:3]]

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
XallS = StandardScaler().fit_transform(Xstd); XallR = StandardScaler().fit_transform(Xrich)
cvmodels = {"Random Forest": (RandomForestClassifier(random_state=RNG, **BAL), XallS),
            "XGBoost": (XGBClassifier(random_state=RNG, eval_metric='logloss'), XallS),
            "LightGBM": (LGBMClassifier(random_state=RNG, verbose=-1), XallS),
            "CatBoost": (CatBoostClassifier(random_state=RNG, verbose=0), XallS),
            "RBF-SVM": (SVC(probability=True, random_state=RNG, **BAL), XallS),
            "Full LR": (LogisticRegression(max_iter=1000, **BAL), XallS),
            "3-feature LR": (LogisticRegression(max_iter=1000, **BAL), XallS[:, top3]),
            "MLP": (MLPClassifier(random_state=RNG, max_iter=800), XallS)}

cvraw = {}
for nm, (mdl, X) in cvmodels.items():
    s = cross_val_score(mdl, X, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
    cvraw[nm] = s
    print(f"{nm:16s} mean={s.mean():.4f} std={s.std():.4f}")
s = cross_val_score(proposed(), XallR, y, cv=rcv, scoring='roc_auc', n_jobs=-1)
cvraw["CGB-CLR (ours)"] = s
print(f"{'CGB-CLR (ours)':16s} mean={s.mean():.4f} std={s.std():.4f}")


def nadeau_bengio_ttest(diffs, n_train, n_test, n_repeats=5, n_splits=5):
    """Nadeau & Bengio (2003) corrected resampled t-test for k-fold CV repeated
    r times. diffs = per-fold score differences (length r*k). Corrects the
    variance estimate for the correlation induced by overlapping train/test
    sets across repeats, using the train/test-ratio correction factor.
    """
    n = len(diffs)
    mean_diff = diffs.mean()
    var_diff = diffs.var(ddof=1)
    correction = (1.0 / n) + (n_test / n_train)
    denom = np.sqrt(correction * var_diff)
    if denom == 0:
        return mean_diff, 0.0, 1.0
    t_stat = mean_diff / denom
    dof = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    return mean_diff, t_stat, p_val


# fold sizes for 5x5 stratified CV on n=1505: train ~4/5, test ~1/5
n_total = len(df)
n_test_fold = n_total // 5
n_train_fold = n_total - n_test_fold

pr = cvraw["CGB-CLR (ours)"]
print("\n== Nadeau-Bengio corrected paired t-test: CGB-CLR vs each baseline ==")
results = {}
for nm in ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "RBF-SVM", "MLP", "Full LR", "3-feature LR"]:
    bl = cvraw[nm]
    diffs = pr - bl
    # uncorrected (naive) paired t-test, as currently reported in the paper
    t_naive, p_naive = stats.ttest_rel(pr, bl)
    # corrected
    mean_d, t_corr, p_corr = nadeau_bengio_ttest(diffs, n_train_fold, n_test_fold)
    results[nm] = dict(delta=float(mean_d), naive_p=float(p_naive), corrected_p=float(p_corr))
    print(f"  vs {nm:14s} delta={mean_d:+.4f}  naive_p={p_naive:.4f}  NB_corrected_p={p_corr:.4f}")

out = {
    "cv_raw": {k: v.tolist() for k, v in cvraw.items()},
    "nadeau_bengio": results,
    "n_train_fold": int(n_train_fold), "n_test_fold": int(n_test_fold),
}
json.dump(out, open("nb_correction_results.json", "w"), indent=2)
print("\nSaved nb_correction_results.json")
