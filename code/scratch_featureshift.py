#!/usr/bin/env python3
"""Quantify feature shift (R2 point 3) honestly: evaluate the SAME calibrated
GB core on the SAME held-out Jamalpur split, but restricted to the 6 features
shared with the external cohort, so the internal and external numbers are an
apples-to-apples comparison on identical features. This isolates how much of
the AUC gap from internal (full CLR) to external is attributable to losing the
CLR leukocyte differential + ratios, vs. true cross-hospital shift.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix

RNG = 42; np.random.seed(RNG)
CSV = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
SHARED = ["Age", "Hemoglobin(g/dl)", "Total WBC count(/cumm)",
          "Total Platelet Count(/cumm)", "PDW(%)"]


def load():
    df = pd.read_csv(CSV)
    df['y'] = (df['Result'] == 'positive').astype(int)
    key = df.drop(columns=['Result', 'y']).astype(str).agg('|'.join, axis=1)
    conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
    df = df[~conflict].drop_duplicates(subset=[c for c in df.columns if c != 'Result'],
                                       keep='first').reset_index(drop=True)
    X = df[SHARED].astype(float).copy()
    X['PLT_WBC'] = df['Total Platelet Count(/cumm)'] / df['Total WBC count(/cumm)']
    return X, df['y'].values


X6, y = load()
idx_tr, idx_te = train_test_split(np.arange(len(X6)), test_size=0.25, stratify=y, random_state=RNG)
ytr, yte = y[idx_tr], y[idx_te]
sc = StandardScaler().fit(X6.iloc[idx_tr])
Xtr, Xte = sc.transform(X6.iloc[idx_tr]), sc.transform(X6.iloc[idx_te])

prop = CalibratedClassifierCV(GradientBoostingClassifier(random_state=RNG), method="isotonic", cv=5)
prop.fit(Xtr, ytr)
p = prop.predict_proba(Xte)[:, 1]
pred = (p >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
result = dict(
    auc=float(roc_auc_score(yte, p)), acc=float(accuracy_score(yte, pred)),
    sens=float(tp / (tp + fn)), spec=float(tn / (tn + fp)),
    brier=float(brier_score_loss(yte, p)),
    npv=float(tn / (tn + fn)) if (tn + fn) else float('nan'))
print("== internal held-out AUC, 6-feature (external-parity) subset ==")
print(f"  AUC={result['auc']:.4f} acc={result['acc']:.4f} sens={result['sens']:.4f} "
      f"spec={result['spec']:.4f} NPV={result['npv']:.4f} Brier={result['brier']:.4f}")
json.dump(result, open("feature_shift_results.json", "w"), indent=2)
print("Saved feature_shift_results.json")
