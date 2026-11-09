#!/usr/bin/env python3
"""External geographic validation (R1): train on Jamalpur, test on the
independent Kalai/Joypurhat cohort. The external cohort records the leukocyte
differential and red-cell panel only as binary flags, so external validation
uses the shared quantitative CBC panel (Age, Hemoglobin, WBC, Platelet, PDW) plus
the platelet-to-WBC ratio. The full CLR model is validated internally; this tests
whether a calibrated CBC screener transfers across hospitals."""
import warnings, zipfile, io; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix
from lightgbm import LGBMClassifier

RNG = 42; np.random.seed(RNG)
JAM = ("dengue_data/Dengue Fever Hematological Dataset Clinical Insights for "
       "Improved Diagnosis and Patient Management/Dengue Fever Hematological Dataset.csv")
SHARED = ["Age", "Hemoglobin(g/dl)", "Total WBC count(/cumm)",
          "Total Platelet Count(/cumm)", "PDW(%)"]


def load_jam():
    df = pd.read_csv(JAM)
    df['y'] = (df['Result'] == 'positive').astype(int)
    key = df.drop(columns=['Result', 'y']).astype(str).agg('|'.join, axis=1)
    conflict = key.groupby(key).transform(lambda z: df.loc[z.index, 'y'].nunique() > 1)
    df = df[~conflict].drop_duplicates(subset=[c for c in df.columns if c not in ('Result',)],
                                       keep='first').reset_index(drop=True)
    X = df[SHARED].astype(float).copy()
    X['PLT_WBC'] = df['Total Platelet Count(/cumm)'] / df['Total WBC count(/cumm)']
    return X, df['y'].values


def load_kalai():
    z = zipfile.ZipFile("Predictive Clinical Dataset for Dengue Fever Using Vital Signs and Blood Parameters.zip")
    n = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
    d = pd.read_csv(z.open(n))
    d = d.dropna(subset=["Age", "Haemoglobin", "WBC Count", "Platelet Count", "PDW",
                         "Final Output"]).reset_index(drop=True)
    X = pd.DataFrame({
        "Age": d["Age"].astype(float),
        "Hemoglobin(g/dl)": d["Haemoglobin"].astype(float),
        "Total WBC count(/cumm)": d["WBC Count"].astype(float),
        "Total Platelet Count(/cumm)": d["Platelet Count"].astype(float),
        "PDW(%)": d["PDW"].astype(float)})
    X['PLT_WBC'] = X['Total Platelet Count(/cumm)'] / X['Total WBC count(/cumm)']
    y = d["Final Output"].astype(int).values
    return X[SHARED + ['PLT_WBC']], y


Xj, yj = load_jam(); Xk, yk = load_kalai()
print(f"Jamalpur (train) n={len(Xj)} pos={yj.mean():.3f} | Kalai (external) n={len(Xk)} pos={yk.mean():.3f}")
sc = StandardScaler().fit(Xj)
Xjs, Xks = sc.transform(Xj), sc.transform(Xk)


def report(y, p):
    pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(auc=float(roc_auc_score(y, p)), acc=float(accuracy_score(y, pred)),
                rec=float(tp / (tp + fn)), spec=float(tn / (tn + fp)),
                brier=float(brier_score_loss(y, p)),
                npv=float(tn / (tn + fn)) if (tn + fn) else float('nan'))


R = {"n_train": int(len(Xj)), "n_ext": int(len(Xk)),
     "train_pos": float(yj.mean()), "ext_pos": float(yk.mean())}
# proposed framework core (calibrated GB) and the strongest baseline (LightGBM)
prop = CalibratedClassifierCV(GradientBoostingClassifier(random_state=RNG),
                              method="isotonic", cv=5).fit(Xjs, yj)
lgbm = LGBMClassifier(random_state=RNG, verbose=-1).fit(Xjs, yj)
# internal (Jamalpur 5-fold OOF-ish: just resubstitution AUC is optimistic; use train AUC note)
R["proposed_external"] = report(yk, prop.predict_proba(Xks)[:, 1])
R["lightgbm_external"] = report(yk, lgbm.predict_proba(Xks)[:, 1])
R["proposed_internal_train"] = report(yj, prop.predict_proba(Xjs)[:, 1])
json.dump(R, open("external.json", "w"), indent=2)
print("== external validation (train Jamalpur -> test Kalai), shared 6-feature panel ==")
for k in ("proposed_external", "lightgbm_external"):
    m = R[k]
    print(f"  {k:20s} AUC={m['auc']:.3f} acc={m['acc']:.3f} sens={m['rec']:.3f} "
          f"spec={m['spec']:.3f} NPV={m['npv']:.3f} Brier={m['brier']:.3f}")
