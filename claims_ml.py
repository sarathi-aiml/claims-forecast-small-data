#!/usr/bin/env python3
"""
claims_ml.py - healthcare claims prediction + weekly forecasting for SMALL datasets (~10k rows).

Everything is local: CSV in, CSV out, one folder, no database.

  python claims_ml.py generate                -> data/claims.csv (SYNTHETIC, 10k rows; no real patients)
  python claims_ml.py train [--fast]          -> model_paid.joblib, model_denial.joblib, metrics_*.csv
  python claims_ml.py forecast                -> forecast_metrics.csv, forecast_backtest.csv, forecast_next.csv
  python claims_ml.py predict FILE.csv        -> predictions.csv
  python claims_ml.py update FILE.csv         -> append a new week, score last week's forecast, drift check, retrain
  python claims_ml.py simulate [--weeks 8]    -> learning_log_simulation.csv (replays the weekly loop on history)

Two claim-level models (a "hurdle" pair) + one weekly forecaster:
  1. model_denial  : P(claim is denied)                     - classifier
  2. model_paid    : paid amount given the claim is approved - regressor (log target)
     expected_paid = (1 - P(denied)) * paid_if_approved
  3. weekly forecast of total paid $ and claim count, N weeks ahead, with prediction intervals
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")   # also silences joblib worker processes
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             explained_variance_score, log_loss, matthews_corrcoef, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, RepeatedKFold, cross_val_predict,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG - the knobs a customer is expected to tune. Everything else is derived.
# =============================================================================
SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "claims.csv"   # SYNTHETIC data by default; replace with your own export

HOLDOUT_WEEKS = 12          # last N weeks of claims are held out = honest "future" test
CV_FOLDS, CV_REPEATS = 5, 2  # repeated K-fold: stable estimates on small data
SEARCH_ITER = 20            # random-search budget for the boosted model (0 = skip)
PI_LEVEL = 0.90             # prediction-interval level (0.90 = 90% of actuals should fall inside)
DENIAL_THRESHOLD = None     # None = choose the threshold that maximises F1 on cross-validated probabilities
SIMPLE_TOLERANCE = 0.02     # prefer a single boosted model if its CV MAE is within 2% of the best (smaller file, faster, easier to explain)

HORIZON = 4                 # weeks ahead for the weekly forecast
BACKTEST_ORIGINS = 12       # rolling-origin backtest windows (more = more reliable metrics, slower)
DRIFT_REF_WEEKS = 8         # drift check compares the new week against this many trailing weeks
AS_OF = None                # date the data is complete through, e.g. "2025-12-28" (or --as-of on the CLI);
                            # None = end of the week containing the last service date
MAX_CATEGORY_LEVELS = 200   # group rarer levels into "Other" before loading; the boosted model caps at 255

TARGET_REG, TARGET_CLF = "paid_amount", "is_denied"

# raw columns the customer's CSV must contain (see README "Data dictionary")
RAW_NUM = ["age", "tenure_months", "chronic_conditions", "prior_claims_12m", "prior_paid_12m",
           "length_of_stay", "units", "days_to_submit", "deductible_remaining", "billed_amount",
           "in_network", "is_emergency", "prior_auth"]
CAT = ["gender", "plan_type", "region", "provider_specialty", "place_of_service",
       "diagnosis_group", "procedure_group"]
# derived in make_features()
DERIVED = ["log_billed", "billed_per_unit", "prior_paid_per_claim", "deductible_ratio",
           "month", "week_of_year", "day_of_week"]
NUM = RAW_NUM + DERIVED
FEATURES = NUM + CAT

# Domain priors as monotonic constraints (+1 = paid can only go up with the feature, -1 = down).
# This is the single most useful small-data trick: it stops the tree model from learning noise
# that contradicts how claims actually get paid.
# (units is deliberately NOT constrained: with billed fixed, more units means a cheaper unit, not a bigger payment)
MONO = {"billed_amount": 1, "log_billed": 1, "billed_per_unit": 1, "length_of_stay": 1,
        "deductible_remaining": -1, "deductible_ratio": -1}


# =============================================================================
# 1. SYNTHETIC DATA
# =============================================================================
PLACES = ["Office", "Outpatient", "Inpatient", "ER", "Telehealth", "Lab", "Pharmacy"]
PLACE_P = [.38, .17, .06, .09, .12, .10, .08]
PLACE_MULT = {"Office": 1.0, "Outpatient": 1.8, "Inpatient": 3.5, "ER": 2.6, "Telehealth": 0.6, "Lab": 1.0, "Pharmacy": 1.0}
PROCS = ["Evaluation", "Imaging", "Surgery", "Lab", "Therapy", "Drug", "DME"]
PROC_COST = {"Evaluation": 180, "Imaging": 650, "Surgery": 6500, "Lab": 140, "Therapy": 220, "Drug": 320, "DME": 450}
PROC_BY_PLACE = {  # which procedures happen where
    "Office":     [.55, .10, .03, .12, .12, .05, .03],
    "Outpatient": [.30, .25, .20, .10, .10, .03, .02],
    "Inpatient":  [.30, .15, .45, .05, .03, .02, .00],
    "ER":         [.45, .35, .08, .10, .00, .02, .00],
    "Telehealth": [.85, .00, .00, .00, .12, .03, .00],
    "Lab":        [.02, .00, .00, .96, .00, .02, .00],
    "Pharmacy":   [.00, .00, .00, .00, .00, .95, .05],
}
DIAGS = ["Circulatory", "Respiratory", "Musculoskeletal", "Injury", "Neoplasm", "Endocrine",
         "MentalHealth", "Digestive", "Pregnancy", "Other"]
DIAG_P = [.14, .10, .15, .10, .06, .11, .09, .08, .04, .13]
SPECIALTIES = ["PrimaryCare", "Cardiology", "Orthopedics", "Oncology", "Radiology", "Emergency",
               "Endocrinology", "Psychiatry", "Gastroenterology", "Pathology"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
DEDUCTIBLE = {"HMO": 500, "PPO": 1000, "HDHP": 3000, "MedicareAdv": 250}
COINSURANCE = {"HMO": .85, "PPO": .80, "HDHP": .75, "MedicareAdv": .90}


def generate(n: int = 10_000, start: str = "2024-01-01", weeks: int = 104, out: Path = DATA) -> pd.DataFrame:
    """SYNTHETIC claims. Every row is generated from the random process below; there are no real
    members, providers or claims anywhere in this repository. Structure is realistic: seasonality,
    trend, holiday dips, deductible resets in January, slow contract drift, ~12% denials, missing values."""
    rng = np.random.default_rng(SEED)
    wk = np.arange(weeks)
    season = 1 + 0.18 * np.cos(2 * np.pi * (wk - 4) / 52)        # winter peak (flu), summer trough
    trend = 1 + 0.08 * wk / 52                                   # +8%/yr volume growth
    holiday = np.where(np.isin(wk % 52, [51, 0]), 0.75, 1.0)     # Christmas / New Year dip
    w = season * trend * holiday * rng.normal(1, 0.06, weeks).clip(0.7, 1.3)
    week_of = rng.choice(weeks, n, p=w / w.sum())
    service_date = pd.Timestamp(start) + pd.to_timedelta(week_of * 7 + rng.integers(0, 7, n), unit="D")
    years_in = week_of / 52

    # --- member
    age = rng.normal(46, 19, n).clip(1, 95).round().astype(int)
    plan = np.where(age >= 65, rng.choice(["MedicareAdv", "PPO", "HMO"], n, p=[.7, .2, .1]),
                    rng.choice(["HMO", "PPO", "HDHP"], n, p=[.42, .38, .20]))
    gender = rng.choice(["F", "M"], n, p=[.52, .48])
    region = rng.choice(REGIONS, n, p=[.20, .25, .22, .13, .20])
    tenure = rng.integers(1, 120, n).astype(float)
    chronic = rng.poisson(0.3 + age / 45, n).clip(0, 6)
    prior_claims = rng.poisson(1.2 + 1.3 * chronic, n)
    prior_paid = (prior_claims * rng.lognormal(5.8, 0.9, n)).round(2)

    # --- claim
    place = rng.choice(PLACES, n, p=PLACE_P)
    proc = np.array([rng.choice(PROCS, p=PROC_BY_PLACE[p]) for p in place])
    diag = rng.choice(DIAGS, n, p=DIAG_P)
    winter = rng.random(n) < 0.12 * (season[week_of] - 1).clip(0) / 0.18   # respiratory spikes in winter
    diag = np.where(winter, "Respiratory", diag)
    specialty = rng.choice(SPECIALTIES, n)
    in_network = rng.random(n) < np.where(plan == "HMO", 0.95, 0.86)
    is_emergency = ((place == "ER") & (rng.random(n) < 0.7)) | ((place == "Inpatient") & (rng.random(n) < 0.3))
    los = np.where(place == "Inpatient", 1 + rng.poisson(2.5, n), 0)
    units = 1 + rng.poisson(0.6, n)
    needs_auth = np.isin(proc, ["Surgery", "Imaging"]) | (place == "Inpatient")
    prior_auth = np.where(needs_auth, rng.random(n) < 0.85, True)
    days_to_submit = rng.gamma(2, 12, n).round().clip(0, 180).astype(int)

    burn = rng.uniform(1.5, 4.0, n) * service_date.dayofyear.values / 365 * (1 + 0.5 * chronic)
    ded_rem = (np.vectorize(DEDUCTIBLE.get)(plan) * np.clip(1 - burn, 0, 1)).round(2)

    base = np.vectorize(PROC_COST.get)(proc) * np.vectorize(PLACE_MULT.get)(place)
    billed = (base * units * (1 + 0.6 * los) * (1 + 0.06 * chronic) * np.where(in_network, 1, 1.3)
              * rng.lognormal(0, 0.40, n)).clip(20, 120_000).round(2)

    # --- denial (classification target)
    z = (np.log(billed) - np.log(billed).mean()) / np.log(billed).std()
    logit = (-3.7 + 2.0 * (~in_network) + 2.4 * (days_to_submit > 60) + 2.6 * (needs_auth & ~prior_auth)
             + 1.0 * ((place == "ER") & ~is_emergency) + 0.5 * z + 0.4 * (prior_claims > 6)
             - 0.4 * (plan == "MedicareAdv") + 0.025 * days_to_submit + 0.6 * (proc == "DME")
             + 0.5 * (specialty == "Psychiatry") + rng.normal(0, 0.5, n))
    denied = rng.random(n) < 1 / (1 + np.exp(-logit))

    # --- paid (regression target); slow drift in allowed ratio = "contracts get renegotiated"
    allowed = billed * np.where(in_network, 0.62, 0.45) * (1 + 0.06 * years_in) * rng.lognormal(0, 0.08, n)
    paid = np.maximum(allowed - ded_rem, 0) * np.vectorize(COINSURANCE.get)(plan) * rng.lognormal(0, 0.12, n)
    paid = np.where(denied, 0.0, paid).round(2)

    df = pd.DataFrame({
        "claim_id": [f"CLM{i:06d}" for i in range(1, n + 1)],
        "member_id": [f"M{m:05d}" for m in rng.integers(1, 4000, n)],
        "service_date": service_date,
        "submitted_date": service_date + pd.to_timedelta(days_to_submit, unit="D"),
        "age": age, "gender": gender, "plan_type": plan, "region": region, "tenure_months": tenure,
        "chronic_conditions": chronic, "prior_claims_12m": prior_claims, "prior_paid_12m": prior_paid,
        "provider_specialty": specialty, "place_of_service": place, "diagnosis_group": diag,
        "procedure_group": proc, "in_network": in_network, "is_emergency": is_emergency,
        "prior_auth": prior_auth.astype(object), "length_of_stay": los, "units": units,
        "days_to_submit": days_to_submit, "deductible_remaining": ded_rem, "billed_amount": billed,
        "is_denied": denied.astype(int), "paid_amount": paid,
    })
    # realistic missingness
    df.loc[rng.random(n) < 0.03, "tenure_months"] = np.nan
    df.loc[rng.random(n) < 0.02, "prior_auth"] = np.nan
    df = df.sort_values("service_date").reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(df)}  denial_rate={df.is_denied.mean():.1%}  "
          f"median_paid(approved)=${df.loc[df.is_denied == 0, 'paid_amount'].median():,.0f}")
    return df


# =============================================================================
# 2. DATA + FEATURES
# =============================================================================
def load(path: Path = DATA) -> pd.DataFrame:
    if not Path(path).exists():
        sys.exit(f"{path} not found - run: python claims_ml.py generate (synthetic) or place your export there")
    df = pd.read_csv(path)
    for c in ("service_date", "submitted_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def validate(df: pd.DataFrame, need_targets: bool) -> None:
    need = ["service_date"] + RAW_NUM + CAT + ([TARGET_REG, TARGET_CLF] if need_targets else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}\nSee README 'Data dictionary'.")
    for c in RAW_NUM:
        if c not in BOOLS and not pd.api.types.is_numeric_dtype(df[c]):
            sys.exit(f"column {c} is not numeric (found e.g. {df[c].dropna().iloc[0]!r}); strip $ and , before loading")
    for c in CAT:
        if df[c].nunique() > MAX_CATEGORY_LEVELS:
            sys.exit(f"column {c} has {df[c].nunique()} levels (max {MAX_CATEGORY_LEVELS}); group rare levels into 'Other'")


BOOLS = ("in_network", "is_emergency", "prior_auth")
_TRUE, _FALSE = {"true", "t", "1", "1.0", "yes", "y"}, {"false", "f", "0", "0.0", "no", "n"}


def to_bool(col: pd.Series) -> pd.Series:
    """Accepts True/False, 1/0, Yes/No, Y/N, T/F in any case; NaN stays NaN; anything else is an error."""
    txt = col.astype(str).str.strip().str.lower()
    out = pd.Series(np.where(txt.isin(_TRUE), 1.0, np.where(txt.isin(_FALSE), 0.0, np.nan)), index=col.index)
    bad = col.notna() & out.isna()
    if bad.any():
        sys.exit(f"column {col.name}: unrecognised boolean values {sorted(col[bad].astype(str).unique()[:5])}")
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["service_date"] = pd.to_datetime(d["service_date"])
    for b in BOOLS:
        d[b] = to_bool(d[b])
    d["log_billed"] = np.log1p(d["billed_amount"])
    d["billed_per_unit"] = d["billed_amount"] / d["units"].clip(lower=1)
    d["prior_paid_per_claim"] = d["prior_paid_12m"] / d["prior_claims_12m"].clip(lower=1)
    d["deductible_ratio"] = d["deductible_remaining"] / d["billed_amount"].clip(lower=1)
    d["month"] = d["service_date"].dt.month
    d["week_of_year"] = d["service_date"].dt.isocalendar().week.astype(int)
    d["day_of_week"] = d["service_date"].dt.dayofweek
    d["week"] = (d["service_date"] - pd.to_timedelta(d["service_date"].dt.dayofweek, unit="D")).dt.normalize()
    return d


def prep_onehot() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)), CAT),
    ])


def prep_ordinal() -> ColumnTransformer:
    # for HistGradientBoosting: native categorical support, NaN = missing, unknown category = missing
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), CAT),
    ])


CAT_MASK = [False] * len(NUM) + [True] * len(CAT)
MONO_CST = [MONO.get(c, 0) for c in NUM] + [0] * len(CAT)


def hgb_reg(**kw) -> HistGradientBoostingRegressor:
    p = dict(max_iter=500, learning_rate=0.05, max_depth=4, min_samples_leaf=25, l2_regularization=1.0,
             early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED,
             categorical_features=CAT_MASK, monotonic_cst=MONO_CST)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def clipped(pred, cap):
    """Never predict below 0 or above 2x the largest paid amount in training data."""
    return np.clip(pred, 0, cap)


def logt(est) -> TransformedTargetRegressor:
    """Fit on log1p(paid): claim amounts are heavy-tailed; this is the second small-data trick."""
    return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)


def reg_candidates() -> dict:
    return {
        "baseline_median": DummyRegressor(strategy="median"),
        "random_forest": logt(Pipeline([("prep", prep_onehot()),
                                        ("m", RandomForestRegressor(300, min_samples_leaf=4, max_features=0.5,
                                                                    random_state=SEED))])),
        "extra_trees": logt(Pipeline([("prep", prep_onehot()),
                                      ("m", ExtraTreesRegressor(300, min_samples_leaf=4, max_features=0.5,
                                                                random_state=SEED))])),
        "hist_gbm": logt(Pipeline([("prep", prep_ordinal()), ("m", hgb_reg())])),
    }


def clf_candidates() -> dict:
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic": Pipeline([("prep", prep_onehot()), ("scale", StandardScaler(with_mean=False)),
                              ("m", LogisticRegression(C=0.5, max_iter=3000))]),
        "random_forest": Pipeline([("prep", prep_onehot()),
                                   ("m", RandomForestClassifier(400, min_samples_leaf=5, max_features=0.4,
                                                                random_state=SEED))]),
        "hist_gbm": Pipeline([("prep", prep_ordinal()),
                              ("m", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                                                   min_samples_leaf=25, l2_regularization=1.0,
                                                                   early_stopping=True,
                                                                   categorical_features=CAT_MASK,
                                                                   random_state=SEED))]),
    }


# =============================================================================
# 3. METRICS
# =============================================================================
def _bootstrap_mae(y, p, B=500):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), (B, len(y)))
    vals = np.abs(p[idx] - y[idx]).mean(axis=1)
    return np.percentile(vals, [2.5, 97.5])


def reg_metrics(y, p, lo=None, hi=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e, ae, eps = p - y, np.abs(p - y), 1e-9
    big = y >= 50
    ci = _bootstrap_mae(y, p)
    m = {
        "n": len(y),
        "MAE": ae.mean(), "MAE_CI95_low": ci[0], "MAE_CI95_high": ci[1],
        "RMSE": np.sqrt((e ** 2).mean()), "MedianAE": np.median(ae), "MaxAE": ae.max(),
        "WAPE": ae.sum() / max(np.abs(y).sum(), eps),
        "sMAPE": np.mean(2 * ae / (np.abs(y) + np.abs(p) + eps)),
        "MAPE_y>=50": np.mean(ae[big] / y[big]) if big.any() else np.nan,
        "R2": r2_score(y, p), "ExplainedVar": explained_variance_score(y, p),
        "Bias": e.mean(),
        "RMSLE": np.sqrt(np.mean((np.log1p(p.clip(0)) - np.log1p(y.clip(0))) ** 2)),
        "Within10pct": np.mean(ae <= 0.10 * np.abs(y) + 1),
    }
    if lo is not None:
        m["PI_coverage"] = np.mean((y >= lo) & (y <= hi))
        m["PI_median_width"] = np.median(hi - lo)
    return m


def _ece(y, prob, bins=10):
    b = np.minimum((prob * bins).astype(int), bins - 1)
    return sum(abs(y[b == i].mean() - prob[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def clf_metrics(y, prob, thr) -> dict:
    y, prob = np.asarray(y, int), np.asarray(prob, float)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "n": len(y), "prevalence": y.mean(), "threshold": thr,
        "accuracy": (tp + tn) / len(y), "balanced_accuracy": (rec + spec) / 2,
        "precision": prec, "recall": rec, "specificity": spec,
        "F1": 2 * prec * rec / max(prec + rec, 1e-9), "F2": 5 * prec * rec / max(4 * prec + rec, 1e-9),
        "MCC": matthews_corrcoef(y, pred),
        "ROC_AUC": roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan,
        "PR_AUC": average_precision_score(y, prob) if len(set(y)) > 1 else np.nan,
        "log_loss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "Brier": brier_score_loss(y, prob), "ECE": _ece(y, prob),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def best_f1_threshold(y, prob) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    f1 = []
    for t in grid:
        pred = prob >= t
        tp = (pred & (y == 1)).sum()
        p_ = tp / max(pred.sum(), 1)
        r_ = tp / max((y == 1).sum(), 1)
        f1.append(2 * p_ * r_ / max(p_ + r_, 1e-9))
    return float(grid[int(np.argmax(f1))])


def interval(pred, q):
    """Conformal-style interval: +/- q (a quantile of |log residual|) applied in log space."""
    lp = np.log1p(np.clip(pred, 0, None))
    return np.expm1(lp - q), np.expm1(lp + q)


# =============================================================================
# 4. TRAIN (claim-level models)
# =============================================================================
def train(fast: bool = False):
    df = load()
    validate(df, need_targets=True)
    df = make_features(df)
    cutoff = df["service_date"].max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    tr, ho = df[df["service_date"] < cutoff], df[df["service_date"] >= cutoff]
    print(f"train: {len(tr)} claims before {cutoff.date()} | holdout: {len(ho)} claims (last {HOLDOUT_WEEKS} weeks)")
    kf = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=1 if fast else CV_REPEATS, random_state=SEED)

    # ---------------------------------------------------------------- paid amount (approved only)
    trA, hoA = tr[tr[TARGET_CLF] == 0], ho[ho[TARGET_CLF] == 0]
    XA, yA = trA[FEATURES], trA[TARGET_REG].values
    cands, cv_mae, rows = reg_candidates(), {}, []
    cap = float(yA.max()) * 2
    print("\n[paid_amount] cross-validation (approved claims)")
    for name, est in cands.items():
        s = cross_validate(est, XA, yA, cv=cv, n_jobs=-1,
                           scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"})
        cv_mae[name] = -s["test_mae"].mean()
        rows.append({"model": name, "split": "cv", "MAE": cv_mae[name], "MAE_std": s["test_mae"].std(),
                     "RMSE": -s["test_rmse"].mean(), "R2": s["test_r2"].mean()})
        print(f"  {name:22s} MAE={cv_mae[name]:9.2f} +/- {s['test_mae'].std():6.2f}  R2={s['test_r2'].mean():.3f}")

    if not fast and SEARCH_ITER:
        grid = {"regressor__m__learning_rate": [0.02, 0.03, 0.05, 0.08],
                "regressor__m__max_depth": [3, 4, 5, 6, None],
                "regressor__m__min_samples_leaf": [10, 20, 30, 50, 80],
                "regressor__m__l2_regularization": [0, 0.5, 1, 3, 10],
                "regressor__m__max_leaf_nodes": [8, 15, 31]}
        rs = RandomizedSearchCV(cands["hist_gbm"], grid, n_iter=SEARCH_ITER, cv=KFold(4, shuffle=True, random_state=SEED),
                                scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1).fit(XA, yA)
        cands["hist_gbm_tuned"], cv_mae["hist_gbm_tuned"] = rs.best_estimator_, -rs.best_score_
        rows.append({"model": "hist_gbm_tuned", "split": "cv", "MAE": -rs.best_score_})
        print(f"  {'hist_gbm_tuned':22s} MAE={-rs.best_score_:9.2f}  params={ {k.split('__')[-1]: v for k, v in rs.best_params_.items()} }")

    top = sorted((k for k in cv_mae if k != "baseline_median"), key=cv_mae.get)[:3]
    cands["stack"] = TransformedTargetRegressor(
        regressor=StackingRegressor([(k, clone(cands[k].regressor)) for k in top], final_estimator=RidgeCV(), cv=3, n_jobs=-1),
        func=np.log1p, inverse_func=np.expm1)
    cv_mae["stack"] = -cross_val_score(cands["stack"], XA, yA, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1).mean()
    rows.append({"model": "stack", "split": "cv", "MAE": cv_mae["stack"], "stack_of": "+".join(top)})
    print(f"  {'stack(' + '+'.join(top) + ')':50s} MAE={cv_mae['stack']:9.2f}")

    best_reg = min(cv_mae, key=cv_mae.get)
    simple = [k for k in ("hist_gbm_tuned", "hist_gbm") if k in cv_mae and cv_mae[k] <= cv_mae[best_reg] * (1 + SIMPLE_TOLERANCE)]
    if simple and best_reg not in simple:
        print(f"  ({best_reg} best by {cv_mae[best_reg]:.2f}, but {simple[0]} is within {SIMPLE_TOLERANCE:.0%}: taking the simpler model)")
        best_reg = simple[0]
    print(f"\n  -> selected by CV: {best_reg}")

    # honest holdout evaluation for every candidate (fit on train only)
    print("\n[paid_amount] holdout (last %d weeks)" % HOLDOUT_WEEKS)
    fitted = {}
    for name, est in cands.items():
        fitted[name] = clone(est).fit(XA, yA)
        p = clipped(fitted[name].predict(hoA[FEATURES]), cap)
        r = {"model": name, "split": "holdout", **reg_metrics(hoA[TARGET_REG].values, p)}
        rows.append(r)
        print(f"  {name:22s} MAE={r['MAE']:9.2f}  WAPE={r['WAPE']:.3f}  R2={r['R2']:.3f}  sMAPE={r['sMAPE']:.3f}")

    # prediction interval calibrated on train CV residuals, coverage checked on holdout
    # log-target models under-predict the mean (Jensen). Duan-style smearing: scale so CV predictions sum to actuals.
    cvp = clipped(cross_val_predict(clone(cands[best_reg]), XA, yA, cv=kf, n_jobs=-1), cap)
    smear = float(yA.sum() / cvp.sum())
    cvp = clipped(cvp * smear, cap)
    q = float(np.quantile(np.abs(np.log1p(yA) - np.log1p(cvp)), PI_LEVEL))
    print(f"  smearing factor (bias correction): x{smear:.3f}")
    p_best = clipped(fitted[best_reg].predict(hoA[FEATURES]) * smear, cap)
    lo, hi = interval(p_best, q)
    final_row = {"model": best_reg + " (final)", "split": "holdout", **reg_metrics(hoA[TARGET_REG].values, p_best, lo, hi)}
    rows.append(final_row)
    print(f"  {PI_LEVEL:.0%} interval coverage on holdout: {final_row['PI_coverage']:.1%} "
          f"(median width ${final_row['PI_median_width']:,.0f})")

    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(HERE / "metrics_regression.csv", index=False)

    pi = permutation_importance(fitted[best_reg], hoA[FEATURES], hoA[TARGET_REG].values, n_repeats=5,
                                random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}
                       ).sort_values("importance_mean", ascending=False)
    imp.to_csv(HERE / "feature_importance.csv", index=False)
    print("\n  top features:", ", ".join(imp.feature.head(8)))

    # ---------------------------------------------------------------- denial (all claims)
    X, y = tr[FEATURES], tr[TARGET_CLF].values
    ccands, cv_ap, crows = clf_candidates(), {}, []
    print("\n[is_denied] cross-validation")
    for name, est in ccands.items():
        s = cross_validate(est, X, y, cv=cv, n_jobs=-1, scoring={"ap": "average_precision", "auc": "roc_auc", "ll": "neg_log_loss"})
        cv_ap[name] = s["test_ap"].mean()
        crows.append({"model": name, "split": "cv", "PR_AUC": cv_ap[name], "PR_AUC_std": s["test_ap"].std(),
                      "ROC_AUC": s["test_auc"].mean(), "log_loss": -s["test_ll"].mean()})
        print(f"  {name:22s} PR_AUC={cv_ap[name]:.3f} +/- {s['test_ap'].std():.3f}  ROC_AUC={s['test_auc'].mean():.3f}")
    best_clf = max(cv_ap, key=cv_ap.get)
    cvprob = cross_val_predict(clone(ccands[best_clf]), X, y, cv=kf, n_jobs=-1, method="predict_proba")[:, 1]
    thr = DENIAL_THRESHOLD if DENIAL_THRESHOLD is not None else best_f1_threshold(y, cvprob)
    print(f"\n  -> selected by CV: {best_clf}, threshold={thr:.2f}")
    print("\n[is_denied] holdout")
    cfitted = {}
    for name, est in ccands.items():
        cfitted[name] = clone(est).fit(X, y)
        prob = cfitted[name].predict_proba(ho[FEATURES])[:, 1]
        r = {"model": name, "split": "holdout", **clf_metrics(ho[TARGET_CLF].values, prob, thr)}
        crows.append(r)
        print(f"  {name:22s} PR_AUC={r['PR_AUC']:.3f}  ROC_AUC={r['ROC_AUC']:.3f}  F1={r['F1']:.3f}  "
              f"recall={r['recall']:.3f}  precision={r['precision']:.3f}  Brier={r['Brier']:.3f}")
    clf_df = pd.DataFrame(crows)
    clf_df.to_csv(HERE / "metrics_classification.csv", index=False)

    # ---------------------------------------------------------------- holdout predictions file
    prob_ho = cfitted[best_clf].predict_proba(ho[FEATURES])[:, 1]
    paid_ho = clipped(fitted[best_reg].predict(ho[FEATURES]) * smear, cap)
    lo_ho, hi_ho = interval(paid_ho, q)
    # the quantity the business actually consumes: expected paid over ALL claims, denials included
    exp_row = {"model": best_reg + " expected_paid (all claims)", "split": "holdout",
               **reg_metrics(ho[TARGET_REG].values, (1 - prob_ho) * paid_ho)}
    reg_df = pd.concat([reg_df, pd.DataFrame([exp_row])], ignore_index=True)
    reg_df.to_csv(HERE / "metrics_regression.csv", index=False)
    print(f"\n[expected_paid = (1-p_denied) * paid] all holdout claims: MAE={exp_row['MAE']:.2f}  WAPE={exp_row['WAPE']:.3f}  "
          f"Bias={exp_row['Bias']:.2f}  (denial uncertainty included; this is the number to quote for totals)")
    pd.DataFrame({
        "claim_id": ho.get("claim_id", pd.Series(ho.index, index=ho.index)), "service_date": ho["service_date"].dt.date,
        "actual_is_denied": ho[TARGET_CLF].values, "p_denied": prob_ho.round(4),
        "actual_paid": ho[TARGET_REG].values, "pred_paid_if_approved": paid_ho.round(2),
        "pred_paid_low": lo_ho.round(2), "pred_paid_high": hi_ho.round(2),
        "expected_paid": ((1 - prob_ho) * paid_ho).round(2),
    }).to_csv(HERE / "holdout_predictions.csv", index=False)

    # ---------------------------------------------------------------- final models: refit on ALL data
    allA = df[df[TARGET_CLF] == 0]
    final_reg = clone(cands[best_reg]).fit(allA[FEATURES], allA[TARGET_REG].values)
    cvp_all = clipped(cross_val_predict(clone(cands[best_reg]), allA[FEATURES], allA[TARGET_REG].values, cv=kf, n_jobs=-1), cap)
    smear_all = float(allA[TARGET_REG].sum() / cvp_all.sum())
    q_all = float(np.quantile(np.abs(np.log1p(allA[TARGET_REG].values) - np.log1p(clipped(cvp_all * smear_all, cap))), PI_LEVEL))
    final_clf = clone(ccands[best_clf]).fit(df[FEATURES], df[TARGET_CLF].values)
    meta = {"trained_at": datetime.now().isoformat(timespec="seconds"), "n_rows": len(df),
            "data_from": str(df.service_date.min().date()), "data_to": str(df.service_date.max().date()),
            "features": FEATURES}
    joblib.dump({"model": final_reg, "q": q_all, "cap": float(allA[TARGET_REG].max()) * 2, "smear": smear_all, "name": best_reg, "pi_level": PI_LEVEL, **meta}, HERE / "model_paid.joblib", compress=3)
    joblib.dump({"model": final_clf, "threshold": thr, "name": best_clf, **meta}, HERE / "model_denial.joblib", compress=3)
    pd.DataFrame([
        {"key": "paid_model", "value": best_reg}, {"key": "paid_holdout_MAE", "value": round(final_row["MAE"], 2)},
        {"key": "paid_holdout_WAPE", "value": round(final_row["WAPE"], 4)}, {"key": "paid_holdout_R2", "value": round(final_row["R2"], 4)},
        {"key": "paid_PI_level", "value": PI_LEVEL}, {"key": "paid_PI_coverage_holdout", "value": round(final_row["PI_coverage"], 4)},
        {"key": "paid_PI_log_halfwidth", "value": round(q_all, 4)}, {"key": "paid_smearing_factor", "value": round(smear_all, 4)},
        {"key": "expected_paid_all_claims_holdout_MAE", "value": round(exp_row["MAE"], 2)},
        {"key": "expected_paid_all_claims_holdout_WAPE", "value": round(exp_row["WAPE"], 4)},
        {"key": "denial_model", "value": best_clf}, {"key": "denial_threshold", "value": round(thr, 3)},
        {"key": "denial_holdout_PR_AUC", "value": round(clf_df[(clf_df.model == best_clf) & (clf_df.split == "holdout")].PR_AUC.iloc[0], 4)},
        {"key": "denial_holdout_ROC_AUC", "value": round(clf_df[(clf_df.model == best_clf) & (clf_df.split == "holdout")].ROC_AUC.iloc[0], 4)},
        {"key": "trained_at", "value": meta["trained_at"]}, {"key": "n_rows", "value": len(df)},
        {"key": "data_from", "value": meta["data_from"]}, {"key": "data_to", "value": meta["data_to"]},
    ]).to_csv(HERE / "model_info.csv", index=False)
    print(f"\nsaved model_paid.joblib ({best_reg}), model_denial.joblib ({best_clf}), metrics_*.csv, "
          f"feature_importance.csv, holdout_predictions.csv, model_info.csv")
    return reg_df, clf_df


# =============================================================================
# 5. WEEKLY FORECAST
# =============================================================================
def weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df if "week" in df else make_features(df)
    g = d.groupby("week").agg(claims=("paid_amount", "size"), paid=("paid_amount", "sum"),
                              denials=("is_denied", "sum"), billed=("billed_amount", "sum"))
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="7D"), fill_value=0)
    if AS_OF:
        as_of = pd.Timestamp(AS_OF)
    else:   # assume the week containing the last service date is complete (weekday-only data has no Sunday rows)
        last = d["service_date"].max()
        as_of = last + pd.Timedelta(days=6 - last.dayofweek)
    g = g[g.index + pd.Timedelta(days=6) <= as_of]        # complete weeks only; pass --as-of when the extract is partial
    g.index.name = "week_start"
    return g


LAGS, K_FOURIER, MIN_HIST = [1, 2, 3, 4, 8], 3, 13


def _feat(y, t):
    return ([t] + [f(2 * np.pi * k * t / 52) for k in range(1, K_FOURIER + 1) for f in (np.sin, np.cos)]
            + [y[t - l] for l in LAGS] + [np.mean(y[t - 4:t]), np.mean(y[t - 13:t])])


def _recursive(y, h, make_model):
    y = list(map(float, y))
    n = len(y)
    X = np.array([_feat(y, t) for t in range(MIN_HIST, n)])
    m = make_model().fit(X, y[MIN_HIST:])
    out = []
    for i in range(h):
        p = max(float(m.predict(np.array([_feat(y, n + i)]))[0]), 0.0)
        y.append(p)
        out.append(p)
    return np.array(out)


def fc_naive(y, h):
    return np.repeat(y[-1], h)


def fc_seasonal_naive(y, h):
    n = len(y)
    return np.array([y[n - 52 + i] if n >= 52 + i else y[-1] for i in range(h)])


def fc_moving_avg(y, h):
    return np.repeat(np.mean(y[-4:]), h)


def fc_holt(y, h):
    from statsmodels.tsa.holtwinters import Holt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels re-enables its own ConvergenceWarning on import
        fit = Holt(np.asarray(y, float), damped_trend=True, initialization_method="estimated").fit()
    return np.clip(fit.forecast(h), 0, None)


def fc_ridge_fourier(y, h):
    return _recursive(y, h, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))


def fc_hgb_lags(y, h):
    return _recursive(y, h, lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                                  min_samples_leaf=5, l2_regularization=1.0, random_state=SEED))


def fc_ensemble(y, h):
    # seasonal-naive carries last year's pattern, ridge_fourier carries trend + smooth seasonality;
    # averaging the two beat every 3- and 4-model mix in backtests on this data
    return np.mean([fc_seasonal_naive(y, h), fc_ridge_fourier(y, h)], axis=0)


FC_MODELS = {"naive": fc_naive, "seasonal_naive": fc_seasonal_naive, "moving_avg_4w": fc_moving_avg,
             "holt_damped": fc_holt, "ridge_fourier": fc_ridge_fourier, "hgb_lags": fc_hgb_lags, "ensemble": fc_ensemble}


def backtest(y, h=HORIZON, origins=BACKTEST_ORIGINS) -> pd.DataFrame:
    n, rows = len(y), []
    for o in range(n - h - origins + 1, n - h + 1):
        for name, fn in FC_MODELS.items():
            p = fn(y[:o], h)
            rows += [{"origin": o, "step": i + 1, "model": name, "actual": y[o + i], "pred": p[i]} for i in range(h)]
    return pd.DataFrame(rows)


def fc_metrics(bt: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = []
    for name, g in bt.groupby("model"):
        e, ae, eps = g.pred - g.actual, (g.pred - g.actual).abs(), 1e-9
        pos = g.actual > 0
        r = {"model": name, "MAE": ae.mean(), "RMSE": np.sqrt((e ** 2).mean()), "MASE": ae.mean() / scale,
             "WAPE": ae.sum() / max(g.actual.abs().sum(), eps), "sMAPE": (2 * ae / (g.actual.abs() + g.pred.abs() + eps)).mean(),
             "MAPE": (ae[pos] / g.actual[pos]).mean(), "Bias": e.mean(), "Bias_pct": e.sum() / max(g.actual.sum(), eps)}
        for s, gs in g.groupby("step"):
            r[f"MAE_h{s}"] = (gs.pred - gs.actual).abs().mean()
        out.append(r)
    return pd.DataFrame(out).sort_values("MAE").reset_index(drop=True)


def _pi_bounds(resid: pd.DataFrame, exclude_origin=None):
    """Empirical interval from backtest residuals, pooled across horizon steps (12 origins x 4 steps
    is too few to do per-step quantiles). ponytail: switch to per-step pools once BACKTEST_ORIGINS >= 40."""
    r = resid[resid.origin != exclude_origin].resid.values if exclude_origin is not None else resid.resid.values
    a = (1 - PI_LEVEL) / 2
    return np.quantile(r, a), np.quantile(r, 1 - a)


def forecast(horizon: int = HORIZON, series=("paid", "claims")):
    df = make_features(load())
    ws = weekly_series(df)
    print(f"weekly history: {len(ws)} complete weeks  {ws.index[0].date()} .. {ws.index[-1].date()}")
    all_m, all_bt, nxt = [], [], []
    for s in series:
        y = ws[s].values.astype(float)
        bt = backtest(y, horizon)
        bt["series"] = s
        scale = np.mean(np.abs(np.diff(y[: bt.origin.min()])))   # in-sample naive MAE -> MASE denominator
        m = fc_metrics(bt, scale)
        m.insert(0, "series", s)
        best = m.model.iloc[0]
        # honest interval coverage: each origin's bounds come from the OTHER origins' residuals
        b = bt[bt.model == best].copy()
        b["resid"] = b.actual - b.pred
        cov, width = [], []
        for _, row in b.iterrows():
            lo, hi = _pi_bounds(b, exclude_origin=row.origin)
            cov.append(lo <= row.resid <= hi)
            width.append(hi - lo)
        m.loc[m.model == best, f"PI{int(PI_LEVEL * 100)}_coverage"] = np.mean(cov)
        m.loc[m.model == best, "PI_median_width"] = np.median(width)
        all_m.append(m)
        all_bt.append(bt)
        print(f"\n[{s}] rolling-origin backtest: {BACKTEST_ORIGINS} origins x {horizon} weeks ahead")
        print(m[["model", "MAE", "RMSE", "MASE", "WAPE", "sMAPE", "Bias_pct"]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        print(f"  -> best: {best}   {PI_LEVEL:.0%} interval coverage (leave-one-origin-out): {np.mean(cov):.1%}")
        p = FC_MODELS[best](y, horizon)
        for i in range(horizon):
            lo, hi = _pi_bounds(b)
            nxt.append({"series": s, "week_start": (ws.index[-1] + pd.Timedelta(weeks=i + 1)).date(), "step": i + 1,
                        "model": best, "forecast": round(p[i], 2), "low": round(max(p[i] + lo, 0), 2), "high": round(p[i] + hi, 2)})
    metrics = pd.concat(all_m, ignore_index=True)
    bt_all = pd.concat(all_bt, ignore_index=True)
    nxt = pd.DataFrame(nxt)
    metrics.to_csv(HERE / "forecast_metrics.csv", index=False)
    bt_all.to_csv(HERE / "forecast_backtest.csv", index=False)
    nxt.to_csv(HERE / "forecast_next.csv", index=False)
    hist = nxt.assign(made_at=datetime.now().strftime("%Y-%m-%d %H:%M"), made_from=ws.index[-1].date())
    hist.to_csv(HERE / "forecast_history.csv", mode="a", header=not (HERE / "forecast_history.csv").exists(), index=False)
    ws.to_csv(HERE / "weekly_series.csv")
    print("\nnext weeks:")
    print(nxt.to_string(index=False))
    print("\nsaved forecast_metrics.csv, forecast_backtest.csv, forecast_next.csv, weekly_series.csv")
    return metrics, bt_all, nxt


# =============================================================================
# 6. PREDICT (score a new CSV)
# =============================================================================
def predict(path: str, out: Path = HERE / "predictions.csv") -> pd.DataFrame:
    mp, md = joblib.load(HERE / "model_paid.joblib"), joblib.load(HERE / "model_denial.joblib")
    raw = load(Path(path))
    validate(raw, need_targets=False)
    d = make_features(raw)
    X = d[FEATURES]
    p_den = md["model"].predict_proba(X)[:, 1]
    paid = clipped(mp["model"].predict(X) * mp["smear"], mp["cap"])
    lo, hi = interval(paid, mp["q"])
    res = pd.DataFrame({
        "claim_id": raw["claim_id"] if "claim_id" in raw else np.arange(len(raw)),
        "service_date": d["service_date"].dt.date,
        "p_denied": p_den.round(4), "denial_flag": (p_den >= md["threshold"]).astype(int),
        "pred_paid_if_approved": paid.round(2), "pred_paid_low": lo.round(2), "pred_paid_high": hi.round(2),
        "expected_paid": ((1 - p_den) * paid).round(2),
    })
    res.to_csv(out, index=False)
    print(f"scored {len(res)} claims -> {out}   expected total paid ${res.expected_paid.sum():,.0f}   "
          f"flagged for denial {res.denial_flag.mean():.1%}")
    return res


# =============================================================================
# 7. WEEKLY LEARNING LOOP
# =============================================================================
def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: <0.10 stable, 0.10-0.25 watch, >0.25 drift."""
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if len(ref) < 20 or len(cur) < 20:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


LOG_COLS = ["logged_at", "kind", "series", "model", "model_trained_at", "made_at", "week_start", "step", "n",
            "forecast", "low", "high", "actual", "abs_error", "pct_error", "inside_interval",
            "paid_MAE", "paid_WAPE", "paid_PI_coverage", "denial_PR_AUC", "denial_recall", "denial_precision"]


def _append_log(rows: list, name: str = "learning_log.csv"):
    if not rows:
        return
    p = HERE / name
    pd.DataFrame(rows).reindex(columns=LOG_COLS).to_csv(p, mode="a", header=not p.exists(), index=False)


def update(path: str):
    """One turn of the loop: new week's adjudicated claims come in -> score what we predicted for them,
    score last week's weekly forecast, check drift, append to history, retrain, re-forecast."""
    new = load(Path(path))
    validate(new, need_targets=True)
    old = load()
    if "claim_id" in new and "claim_id" in old:
        seen = new.claim_id.isin(old.claim_id)
        if seen.any():
            print(f"{seen.sum()} of {len(new)} claims already in claims.csv: they will be replaced, not scored")
            new = new[~seen]
    if new.empty:
        sys.exit("nothing new to add")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old.to_csv(HERE / f"claims_backup_{stamp}.csv", index=False)
    logs = []

    # 1. how good was last week's claim-level prediction? (model trained BEFORE seeing these claims)
    if (HERE / "model_paid.joblib").exists():
        mp, md = joblib.load(HERE / "model_paid.joblib"), joblib.load(HERE / "model_denial.joblib")
        d = make_features(new)
        p_den = md["model"].predict_proba(d[FEATURES])[:, 1]
        appr = d[TARGET_CLF].values == 0
        paid = clipped(mp["model"].predict(d[FEATURES]) * mp["smear"], mp["cap"])
        rm = reg_metrics(d.loc[appr, TARGET_REG].values, paid[appr], *interval(paid[appr], mp["q"])) if appr.sum() > 5 else {}
        cm = clf_metrics(d[TARGET_CLF].values, p_den, md["threshold"]) if len(set(d[TARGET_CLF])) > 1 else {}
        logs.append({"logged_at": stamp, "kind": "claim_model_on_new_week", "model": mp["name"], "model_trained_at": mp["trained_at"],
                     "n": len(new), "paid_MAE": rm.get("MAE"), "paid_WAPE": rm.get("WAPE"), "paid_PI_coverage": rm.get("PI_coverage"),
                     "denial_PR_AUC": cm.get("PR_AUC"), "denial_recall": cm.get("recall"), "denial_precision": cm.get("precision")})
        print(f"claim model on new week: MAE=${rm.get('MAE', float('nan')):,.2f}  WAPE={rm.get('WAPE', float('nan')):.3f}  "
              f"denial PR_AUC={cm.get('PR_AUC', float('nan')):.3f}")

    combined = (pd.concat([old, load(Path(path))]).drop_duplicates("claim_id", keep="last") if "claim_id" in old else pd.concat([old, new]))
    combined = combined.sort_values("service_date").reset_index(drop=True)

    # 2. how good was the weekly forecast for weeks that are now complete?
    fn = HERE / "forecast_history.csv"
    if fn.exists():
        prev = pd.read_csv(fn, parse_dates=["week_start"])
        ws = weekly_series(make_features(combined))
        done = set()
        if (HERE / "learning_log.csv").exists():
            ll = pd.read_csv(HERE / "learning_log.csv")
            ll = ll[ll.kind == "weekly_forecast"]
            done = set(zip(ll.series, ll.week_start.astype(str), ll.step.astype(int), ll.made_at.astype(str)))
        for _, r in prev.iterrows():
            complete = r.week_start in ws.index and ws.loc[r.week_start, "claims"] > 0   # a week with no claims = data gap, not a zero
            if complete and (r.series, str(r.week_start.date()), int(r.step), str(r.made_at)) not in done:
                actual = ws.loc[r.week_start, r.series]
                logs.append({"logged_at": stamp, "kind": "weekly_forecast", "series": r.series, "model": r.model, "made_at": r.made_at,
                             "week_start": r.week_start.date(), "step": r.step, "forecast": r.forecast, "low": r.low, "high": r.high,
                             "actual": actual, "abs_error": abs(actual - r.forecast),
                             "pct_error": abs(actual - r.forecast) / max(actual, 1e-9), "inside_interval": int(r.low <= actual <= r.high)})
                print(f"forecast check {r.series} {r.week_start.date()} step{r.step}: forecast={r.forecast:,.0f} actual={actual:,.0f} "
                      f"err={abs(actual - r.forecast) / max(actual, 1e-9):.1%} inside={r.low <= actual <= r.high}")

    # 3. drift: new week vs. the trailing DRIFT_REF_WEEKS (not all history: deductibles, flu season etc.
    #    would flag every January as "drift")
    ref = make_features(old[old["service_date"] >= old["service_date"].max() - pd.Timedelta(weeks=DRIFT_REF_WEEKS)])
    cur = make_features(new)
    drift = [{"feature": c, "psi": psi(ref[c], cur[c])} for c in RAW_NUM + [TARGET_REG]]
    drift = pd.DataFrame(drift)
    drift["status"] = pd.cut(drift.psi, [-1, 0.10, 0.25, 99], labels=["stable", "watch", "DRIFT"])
    drift["logged_at"] = stamp
    drift.to_csv(HERE / "drift_report.csv", index=False)
    flagged = drift[drift.status != "stable"]
    print("drift:", "none" if flagged.empty else ", ".join(f"{r.feature}={r.psi:.2f}({r.status})" for r in flagged.itertuples()))

    # 4. persist + retrain + re-forecast
    combined.to_csv(DATA, index=False)
    _append_log(logs)
    print(f"\nclaims.csv now {len(combined)} rows (+{len(combined) - len(old)}); backup claims_backup_{stamp}.csv; learning_log.csv appended")
    train()          # full mode on purpose: fast mode can pick a different model family (see README)
    forecast()


def simulate(weeks: int = 8) -> pd.DataFrame:
    """Replay the weekly loop on history: each week, forecast it, then reveal it. Compares a model
    retrained every week with a 'stale' model frozen 26 weeks earlier - the value of the loop."""
    df = make_features(load())
    ws = weekly_series(df)
    sim_weeks = ws.index[-weeks:]
    stale_cut = sim_weeks[0] - pd.Timedelta(weeks=26)
    stale_df = df[(df.week < stale_cut) & (df[TARGET_CLF] == 0)]
    mk = lambda: logt(Pipeline([("prep", prep_ordinal()), ("m", hgb_reg())]))
    stale = mk().fit(stale_df[FEATURES], stale_df[TARGET_REG].values)
    rows = []
    print(f"simulating {weeks} weeks: {sim_weeks[0].date()} .. {sim_weeks[-1].date()}  (stale model frozen at {stale_cut.date()})")
    for w in sim_weeks:
        hist, cur = df[df.week < w], df[df.week == w]
        histA, curA = hist[hist[TARGET_CLF] == 0], cur[cur[TARGET_CLF] == 0]
        fresh = mk().fit(histA[FEATURES], histA[TARGET_REG].values)
        y = weekly_series(hist)["paid"].values.astype(float)
        f_ens, f_sn, f_naive = fc_ensemble(y, 1)[0], fc_seasonal_naive(y, 1)[0], fc_naive(y, 1)[0]
        actual = ws.loc[w, "paid"]
        mae = lambda m: np.abs(np.clip(m.predict(curA[FEATURES]), 0, None) - curA[TARGET_REG].values).mean()
        wape = lambda m: np.abs(np.clip(m.predict(curA[FEATURES]), 0, None) - curA[TARGET_REG].values).sum() / curA[TARGET_REG].sum()
        rows.append({"week_start": w.date(), "n_claims": len(cur), "actual_paid": round(actual, 2),
                     "forecast_ensemble": round(f_ens, 2), "ape_ensemble": abs(f_ens - actual) / actual,
                     "forecast_seasonal_naive": round(f_sn, 2), "ape_seasonal_naive": abs(f_sn - actual) / actual,
                     "forecast_naive": round(f_naive, 2), "ape_naive": abs(f_naive - actual) / actual,
                     "claim_MAE_retrained_weekly": mae(fresh), "claim_MAE_stale_26w": mae(stale),
                     "claim_WAPE_retrained_weekly": wape(fresh), "claim_WAPE_stale_26w": wape(stale)})
        r = rows[-1]
        print(f"  {w.date()}  paid actual={actual:>10,.0f} ens={f_ens:>10,.0f} ({r['ape_ensemble']:.1%})  "
              f"claim MAE retrained={r['claim_MAE_retrained_weekly']:.0f} stale={r['claim_MAE_stale_26w']:.0f}")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "learning_log_simulation.csv", index=False)
    print(f"\nmean APE  ensemble={out.ape_ensemble.mean():.1%}  seasonal_naive={out.ape_seasonal_naive.mean():.1%}  naive={out.ape_naive.mean():.1%}")
    print(f"mean claim MAE  retrained weekly={out.claim_MAE_retrained_weekly.mean():.2f}  stale 26w={out.claim_MAE_stale_26w.mean():.2f}")
    print("saved learning_log_simulation.csv")
    return out


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=10_000)
    t = sub.add_parser("train"); t.add_argument("--fast", action="store_true", help="skip tuning + repeats (~4x faster)")
    f = sub.add_parser("forecast"); f.add_argument("--horizon", type=int, default=HORIZON)
    p = sub.add_parser("predict"); p.add_argument("file")
    u = sub.add_parser("update"); u.add_argument("file")
    s = sub.add_parser("simulate"); s.add_argument("--weeks", type=int, default=8)
    for sp in (f, u, s):
        sp.add_argument("--as-of", default=None, help="date the data is complete through, e.g. 2025-12-28")
    a = ap.parse_args()
    if getattr(a, "as_of", None):
        global AS_OF
        AS_OF = a.as_of
    {"generate": lambda: generate(a.n), "train": lambda: train(a.fast), "forecast": lambda: forecast(a.horizon),
     "predict": lambda: predict(a.file), "update": lambda: update(a.file), "simulate": lambda: simulate(a.weeks)}[a.cmd]()


if __name__ == "__main__":
    main()
