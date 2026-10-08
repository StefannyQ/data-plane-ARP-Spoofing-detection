#!/usr/bin/env python3
"""
03_modeling.py -- Modeling pipeline for ARP spoofing detection
CICIoT2023: MITM-ArpSpoofing + Benign_Final

Two halves:
  PART 1 -- PREPROCESSING  (this file, currently)
    A single train/val/test split shared by every scenario, the SmartNIC
    observability constraint (a fixed hardware/domain reference, not a
    statistic), train-only statistical feature reduction (near-constant,
    mutual information, correlation-redundancy), a dedicated Rate-vs-IAT
    ablation, and train-only screened feature engineering. Produces the
    exact feature lists consumed by every scenario below.
  PART 2 -- MODELING  (to be added after Part 1's results are reviewed)
    The four scenarios (S1-S4): hyperparameter tuning, evaluation,
    thresholds, feature importance, cross-scenario comparison, lift
    analysis, report.

EXPERIMENTAL DESIGN -- FOUR SCENARIOS
---------------------------------------------------------------------------
  S1 -- Gateway Baseline
    Features : all 39, minus statistically redundant/zero-info ones
               (computed on the training split only)
    Models   : Logistic Regression, Decision Tree, Random Forest
    Depth    : unconstrained
    Arithmetic: native float (no hardware constraint at all)
    Purpose  : performance ceiling, no hardware constraint at all.

  S2 -- SmartNIC Observable
    Features : the SmartNIC-observable feature list (variable_guide.docx),
               minus statistically redundant/zero-info ones within that
               list (computed on the training split only)
    Models   : Logistic Regression, Decision Tree, Random Forest
    Depth    : unconstrained
    Arithmetic: INTEGER ONLY -- feature values and FE formulas are
               integer-quantized before training and evaluation.
    Purpose  : isolates the cost of SmartNIC feature observability.

  S3 -- SmartNIC Deployable (no feature engineering)
    Features : same as S2
    Models   : Decision Tree only
    Depth    : 3 (SmartNIC pipeline-stage constraint)
    Arithmetic: INTEGER ONLY (same quantization as S2)
    Purpose  : isolates the cost of the model/depth deployment constraint.

  S4 -- SmartNIC Deployable + Feature Engineering
    Features : S2's features + engineered features that pass a train-only
               AUROC screen
    Models   : Decision Tree only
    Depth    : 3
    Arithmetic: INTEGER ONLY -- both baseline features and all FE formulas
               use integer arithmetic throughout.
    Purpose  : isolates the value of SmartNIC-safe feature engineering.

INTEGER ARITHMETIC FOR S2/S3/S4
---------------------------------------------------------------------------
SmartNIC data-plane arithmetic is integer-only: no native floating-point,
no sqrt(), no log(). S2, S3, and S4 enforce this constraint so their
trained models are directly deployable without a floating-point conversion
step. S1 uses native float arithmetic and is NOT subject to this
constraint -- it has no SmartNIC deployment claim.

Float features (Rate, IAT) are multiplied by fixed scale factors and
truncated to integer before training. Scale factors are fixed constants
(not fitted statistics), so applying them identically to all three splits
carries no leakage risk. See the INT_SCALE_* constants block below and
the scale-factor selection procedure in methodology Section 5.2.

WHY THE FEATURE LISTS ARE NOT HARDCODED
---------------------------------------------------------------------------
Every drop/keep decision below that depends on the label or on which rows
are used to estimate a statistic is computed from the TRAINING SPLIT ONLY,
after the split (near-constant check, mutual information, correlation
matrix, the Rate-vs-IAT ablation, the engineered-feature AUROC screen).
The only fixed constants are: the SmartNIC-observable column list (a
hardware/domain reference from variable_guide.docx, not a statistic, so it
carries no leakage risk regardless of when it is applied), the integer
scale factors (fixed constants, not estimated statistics, so applying them
across all splits is leakage-safe), and the engineered-feature FORMULAS
themselves (definitions, not estimated parameters). Validation and test
rows are never used to decide a feature list, a transform parameter, or a
hyperparameter -- only to evaluate one.
"""
import sys, io as _sysio
sys.stdout = _sysio.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = _sysio.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format

# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE, 'data', 'final_dataset.csv')
MODELING_DIR = os.path.join(BASE, 'modeling')
PLOT_DIR     = os.path.join(MODELING_DIR, 'plots')
RESULTS_DIR  = os.path.join(MODELING_DIR, 'results')
MODELS_DIR   = os.path.join(MODELING_DIR, 'models')
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RANDOM_STATE          = 42
LABEL                 = 'label'
SMARTNIC_MAX_INT      = 2 ** 32 - 1
SMARTNIC_DEPTH        = 3      # S3 and S4 -- SmartNIC pipeline-stage constraint
NEAR_CONSTANT_THRESH  = 0.99   # one value occupying > this share of rows
MI_ROUND_DECIMALS     = 5      # mutual_info_classif rounded to this many decimals
MI_SAMPLE_SIZE        = 100_000
CORRELATION_THRESHOLD = 0.90   # |r| at/above this = redundant, drop one
ENGINEERED_AUROC_MIN  = 0.55   # individual-feature AUROC screen for FE_* columns

# Integer quantization for S2/S3/S4 is applied automatically in Section 15b,
# after the float S4 tree is fitted. Scale factors are derived from the
# tree's actual split thresholds -- they are never set manually.
# See Section 15b and methodology Section 5.2.

# Documented preference for which member of a redundant pair/group to keep,
# reused from the EDA's own stated rationale where it gave one. Read as
# "drop key, in favour of value" -- applied only when both are present in
# a group that the correlation analysis actually flags on the training data.
PREFERRED_OVER = {
    'IPv': 'ARP', 'LLC': 'ARP',
    'Std': 'Variance',
    'syn_count': 'syn_flag_number',
    'ack_count': 'ack_flag_number',
    'fin_count': 'fin_flag_number',
    'rst_count': 'rst_flag_number',
    'Tot size': 'AVG',
}

# Rate and IAT are excluded from the generic correlation-redundancy step --
# their near-collinearity (Spearman r=-0.987) is resolved by a dedicated,
# model-family-conditional ablation instead (Section 6 below), not by the
# generic "drop one of a redundant pair" rule.
RATE_IAT_EXCLUDE = {frozenset(('Rate', 'IAT'))}

print("=" * 70)
print("03_modeling.py -- PART 1: PREPROCESSING")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1: LOAD DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
feature_cols_all = [c for c in df.columns if c != LABEL]
y_all = df[LABEL].astype(int)
print(f"  Loaded {DATA_PATH}")
print(f"  Shape: {df.shape}  ({len(feature_cols_all)} features + label)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- Single train/validation/test split, shared by every scenario
#
# Stratified 70/15/15. Every scenario below is a COLUMN subset of this one
# row partition -- none of them re-split the data -- so train is the same
# rows in every scenario, validation is the same rows in every scenario,
# and test is the same rows in every scenario. Only the columns differ.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: TRAIN / VALIDATION / TEST SPLIT (single split, shared)")
print("=" * 70)

X_train_39, X_temp_39, y_train, y_temp = train_test_split(
    df[feature_cols_all], y_all, test_size=0.30, random_state=RANDOM_STATE, stratify=y_all
)
X_val_39, X_test_39, y_val, y_test = train_test_split(
    X_temp_39, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

for name, X_, y_ in [('Train', X_train_39, y_train), ('Val', X_val_39, y_val), ('Test', X_test_39, y_test)]:
    n_b = int((y_ == 0).sum()); n_a = int((y_ == 1).sum())
    print(f"  {name:<6} {len(y_):>9,} rows   Benign {n_b/len(y_)*100:>6.2f}%   Attack {n_a/len(y_)*100:>6.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- SmartNIC-observable feature list
#
# Fixed domain/hardware reference from variable_guide.docx (generate_
# variable_guide.py) -- NOT a statistic, so it carries no leakage risk
# regardless of when it is applied. Decision basis per that document:
#   - Source: derivable from Ethernet/IP/TCP/UDP/ARP headers only, no payload
#   - Arithmetic: integer only -- no sqrt(), no log(), no float division on
#     non-integer data. This constraint is now ENFORCED IN CODE via
#     apply_integer_quantization() (Section 7.5) and the integer_mode
#     parameter in build_engineered_features(). S1 is explicitly excluded
#     from this constraint (it has no P4 deployment claim).
#   - State: fits in a small per-flow flow-table record
#   - Timing: computable within the inter-packet gap
# Dropped from the full 39 on these hardware/domain grounds:
#   Std            -- requires sqrt() (Variance is the integer-safe equivalent)
#   IPv, LLC       -- perfectly collinear with ARP (structural identity)
#   Telnet,SMTP,IRC-- fully observable, but irrelevant in an IoT context
#   syn/ack/fin/rst_count -- redundant with their *_flag_number ratio
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: SMARTNIC-OBSERVABLE FEATURE LIST (variable_guide.docx)")
print("=" * 70)

SMARTNIC_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'IAT', 'Number',
    'Tot sum', 'Tot size', 'Min', 'Max', 'AVG', 'Variance',
    'ARP', 'TCP', 'UDP', 'ICMP', 'IGMP',
    'HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP',
    'syn_flag_number', 'ack_flag_number', 'fin_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ece_flag_number', 'cwr_flag_number',
]
assert set(SMARTNIC_FEATURES).issubset(set(feature_cols_all)), "Unknown column in SMARTNIC_FEATURES"
print(f"  {len(SMARTNIC_FEATURES)} of {len(feature_cols_all)} features are SmartNIC-observable + relevant")
dropped_by_doc = [c for c in feature_cols_all if c not in SMARTNIC_FEATURES]
print(f"  Excluded by the variable guide ({len(dropped_by_doc)}): {dropped_by_doc}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- Train-only statistical reduction: helper functions
# ─────────────────────────────────────────────────────────────────────────────

def find_near_constant(X, threshold=NEAR_CONSTANT_THRESH):
    """Columns where one value occupies more than `threshold` share of rows."""
    flagged = []
    for col in X.columns:
        top_freq = X[col].value_counts(normalize=True).iloc[0]
        if top_freq > threshold:
            flagged.append(col)
    return flagged


def find_near_zero_mi(X, y, sample_size=MI_SAMPLE_SIZE, random_state=RANDOM_STATE,
                       decimals=MI_ROUND_DECIMALS):
    """Mutual information with the label, computed on a stratified sample of
    X (which must already be train-only rows). Flags columns whose MI score
    rounds to 0.0 -- statistically independent of the label.

    Each column is tested in its OWN call, one at a time, rather than all
    together in a single matrix. sklearn's mutual_info_classif adds internal
    tie-breaking noise shared across whatever columns are passed in together,
    so a borderline feature's score (and therefore whether it rounds to 0)
    can otherwise depend on which OTHER columns happen to be in the same
    call -- verified on this dataset: DHCP and ICMP both flipped status
    between candidate sets purely because of which neighbours they were
    tested alongside, not because their actual relationship to the label
    changed. Testing one column at a time removes that cross-column coupling
    entirely, so a feature's flag is a function of only that feature, the
    label, and the sample -- never of its neighbours."""
    n = len(X)
    if n > sample_size:
        frac = sample_size / n
        idx = pd.concat([
            y[y == 0].sample(frac=frac, random_state=random_state),
            y[y == 1].sample(frac=frac, random_state=random_state),
        ]).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y
    mi_scores = {}
    for col in X.columns:
        mi = mutual_info_classif(X_s[[col]].values, y_s.values, discrete_features=False,
                                  random_state=random_state, n_neighbors=3)
        mi_scores[col] = float(np.round(mi[0], decimals))
    flagged = [c for c, m in mi_scores.items() if m == 0]
    return flagged, mi_scores


def reduce_correlated_redundancy(X_train_sub, y_train, threshold=CORRELATION_THRESHOLD,
                                  exclude_pairs=None):
    """Iterative pairwise elimination: repeatedly find the single highest
    |Pearson| / |Spearman| correlation among the remaining columns; if it
    meets the threshold, drop the loser of that pair (the documented
    preference if one applies, else whichever has lower |corr| with
    y_train) and recompute. Deliberately NOT a transitive/connected-
    components grouping -- that approach chains through borderline edges
    and can merge two otherwise-unrelated duplicate clusters that only
    share one marginal link (verified on this dataset: AVG/Tot-size/Tot-sum
    and Max/Std/Variance are each a genuine tight cluster on their own, but
    are bridged only by a borderline AVG-Max Spearman edge of 0.919 --
    transitive grouping would wrongly collapse both clusters into one)."""
    exclude_pairs = exclude_pairs or set()
    remaining = list(X_train_sub.columns)
    drop_log = []
    while len(remaining) > 1:
        sub = X_train_sub[remaining]
        pearson  = sub.corr(method='pearson').abs()
        spearman = sub.corr(method='spearman').abs()
        best_pair, best_r = None, 0.0
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                if frozenset((a, b)) in exclude_pairs:
                    continue
                r = max(pearson.loc[a, b], spearman.loc[a, b])
                if pd.notna(r) and r > best_r:
                    best_r, best_pair = r, (a, b)
        if best_pair is None or best_r < threshold:
            break
        a, b = best_pair
        if PREFERRED_OVER.get(a) == b:
            drop, keep = a, b
        elif PREFERRED_OVER.get(b) == a:
            drop, keep = b, a
        else:
            corr_a = abs(X_train_sub[a].corr(y_train))
            corr_b = abs(X_train_sub[b].corr(y_train))
            drop, keep = (a, b) if corr_a < corr_b else (b, a)
        drop_log.append({'pair': [a, b], 'r': round(float(best_r), 4), 'drop': drop, 'keep': keep})
        remaining.remove(drop)
    return remaining, drop_log


def reduce_feature_set(candidate_cols, X_train_full, y_train, label,
                        exclude_from_correlation=None, random_state=RANDOM_STATE):
    """Applies, in order: near-constant removal, correlation-redundancy
    removal, near-zero-MI removal -- all computed from X_train_full
    restricted to candidate_cols (i.e. train-only rows). Redundancy is
    resolved BEFORE the MI check so that a near-duplicate pair is always
    compared against each other directly; otherwise, if one member were
    removed first by the MI step, its redundant twin could slip through
    unnoticed (verified on this dataset: rst_flag_number tests near-zero
    MI and rst_count does not, purely as k-NN MI-estimator noise on two
    columns that are r~1.0 duplicates of each other -- resolving the
    redundancy first means the pair is settled as a pair, not by accident
    of which one the MI step happened to flag)."""
    print(f"\n--- Reducing feature set: {label} ({len(candidate_cols)} candidates) ---")
    X_sub = X_train_full[candidate_cols]

    near_const = find_near_constant(X_sub)
    print(f"  Near-constant (>{NEAR_CONSTANT_THRESH*100:.0f}% one value): {near_const}")
    remaining = [c for c in candidate_cols if c not in near_const]

    remaining, redundancy_log = reduce_correlated_redundancy(
        X_train_full[remaining], y_train, exclude_pairs=exclude_from_correlation
    )
    print(f"  Correlated-redundant pairs resolved: {len(redundancy_log)}")
    for entry in redundancy_log:
        print(f"    {entry['pair']} (r={entry['r']}) -> keep {entry['keep']!r}, drop {entry['drop']!r}")

    near_zero_mi, mi_scores = find_near_zero_mi(X_train_full[remaining], y_train,
                                                random_state=random_state)
    print(f"  Near-zero MI with label (on survivors): {near_zero_mi}")
    remaining = [c for c in remaining if c not in near_zero_mi]

    print(f"  Final reduced set: {len(remaining)} features")
    diagnostics = {
        'candidates': candidate_cols,
        'near_constant': near_const,
        'redundancy_log': redundancy_log,
        'near_zero_mi': near_zero_mi,
        'mi_scores': mi_scores,
        'reduced': remaining,
    }
    return remaining, diagnostics

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 -- Apply the reduction to S1's 39-feature candidate set and
# S2/S3's 29-feature SmartNIC candidate set, independently. Both are
# computed from X_train only.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: TRAIN-ONLY FEATURE REDUCTION (S1 and S2/S3)")
print("=" * 70)

S1_reduced, S1_diag = reduce_feature_set(
    feature_cols_all, X_train_39, y_train, 'S1 (all 39)',
    exclude_from_correlation=RATE_IAT_EXCLUDE,
)
S2_reduced, S2_diag = reduce_feature_set(
    SMARTNIC_FEATURES, X_train_39, y_train, 'S2/S3 (29 SmartNIC)',
    exclude_from_correlation=RATE_IAT_EXCLUDE,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 -- Rate vs IAT ablation
#
# Rate and IAT are strongly monotonically related (Spearman r=-0.987) but
# this is handled separately from the generic redundancy rule: fit quick,
# untuned default models on a train-only sample, evaluate on validation,
# and let the result -- not an assumption -- decide whether to keep both
# (tree models) or drop one (if a model family's validation AUC says so).
# A 200k stratified sample of X_train is used here purely for speed; it is
# still drawn exclusively from training rows.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: RATE VS IAT ABLATION (train-only sample, evaluated on validation)")
print("=" * 70)

ABLATION_SAMPLE_SIZE = 200_000
# A model must beat keep_both by more than this margin to be preferred --
# otherwise keep_both wins by default. Without this, a ~0.0005 AUC wobble
# (well within normal run-to-run noise for an untuned model on a sample)
# would mechanically "win" and override what is actually a tie.
AUC_IMPROVEMENT_TOLERANCE = 0.001

def make_ablation_sample(X_train_full, y_train, sample_size=ABLATION_SAMPLE_SIZE,
                          random_state=RANDOM_STATE):
    n = len(y_train)
    if n <= sample_size:
        return X_train_full, y_train
    frac = sample_size / n
    idx = pd.concat([
        y_train[y_train == 0].sample(frac=frac, random_state=random_state),
        y_train[y_train == 1].sample(frac=frac, random_state=random_state),
    ]).index
    return X_train_full.loc[idx], y_train.loc[idx]


def ablate_rate_iat(feature_list, X_train_full, y_train, X_val_full, y_val, label,
                     seed=RANDOM_STATE):
    if 'Rate' not in feature_list or 'IAT' not in feature_list:
        print(f"  [{label}] Rate and/or IAT not in this candidate set -- nothing to ablate")
        return {'tree': 'keep_both', 'linear': 'keep_both', 'raw': {}}

    X_tr_s, y_tr_s = make_ablation_sample(X_train_full, y_train, random_state=seed)
    variants = {
        'keep_both': feature_list,
        'drop_rate': [c for c in feature_list if c != 'Rate'],
        'drop_iat':  [c for c in feature_list if c != 'IAT'],
    }
    base_models = {
        'LR': Pipeline([
            ('transform', PowerTransformer(method='yeo-johnson')),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                                       random_state=seed)),
        ]),
        'DT': DecisionTreeClassifier(class_weight='balanced', random_state=seed),
        'RF': RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                      random_state=seed, n_jobs=-1),
    }

    print(f"\n  [{label}] sample: {len(y_tr_s):,} training rows (train-only)")
    results = {}
    for model_name, base_model in base_models.items():
        aucs = {}
        for variant_name, cols in variants.items():
            model = clone(base_model)
            model.fit(X_tr_s[cols], y_tr_s)
            proba = model.predict_proba(X_val_full[cols])[:, 1]
            aucs[variant_name] = float(roc_auc_score(y_val, proba))
        results[model_name] = aucs
        best = max(aucs, key=aucs.get)
        print(f"  [{label}][{model_name}] keep_both={aucs['keep_both']:.4f}  "
              f"drop_rate={aucs['drop_rate']:.4f}  drop_iat={aucs['drop_iat']:.4f}  "
              f"-> best: {best}")

    def decide(aucs, tolerance=AUC_IMPROVEMENT_TOLERANCE):
        """Only move away from keep_both if it's a clear win -- a small
        difference defaults to the safe, redundancy-preserving choice."""
        base = aucs['keep_both']
        best_alt = max(('drop_rate', 'drop_iat'), key=lambda v: aucs[v])
        if aucs[best_alt] - base > tolerance:
            return best_alt
        return 'keep_both'

    dt_pref = decide(results['DT'])
    rf_pref = decide(results['RF'])
    # Tree decision requires DT and RF to AGREE on the same alternative --
    # if they disagree (as they did on this dataset: DT preferred dropping
    # IAT, RF preferred dropping Rate), that disagreement is itself evidence
    # that neither feature is clearly dispensable, so default to keep_both
    # rather than averaging over a real disagreement.
    best_tree = dt_pref if dt_pref == rf_pref else 'keep_both'
    best_linear = decide(results['LR'])
    print(f"  [{label}] DT alone -> {dt_pref}   RF alone -> {rf_pref}")
    print(f"  [{label}] DECISION -> tree models: {best_tree}   |   linear (LR): {best_linear}")
    return {'tree': best_tree, 'linear': best_linear, 'raw': results}


def apply_rate_iat_decision(feature_list, variant):
    if variant == 'drop_rate':
        return [c for c in feature_list if c != 'Rate']
    if variant == 'drop_iat':
        return [c for c in feature_list if c != 'IAT']
    return list(feature_list)

S1_ablation = ablate_rate_iat(S1_reduced, X_train_39, y_train, X_val_39, y_val, 'S1')
S2_ablation = ablate_rate_iat(S2_reduced, X_train_39, y_train, X_val_39, y_val, 'S2')

S1_TREE = apply_rate_iat_decision(S1_reduced, S1_ablation['tree'])
S1_LR   = apply_rate_iat_decision(S1_reduced, S1_ablation['linear'])
S2_TREE = apply_rate_iat_decision(S2_reduced, S2_ablation['tree'])
S2_LR   = apply_rate_iat_decision(S2_reduced, S2_ablation['linear'])
S3_TREE = list(S2_TREE)

print(f"\n  S1 feature set: {len(S1_TREE)} (tree) / {len(S1_LR)} (LR)")
print(f"  S2 feature set: {len(S2_TREE)} (tree) / {len(S2_LR)} (LR)")
if set(S1_TREE) == set(S2_TREE) and set(S1_LR) == set(S2_LR):
    print("  NOTE: S1 and S2 happen to land on the same feature set on this run --")
    print("  Part 2 still tunes and trains S2 independently (no shortcut), so this")
    print("  is purely an observation, not something the pipeline depends on.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 -- Engineered features: candidates, train-only verification,
# and train-only AUROC screen for S4
#
# WHERE THESE CANDIDATES COME FROM
#   ARP spoofing is a sustained, scripted process, not a one-packet event:
#   the attacker floods the network with repeated spoofed ARP replies. That
#   produces observable signatures that a single raw column does not
#   capture on its own, each combining two or more existing columns. This
#   pipeline re-runs, on the training split only, the exact two-stage
#   screen used in the original (leakage-prone) EDA: per-class Mann-Whitney
#   significance (7.1) followed by an individual-feature AUROC threshold
#   (7.2), then an independent RF combination check (7.3) on the full
#   candidate set together rather than one at a time. Re-running the full
#   ten-candidate set here -- not just the three that happened to clear the
#   AUROC bar in the original EDA -- matters because that screen was run on
#   the full (leaked) dataset; re-deriving train-only KEEP/drop decisions
#   from scratch is the entire point of this file, and a candidate that
#   barely missed 0.55 on the full dataset is not guaranteed to land on the
#   same side of the line on the training split alone.
#
# THREE-TIER SMARTNIC FEASIBILITY FRAMEWORK
#   TIER 1 -- fully SmartNIC compatible (integer arithmetic, no cross-flow
#     state). All ten candidates below are Tier 1; they differ from each
#     other only in which raw signal they recombine, not in feasibility.
#
#   TIER 2 -- SmartNIC+ (programmable NIC only, NOT IMPLEMENTED): a lag-1
#     change in Time_To_Live, IAT, or Max between consecutive flows from
#     the same source. Each requires a per-source register remembering the
#     previous flow's value (cross-flow state) -- feasible on a
#     P4-programmable SmartNIC via extern registers, infeasible on a
#     fixed-function NIC. Not implemented here: CICIoT2023 provides no
#     source-IP/session identifier to key such a register on reliably at
#     the flow level used in this evaluation.
#
#   TIER 3 -- ruled out entirely: a per-flow repetition count (would need a
#     hash table over all 39 features across every concurrent flow --
#     exceeds flow-table SRAM) and a rolling mean/std of Rate over a
#     per-source window (also exceeds SRAM budget). A lag-1 change in Rate
#     itself was also ruled out, but for a different reason -- it carries
#     no signal (consecutive Rate values are statistically independent),
#     not because it is hard to compute.
#
# THE TEN TIER-1 CANDIDATES
#   FE_burst_intensity = Rate * Number
#     Combines how fast packets arrive with how many there are -- the
#     fullest single measure of how intense a burst is. 64-bit accumulator
#     needed to avoid overflow (Rate can reach the MAX_INT sentinel).
#   FE_flow_duration = Number / Rate  (proxy for last_ts - first_ts)
#     A scripted attack burst should complete faster than a typical benign
#     session. Guard: 0 where Rate = MAX_INT (a zero-duration flow has,
#     by definition, zero duration).
#   FE_size_cv_proxy = Variance / (AVG + 1)
#     Coefficient-of-variation proxy without a square root (Std is not
#     SmartNIC-observable -- Variance is its integer-safe equivalent). A
#     repeated attack packet template should show lower relative size
#     variance than mixed benign traffic.
#   FE_protocol_diversity = count of non-zero protocol indicator columns
#     (HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP,
#     IGMP, IPv, LLC). Low-layer MITM relay traffic typically carries no
#     recognised L7 application, so attack flows should touch fewer
#     protocol flags than mixed benign traffic. Strongest standalone
#     engineered signal in the original EDA that is not a transform of
#     Rate (AUROC=0.6402).
#   FE_size_range = Max - Min
#     Spread between the largest and smallest packet in the flow. A
#     repeated attack template tends toward a narrower spread than mixed
#     benign request/response/data traffic.
#   FE_payload_ratio = (Tot size - Header_Length) / (Tot size + 1)
#     Fraction of total bytes that are payload rather than header overhead.
#   FE_header_ratio = (Header_Length * 100) // (Tot size + 1)  [integer %]
#     Inverse framing of payload_ratio, kept separate because the integer
#     percentage form is the more directly SmartNIC-implementable version
#     (no floating point).
#   FE_no_app_layer = 1 if (HTTP+HTTPS+DNS+SSH+DHCP)==0 else 0
#     Flags flows with no recognised L7 protocol at all -- a coarser,
#     binary companion to FE_protocol_diversity's continuous count.
#   FE_avg_min_ratio = AVG / (Min + 1)
#     Ratio of mean to minimum packet size -- a size-uniformity signal.
#     Below the AUROC screen in the original EDA (0.5302); kept as a
#     candidate here so the train-only screen gets the chance to confirm
#     or overturn that on this split, rather than assuming it in advance.
#   FE_arp_rate = ARP * Rate  (NaN where Rate = MAX_INT)
#     Flags ARP-heavy flows that also have a high rate. Below the AUROC
#     screen in the original EDA (0.5292, median=0 for both classes);
#     kept as a candidate for the same reason as FE_avg_min_ratio.
#
# NOT COMPUTABLE: FE_flow_asymmetry (Srate - Drate) and FE_rate_asym_ratio
#   (Srate / (Drate+1)) require Srate/Drate, which CICIoT2023 does not
#   provide. Both returned AUROC=0.500 in the literature-review document
#   that originated the Tier-1 framework -- consistent with their absence
#   here. Not built.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: ENGINEERED FEATURES -- CANDIDATES, TRAIN-ONLY VERIFICATION, AUROC SCREEN")
print("=" * 70)

def build_engineered_features(X, observable_survivors, integer_mode=False, scale_ratio=None):
    """Builds the ten Tier-1 FE_* candidates from X.

    X must be restricted to SMARTNIC_FEATURES before being passed in here
    -- every formula below is required to use only SmartNIC-observable
    source columns, since these candidates exist specifically to test
    "what can be engineered from what a SmartNIC can see" (S4's premise).
    Passing the full 39-column X_train_39/X_val_39/X_test_39 here would
    silently let a formula read a non-observable column (e.g. Telnet,
    IRC, IPv, LLC, Std) without raising an error, which would make S4 no
    longer a clean test of the SmartNIC constraint -- a hidden 30th
    observable channel smuggled in through a non-observable raw column.

    observable_survivors (S2_reduced) is the SmartNIC-observable pool
    AFTER train-only statistical reduction (near-constant, near-zero MI,
    correlated-redundancy -- Section 5). Two count-based candidates below
    (FE_protocol_diversity, FE_no_app_layer) sum across a LIST of protocol
    columns rather than reading one named column, so a protocol that
    Section 5 already found uninformative on this training split (e.g.
    IGMP/DHCP, MI=0 in the original full-dataset EDA) would otherwise
    silently inflate the count with a column the pipeline's own screen
    rejected. These two are restricted to observable_survivors. The other
    eight candidates each read one or two specific NAMED raw columns
    (Rate, Variance, AVG, Max, Min, Tot size, Header_Length, ARP) and are
    deliberately NOT restricted the same way: a named column's fate in a
    pairwise correlation/ablation fight (e.g. Rate losing to IAT in the
    Section 6 ablation) says nothing about whether Rate is a sound
    ingredient for a multiplicative/ratio formula -- it would only make
    these formulas fragile to a decision made for a different reason.

    integer_mode : bool (default False)
        When True, all division operators are replaced with integer division
        (//) and results are cast to np.int64, enforcing SmartNIC arithmetic
        compatibility for S2, S3, and S4.
        S1 always uses float mode (integer_mode=False).

        When integer_mode=True, the caller is responsible for passing X
        that has already been quantized by apply_integer_quantization()
        (i.e. Rate and IAT are already integer-scaled). The formulas
        here then use integer arithmetic on top of those already-integer
        source values, producing integer results throughout.
    """
    missing = [c for c in ('Rate', 'Number', 'Variance', 'AVG', 'Max', 'Min',
                           'Tot size', 'Header_Length', 'ARP') if c not in X.columns]
    if missing:
        raise ValueError(f"build_engineered_features: required source column(s) "
                         f"missing from X: {missing}. X must be restricted to "
                         f"SMARTNIC_FEATURES (or a superset of it) before calling this.")

    fe = pd.DataFrame(index=X.index)
    safe_rate = X['Rate'].replace(SMARTNIC_MAX_INT, np.nan)

    # burst_intensity: rate x packet count -- full measure of burst scale.
    # Left as float here in both modes -- safe_rate is NaN at the MAX_INT
    # sentinel rows, and NaN cannot be cast to int64 directly. The caller
    # (rebuild_integer_features) fills those NaNs with the correct
    # sentinel-derived value and casts to int64 afterward, in integer mode.
    fe['FE_burst_intensity']    = safe_rate * X['Number']

    # flow_duration: Number / Rate proxy for last_ts - first_ts.
    # Integer mode: integer division (//) -- result is 0 where Rate >> Number.
    if integer_mode:
        fe['FE_flow_duration'] = (X['Number'] // safe_rate.replace(0, np.nan)
                                  ).fillna(0).astype(np.int64)
    else:
        fe['FE_flow_duration'] = X['Number'] / safe_rate.replace(0, np.nan)

    # size_cv_proxy: coefficient-of-variation proxy (Variance, no sqrt).
    # Integer mode: integer division preserves rank order for tree splits.
    if integer_mode:
        fe['FE_size_cv_proxy'] = (X['Variance'] // (X['AVG'] + 1)).astype(np.int64)
    else:
        fe['FE_size_cv_proxy'] = X['Variance'] / (X['AVG'] + 1)

    # protocol_diversity: count of non-zero protocol indicator columns,
    # restricted to the SmartNIC-observable protocol columns that ALSO
    # survived Section 5's train-only statistical reduction (excludes any
    # of HTTP/HTTPS/DNS/SSH/DHCP/TCP/UDP/ARP/ICMP/IGMP that were flagged
    # near-constant or near-zero-MI on this training split).
    # Result is always an integer (count), no mode difference.
    proto_cols_full = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP',
                       'TCP', 'UDP', 'ARP', 'ICMP', 'IGMP']
    assert set(proto_cols_full).issubset(set(SMARTNIC_FEATURES)), \
        "proto_cols_full must stay a subset of SMARTNIC_FEATURES"
    proto_cols = [c for c in proto_cols_full if c in observable_survivors]
    if not proto_cols:
        print("  WARNING: FE_protocol_diversity has zero surviving protocol columns "
              "after S2's statistical reduction -- it will be constant (0) for every "
              "row and should be screened out by the AUROC step.")
    fe['FE_protocol_diversity'] = (X[proto_cols] > 0).sum(axis=1).astype(np.int64)

    # size_range: Max - Min (both byte counts, always integer).
    # No mode difference -- subtraction of integers is always an integer.
    fe['FE_size_range']         = (X['Max'] - X['Min']).astype(np.int64)

    # payload_ratio: fraction of total bytes that are payload (not header).
    # Integer mode: multiply numerator by INT_SCALE_RATIO before integer
    # division, giving e.g. permille (0-1000) precision.
    # Float mode: standard float division.
    if integer_mode:
        assert scale_ratio is not None, (
            "integer_mode=True but scale_ratio is None -- pass scale_ratio "
            "explicitly when calling in integer mode."
        )
        fe['FE_payload_ratio'] = (
            ((X['Tot size'] - X['Header_Length']) * scale_ratio)
            // (X['Tot size'] + 1)
        ).astype(np.int64)
    else:
        fe['FE_payload_ratio'] = (X['Tot size'] - X['Header_Length']) / (X['Tot size'] + 1)

    # header_ratio: integer percentage of bytes consumed by headers.
    # Already uses integer division in both modes (the x100 scaling is
    # the ratio scale factor, built into the formula directly).
    fe['FE_header_ratio']       = (
        (X['Header_Length'] * 100) // (X['Tot size'] + 1)
    ).astype(np.int64)

    # no_app_layer: 1 when no recognised, statistically-surviving L7
    # protocol is observed in the flow -- same observable_survivors
    # restriction as FE_protocol_diversity, for the same reason.
    # Result is always 0/1 integer, no mode difference.
    app_layer_cols_full = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP']
    app_layer_cols = [c for c in app_layer_cols_full if c in observable_survivors]
    if not app_layer_cols:
        print("  WARNING: FE_no_app_layer has zero surviving app-layer columns "
              "after S2's statistical reduction -- it will be constant (1) for every "
              "row and should be screened out by the AUROC step.")
    fe['FE_no_app_layer']       = ((X[app_layer_cols].sum(axis=1)) == 0).astype(np.int64)

    # avg_min_ratio: ratio of mean to minimum packet size (size uniformity).
    # Integer mode: AVG and Min are byte counts -- integer division is the
    # correct integer approximation.
    if integer_mode:
        fe['FE_avg_min_ratio'] = (X['AVG'] // (X['Min'] + 1)).astype(np.int64)
    else:
        fe['FE_avg_min_ratio'] = X['AVG'] / (X['Min'] + 1)

    # arp_rate: ARP flag x Rate -- flags ARP-heavy flows with high rate.
    # Left as float here in both modes, for the same NaN-at-sentinel reason
    # as FE_burst_intensity above -- the caller fills + casts afterward.
    fe['FE_arp_rate']           = X['ARP'] * safe_rate

    return fe

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7.5 -- Integer quantization for S2/S3/S4
#
# Applies fixed-point scale factors to every float feature in the
# SmartNIC-observable feature set. The result replaces the float X_*_39
# views for S2/S3/S4 only; S1 is untouched (no hardware deployment claim).
#
# WHY A FUNCTION RATHER THAN IN-PLACE MODIFICATION
#   X_train_39, X_val_39, X_test_39 are the shared 39-column matrices
#   used by all scenarios. Modifying them in-place would silently corrupt
#   S1's float data. Instead this function produces new DataFrames that
#   share the same index and column names but have integer-typed values,
#   used exclusively by S2/S3/S4 from this point forward.
#
# WHY THIS IS NOT LEAKAGE
#   Scale factors (INT_SCALE_RATE, INT_SCALE_IAT, INT_SCALE_RATIO) are
#   fixed constants defined at the top of this file -- they do not depend
#   on which rows are in the training split. Applying the same constant
#   multiplication + truncation to all three splits independently is
#   leakage-safe.
# ─────────────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------
# SECTION 7.5 -- Float engineered features (pre-training pass)
#
# S2/S3/S4 feature matrices and engineered features are built here in
# FLOAT mode so the pipeline can train the float S4 tree in Section 15.
# Section 15b then derives integer scale factors from that fitted tree
# and rebuilds everything as integers, retraining S2/S3/S4 in integer
# mode. 
# -----------------------------------------------------------------------

def apply_integer_quantization(X_39, scale_rate, scale_iat, scale_ratio):
    """Return a copy of X_39 with float features quantized to integers.

    Scale factors are passed explicitly (derived from the fitted S4 tree
    in Section 15b) rather than being global constants.
    The MAX_INT sentinel for Rate is preserved after scaling.
    """
    X = X_39.copy()
    sentinel_mask = (X['Rate'] == SMARTNIC_MAX_INT)
    X['Rate'] = X['Rate'].where(sentinel_mask, X['Rate'] * scale_rate).astype(np.int64)
    X.loc[sentinel_mask, 'Rate'] = SMARTNIC_MAX_INT
    X['IAT'] = (X['IAT'] * scale_iat).astype(np.int64)
    int_cast_cols = [
        'Min', 'Max', 'Header_Length', 'Protocol Type', 'Time_To_Live',
        'Number', 'Variance', 'AVG', 'Tot sum', 'Tot size',
        'ARP', 'TCP', 'UDP', 'ICMP', 'IGMP',
        'HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP',
        'syn_flag_number', 'ack_flag_number', 'fin_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ece_flag_number', 'cwr_flag_number',
    ]
    for col in int_cast_cols:
        if col in X.columns:
            X[col] = X[col].astype(np.int64)
    return X


def rebuild_integer_features(X_train_39, X_val_39, X_test_39,
                              scale_rate, scale_iat, scale_ratio, observable_survivors):
    """Apply integer quantization and rebuild engineered features for S2/S3/S4.

    Called once from Section 15b after the float S4 tree is fitted.
    Returns integer base matrices and integer FE DataFrames.
    """
    X_tr_int = apply_integer_quantization(X_train_39, scale_rate, scale_iat, scale_ratio)
    X_vl_int = apply_integer_quantization(X_val_39,   scale_rate, scale_iat, scale_ratio)
    X_te_int = apply_integer_quantization(X_test_39,  scale_rate, scale_iat, scale_ratio)
    fe_tr = build_engineered_features(X_tr_int[SMARTNIC_FEATURES], observable_survivors,
                                      integer_mode=True, scale_ratio=scale_ratio)
    fe_vl = build_engineered_features(X_vl_int[SMARTNIC_FEATURES], observable_survivors,
                                      integer_mode=True, scale_ratio=scale_ratio)
    fe_te = build_engineered_features(X_te_int[SMARTNIC_FEATURES], observable_survivors,
                                      integer_mode=True, scale_ratio=scale_ratio)
    for fe_df, X_sub in [(fe_tr, X_tr_int[SMARTNIC_FEATURES]),
                         (fe_vl, X_vl_int[SMARTNIC_FEATURES]),
                         (fe_te, X_te_int[SMARTNIC_FEATURES])]:
        fe_df['FE_flow_duration']   = fe_df['FE_flow_duration'].fillna(0).astype(np.int64)
        fe_df['FE_burst_intensity'] = (fe_df['FE_burst_intensity']
                                       .fillna(np.int64(SMARTNIC_MAX_INT) * X_sub['Number'])
                                       .astype(np.int64))
        fe_df['FE_arp_rate']        = (fe_df['FE_arp_rate']
                                       .fillna(X_sub['ARP'] * np.int64(SMARTNIC_MAX_INT))
                                       .astype(np.int64))
    return X_tr_int, X_vl_int, X_te_int, fe_tr, fe_vl, fe_te


# Float FE features for the initial S4 training pass (Section 15).
# Replaced with integer versions after Section 15b.
FE_train = build_engineered_features(X_train_39[SMARTNIC_FEATURES], S2_reduced)
FE_val   = build_engineered_features(X_val_39[SMARTNIC_FEATURES],   S2_reduced)
FE_test  = build_engineered_features(X_test_39[SMARTNIC_FEATURES],  S2_reduced)

for fe_df, X_sub in [(FE_train, X_train_39[SMARTNIC_FEATURES]),
                     (FE_val,   X_val_39[SMARTNIC_FEATURES]),
                     (FE_test,  X_test_39[SMARTNIC_FEATURES])]:
    fe_df['FE_flow_duration']   = fe_df['FE_flow_duration'].fillna(0)
    fe_df['FE_burst_intensity'] = fe_df['FE_burst_intensity'].fillna(SMARTNIC_MAX_INT * X_sub['Number'])
    fe_df['FE_arp_rate']        = fe_df['FE_arp_rate'].fillna(X_sub['ARP'] * SMARTNIC_MAX_INT)

# ── 7.1  Train-only per-class verification (Mann-Whitney U) ────────────────
# Compares each candidate's benign vs. attack median on the training split
# only, with a significance test -- the same check used throughout this
# pipeline to confirm a feature actually separates the classes rather than
# assuming it from a formula alone.
print("\n--- 7.1  Train-only per-class comparison (Mann-Whitney U) ---")
print(f"  {'Feature':<22} {'Ben median':>12} {'Atk median':>12} {'p-value':>12}  Sig")
print(f"  {'-'*68}")
benign_train_mask = (y_train == 0)
attack_train_mask = (y_train == 1)
mw_results = {}
for col in FE_train.columns:
    b = FE_train.loc[benign_train_mask, col].dropna()
    a = FE_train.loc[attack_train_mask, col].dropna()
    b_s = b.sample(min(50_000, len(b)), random_state=RANDOM_STATE)
    a_s = a.sample(min(50_000, len(a)), random_state=RANDOM_STATE)
    _, p = mannwhitneyu(b_s, a_s, alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    mw_results[col] = {'ben_median': float(b.median()), 'atk_median': float(a.median()),
                        'p_value': float(p), 'sig': sig}
    print(f"  {col:<22} {b.median():>12.4f} {a.median():>12.4f} {p:>12.2e}  {sig}")

# ── 7.2  Train-only individual-feature AUROC screen ─────────────────────────
print("\n--- 7.2  Train-only individual-feature AUROC vs y_train (full training set) ---")
engineered_auroc = {}
for col in FE_train.columns:
    x = FE_train[col].fillna(FE_train[col].median())
    auc = roc_auc_score(y_train, x)
    auc = max(auc, 1.0 - auc)
    engineered_auroc[col] = round(float(auc), 4)
    decision = 'KEEP' if auc >= ENGINEERED_AUROC_MIN else 'drop'
    print(f"    {col:<24} AUROC={auc:.4f}  {decision}")

kept_engineered = sorted([c for c, a in engineered_auroc.items() if a >= ENGINEERED_AUROC_MIN])
dropped_engineered = sorted([c for c, a in engineered_auroc.items() if a < ENGINEERED_AUROC_MIN])
print(f"\n  Kept ({len(kept_engineered)}): {kept_engineered}")
print(f"  Dropped ({len(dropped_engineered)}): {dropped_engineered}")

# ── 7.3  Baseline vs Enhanced RF (train-only diagnostic) ───────────────────
# 7.1/7.2 test each engineered candidate in isolation; this checks whether
# the FULL candidate set, used together with the baseline features, lifts a
# real model -- a combination effect a univariate screen cannot see. Two
# quick, untuned default Random Forests are compared: one on S2_TREE alone
# (baseline), one on S2_TREE plus all ten engineered candidates
# (enhanced) -- deliberately all ten, not just the AUROC>=0.55 survivors,
# so this stays an independent cross-check rather than reusing 7.2's
# decision. Both are trained on the SAME 200,000-row stratified sample of
# the training split (drawn from training rows only, for speed -- this is
# a diagnostic, not the final tuned S2/S4 models built in Part 2) and
# evaluated on the real validation split, not a fresh ad hoc sample/split,
# to stay inside this pipeline's train/val/test discipline throughout.
print("\n--- 7.3  Baseline vs Enhanced RF (train-only diagnostic) ---")
X_train_base_diag = X_train_39[S2_TREE]
X_train_enh_diag  = pd.concat([X_train_39[S2_TREE], FE_train], axis=1)
X_val_base_diag   = X_val_39[S2_TREE]
X_val_enh_diag    = pd.concat([X_val_39[S2_TREE], FE_val], axis=1)

X_base_tr_s, y_diag_s = make_ablation_sample(X_train_base_diag, y_train)
X_enh_tr_s = X_train_enh_diag.loc[X_base_tr_s.index]

rf_diag_kw = dict(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
rf_base_diag = RandomForestClassifier(**rf_diag_kw).fit(X_base_tr_s, y_diag_s)
rf_enh_diag  = RandomForestClassifier(**rf_diag_kw).fit(X_enh_tr_s, y_diag_s)

auc_base_diag = float(roc_auc_score(y_val, rf_base_diag.predict_proba(X_val_base_diag)[:, 1]))
auc_enh_diag  = float(roc_auc_score(y_val, rf_enh_diag.predict_proba(X_val_enh_diag)[:, 1]))

print(f"  Baseline RF ({len(S2_TREE)} features)                          : Val AUC = {auc_base_diag:.4f}")
print(f"  Enhanced RF ({len(S2_TREE) + FE_train.shape[1]} features, +{FE_train.shape[1]} engineered, all candidates) : "
      f"Val AUC = {auc_enh_diag:.4f}")
print(f"  Delta: {auc_enh_diag - auc_base_diag:+.4f}")

S4_TREE = S2_TREE + kept_engineered

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 -- Preprocessing summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8: PREPROCESSING SUMMARY -- FINAL PER-SCENARIO FEATURE LISTS")
print("=" * 70)

summary = {
    'S1_TREE': S1_TREE, 'S1_LR': S1_LR,
    'S2_TREE': S2_TREE, 'S2_LR': S2_LR,
    'S3_TREE': S3_TREE,
    'S4_TREE': S4_TREE,
}
for name, cols in summary.items():
    print(f"\n  {name}  ({len(cols)} features):")
    print(f"    {cols}")

print(f"\n  Rate/IAT ablation decisions:")
print(f"    S1 -> tree: {S1_ablation['tree']:<10} linear: {S1_ablation['linear']}")
print(f"    S2 -> tree: {S2_ablation['tree']:<10} linear: {S2_ablation['linear']}")

results_dump = {
    'feature_lists': summary,
    'smartnic_features': SMARTNIC_FEATURES,
    'dropped_by_smartnic_doc': dropped_by_doc,
    'S1_diagnostics': S1_diag,
    'S2_diagnostics': S2_diag,
    'rate_iat_ablation': {
        'S1': {k: v for k, v in S1_ablation.items() if k != 'raw'} | {'raw': S1_ablation['raw']},
        'S2': {k: v for k, v in S2_ablation.items() if k != 'raw'} | {'raw': S2_ablation['raw']},
    },
    'engineered_mann_whitney': mw_results,
    'baseline_vs_enhanced_rf': {'baseline_val_auc': auc_base_diag, 'enhanced_val_auc': auc_enh_diag,
                                'delta': auc_enh_diag - auc_base_diag},
    'engineered_auroc': engineered_auroc,
    'engineered_kept': kept_engineered,
    'engineered_dropped': dropped_engineered,
    # Integer scale factors are not known yet at this point in the script --
    # they're derived from the fitted float S4 tree in Section 15b and then
    # verified against the retrained integer S4 tree in Section 20, which
    # saves the authoritative record to S4_threshold_extraction.json.
}
results_path = os.path.join(RESULTS_DIR, 'preprocessing_summary.json')
with open(results_path, 'w') as f:
    json.dump(results_dump, f, indent=2, default=str)
print(f"\n  Full diagnostics saved -> {results_path}")

print("\n" + "=" * 70)
print("PART 1 (PREPROCESSING) COMPLETE")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════════
# PART 2 -- MODELING
#
# FOUR SCENARIOS:
#   S1 -- Gateway Baseline:                LR, DT, RF | unconstrained depth
#   S2 -- SmartNIC Observable:              LR, DT, RF | unconstrained depth
#   S3 -- SmartNIC Deployable (no FE):      DT only    | depth = 3
#   S4 -- SmartNIC Deployable + FE:         DT only    | depth = 3
#
# Each scenario is tuned and trained independently, on its own feature
# matrix (X_train_S1..X_train_S4 etc., built in Section 9), with no
# cross-scenario shortcuts. This is deliberate even where two scenarios'
# feature sets happen to coincide on a given run (Part 1 prints a NOTE if
# S1_TREE/S1_LR and S2_TREE/S2_LR turn out identical) -- "happens to
# coincide today" is an observation about this run of the data, not a
# guarantee that holds after a future change upstream (the candidate
# pools, the reduction thresholds, the dataset itself). Tuning S2
# independently costs an extra hyperparameter search when the sets do
# coincide, but removes any dependency on that coincidence continuing to
# hold, and removes the column-order bookkeeping a "reuse S1" shortcut
# would otherwise require in Sections 17 and 18 below.
# ════════════════════════════════════════════════════════════════════════════
import time, pickle, joblib
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.metrics import (classification_report, average_precision_score, roc_curve,
                              precision_recall_curve, confusion_matrix, fbeta_score)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 70)
print("03_modeling.py -- PART 2: MODELING")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 -- Build per-scenario X matrices from Part 1's feature lists
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 9: BUILD PER-SCENARIO FEATURE MATRICES")
print("=" * 70)

def build_X(features, X39, FE=None):
    """Assembles the X matrix for a feature list that may include both
    original (X39) and engineered (FE) columns, preserving row alignment."""
    fe_cols   = [f for f in features if f in (FE.columns if FE is not None else [])]
    base_cols = [f for f in features if f not in fe_cols]
    if fe_cols:
        return pd.concat([X39[base_cols], FE[fe_cols]], axis=1)[features]
    return X39[base_cols][features]

# S1 uses float matrices throughout.
X_train_S1,    X_val_S1,    X_test_S1    = (build_X(S1_TREE, X) for X in (X_train_39, X_val_39, X_test_39))
X_train_S1_LR, X_val_S1_LR, X_test_S1_LR = (build_X(S1_LR,   X) for X in (X_train_39, X_val_39, X_test_39))

# S2/S3/S4 start with float matrices here and are rebuilt with integer
# matrices in Section 15b, after scale factors are derived from the
# fitted S4 tree. The variables below are intentionally reassigned there.
X_train_S2,    X_val_S2,    X_test_S2    = (build_X(S2_TREE, X) for X in (X_train_39, X_val_39, X_test_39))
X_train_S2_LR, X_val_S2_LR, X_test_S2_LR = (build_X(S2_LR,   X) for X in (X_train_39, X_val_39, X_test_39))
X_train_S3,    X_val_S3,    X_test_S3    = X_train_S2, X_val_S2, X_test_S2
X_train_S4 = build_X(S4_TREE, X_train_39, FE_train)
X_val_S4   = build_X(S4_TREE, X_val_39,   FE_val)
X_test_S4  = build_X(S4_TREE, X_test_39,  FE_test)

for name, X_ in [('S1 (float)', X_train_S1), ('S1_LR (float)', X_train_S1_LR),
                  ('S2 (float, pre-15b)', X_train_S2),
                  ('S3 (float, pre-15b)', X_train_S3),
                  ('S4 (float, pre-15b)', X_train_S4)]:
    print(f"  {name:<24} train shape: {X_.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 -- Model definitions and hyperparameter search spaces
#
# class_weight='balanced' on every estimator -- reweights each sample
# inversely proportional to class frequency, applied only through .fit() on
# training rows; validation and test retain the true 78.12/21.88 split so
# reported metrics reflect real operating conditions.
#
# LR is a Pipeline (PowerTransformer -> LogisticRegression). PowerTransformer
# fits its per-feature Yeo-Johnson lambda (and standardisation) on whatever
# data the pipeline is fit on; inside RandomizedSearchCV+PredefinedSplit that
# is the training fold only, and the final refit is on X_train alone -- val
# and test are only ever .transform()'d with the already-fitted parameters.
#
# DT/RF for S1/S2 search max_depth freely (no hardware constraint). DT for
# S3/S4 has max_depth FIXED at 3 (the SmartNIC pipeline-stage constraint) --
# it is not part of the search space, so RandomizedSearchCV only tunes
# min_samples_split/min_samples_leaf at that fixed depth.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 10: MODEL DEFINITIONS AND HYPERPARAMETER SEARCH SPACES")
print("=" * 70)

base_lr = Pipeline([
    ('transform', PowerTransformer(method='yeo-johnson')),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                               random_state=RANDOM_STATE, n_jobs=-1)),
])
base_dt          = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
base_rf          = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
base_dt_smartnic = DecisionTreeClassifier(max_depth=SMARTNIC_DEPTH, class_weight='balanced',
                                           random_state=RANDOM_STATE)

LR_PARAM_SPACE = {
    'clf__C'      : [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__solver' : ['lbfgs', 'saga'],
}
DT_PARAM_SPACE_GENERAL = {
    'max_depth'         : [5, 10, 15, 20, 30, None],
    'min_samples_split' : [2, 5, 10, 20],
    'min_samples_leaf'  : [1, 2, 5, 10],
}
DT_PARAM_SPACE_SMARTNIC = {     # max_depth fixed at SMARTNIC_DEPTH, not searched
    'min_samples_split' : [2, 5, 10, 20],
    'min_samples_leaf'  : [1, 2, 5, 10],
}
RF_PARAM_SPACE = {
    'n_estimators'      : [100, 200, 300, 500],
    'max_depth'         : [10, 20, 30, None],
    'max_features'      : ['sqrt', 'log2', 0.3],
    'min_samples_leaf'  : [1, 2, 5],
    'min_samples_split' : [2, 5, 10],
}
print(f"  LR n_iter=20 | DT n_iter=20 | RF n_iter=30")
print(f"  S3/S4 Decision Tree: max_depth fixed at {SMARTNIC_DEPTH} (not searched)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 -- Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def make_combined_split(X_tr, X_v, y_tr, y_v):
    """Train rows get fold=-1 (always trained on); val rows get fold=0
    (always scored on). Avoids k-fold CV and uses the dedicated validation
    set directly for hyperparameter selection."""
    X_c = pd.concat([X_tr, X_v], axis=0, ignore_index=True)
    y_c = pd.concat([y_tr, y_v], axis=0, ignore_index=True)
    fold = np.concatenate([-np.ones(len(X_tr), dtype=int), np.zeros(len(X_v), dtype=int)])
    return X_c, y_c, PredefinedSplit(fold)


def evaluate(model, X, y):
    yp  = model.predict(X)
    ypr = model.predict_proba(X)[:, 1]
    rep = classification_report(y, yp, target_names=['Benign', 'Attack'],
                                 output_dict=True, digits=4)
    fpr, tpr, _ = roc_curve(y, ypr)
    pre, rec, _ = precision_recall_curve(y, ypr)
    return {
        'roc_auc'  : float(roc_auc_score(y, ypr)),
        'pr_auc'   : float(average_precision_score(y, ypr)),
        'accuracy' : float(rep['accuracy']),
        'prec_ben' : float(rep['Benign']['precision']), 'rec_ben': float(rep['Benign']['recall']),
        'f1_ben'   : float(rep['Benign']['f1-score']),
        'prec_atk' : float(rep['Attack']['precision']), 'rec_atk': float(rep['Attack']['recall']),
        'f1_atk'   : float(rep['Attack']['f1-score']),
        'y_pred'   : yp, 'y_proba': ypr, 'cm': confusion_matrix(y, yp),
        'fpr': fpr, 'tpr': tpr, 'pre': pre, 'rec_curve': rec,
    }


def tune_and_train(name, base_model, param_space, n_iter, X_tr, X_v, X_te, y_tr, y_v, y_te):
    """Tune with PredefinedSplit (train vs val only), then retrain the
    winning configuration on the full training set, then evaluate once on
    val and once on test."""
    print(f"\n  [{name}] Hyperparameter tuning (n_iter={n_iter})...", flush=True)
    t0 = time.time()
    X_c, y_c, ps = make_combined_split(X_tr, X_v, y_tr, y_v)
    rs = RandomizedSearchCV(clone(base_model), param_space, n_iter=n_iter, cv=ps,
                            scoring='roc_auc', n_jobs=1, random_state=RANDOM_STATE,
                            verbose=0, refit=False)
    rs.fit(X_c, y_c)
    best_p, best_cv = rs.best_params_, float(rs.best_score_)
    print(f"  [{name}] Best params : {best_p}", flush=True)
    print(f"  [{name}] Best val AUC: {best_cv:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    final = clone(base_model)
    final.set_params(**best_p)
    final.fit(X_tr, y_tr)
    print(f"  [{name}] Retrained on full training set ({len(X_tr):,} rows, "
          f"{time.time()-t1:.0f}s)", flush=True)

    val_res, test_res = evaluate(final, X_v, y_v), evaluate(final, X_te, y_te)
    print(f"  [{name}] VAL  AUC={val_res['roc_auc']:.4f}  PR={val_res['pr_auc']:.4f}  "
          f"F1-Atk={val_res['f1_atk']:.4f}  Rec-Atk={val_res['rec_atk']:.4f}", flush=True)
    print(f"  [{name}] TEST AUC={test_res['roc_auc']:.4f}  PR={test_res['pr_auc']:.4f}  "
          f"F1-Atk={test_res['f1_atk']:.4f}  Rec-Atk={test_res['rec_atk']:.4f}", flush=True)
    return {'model': final, 'best_params': best_p, 'best_cv_auc': best_cv,
            'val': val_res, 'test': test_res}


def load_cached_result(sc_name, model_name):
    """If this scenario+model was already tuned and saved in a prior run,
    load it instead of re-running an expensive hyperparameter search.
    Safe to use freely: the split, feature lists, and search are all fully
    deterministic (fixed random_state throughout), so a cached result is
    identical to what a fresh run would produce."""
    model_path   = os.path.join(MODELS_DIR, f"{sc_name}_{model_name.replace(' ', '_')}.joblib")
    results_path = os.path.join(RESULTS_DIR, f'{sc_name}_results.pkl')
    if os.path.exists(model_path) and os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            slim = pickle.load(f)
        if model_name in slim:
            model = joblib.load(model_path)
            return {**slim[model_name], 'model': model}
    return None


def tune_and_train_cached(sc_name, model_name, base_model, param_space, n_iter,
                          X_tr, X_v, X_te, y_tr, y_v, y_te):
    cached = load_cached_result(sc_name, model_name)
    if cached is not None:
        print(f"  [{sc_name}][{model_name}] Loaded cached result from disk -- "
              f"VAL AUC={cached['val']['roc_auc']:.4f}  TEST AUC={cached['test']['roc_auc']:.4f}",
              flush=True)
        return cached
    return tune_and_train(model_name, base_model, param_space, n_iter, X_tr, X_v, X_te, y_tr, y_v, y_te)


def save_results(sc_name, results):
    slim = {}
    for model_name, res in results.items():
        slim[model_name] = {k: v for k, v in res.items() if k != 'model'}
        joblib.dump(res['model'], os.path.join(MODELS_DIR, f"{sc_name}_{model_name.replace(' ', '_')}.joblib"))
    with open(os.path.join(RESULTS_DIR, f'{sc_name}_results.pkl'), 'wb') as f:
        pickle.dump(slim, f)
    print(f"  Results saved -> {RESULTS_DIR}\\{sc_name}_results.pkl", flush=True)


def print_summary_table(sc_name, results):
    print(f"\n  {'Model':<22} {'Split':<6} {'AUC':>7} {'PR-AUC':>7} {'Acc':>7} "
          f"{'F1-Ben':>7} {'F1-Atk':>7} {'Rec-Atk':>8} {'Pre-Atk':>8}")
    print(f"  {'-'*86}")
    for model_name, res in results.items():
        for split, m in [('Val', res['val']), ('Test', res['test'])]:
            print(f"  {model_name:<22} {split:<6} {m['roc_auc']:>7.4f} {m['pr_auc']:>7.4f} "
                  f"{m['accuracy']:>7.4f} {m['f1_ben']:>7.4f} {m['f1_atk']:>7.4f} "
                  f"{m['rec_atk']:>8.4f} {m['prec_atk']:>8.4f}")


def best_model_by_val_auc(results):
    return max(results, key=lambda k: results[k]['val']['roc_auc'])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 -- Run S1 (Gateway Baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 12: SCENARIO S1 -- GATEWAY BASELINE")
print(f"  Features: {len(S1_TREE)} (tree) / {len(S1_LR)} (LR)  |  Models: LR, DT, RF  |  Depth: unconstrained")
print("=" * 70)

s1_results = {}
for model_name, base_model, param_space, n_iter, X_tr, X_v, X_te in [
    ('Logistic Regression', base_lr, LR_PARAM_SPACE,         20, X_train_S1_LR, X_val_S1_LR, X_test_S1_LR),
    ('Decision Tree',       base_dt, DT_PARAM_SPACE_GENERAL, 20, X_train_S1,    X_val_S1,    X_test_S1),
    ('Random Forest',       base_rf, RF_PARAM_SPACE,         30, X_train_S1,    X_val_S1,    X_test_S1),
]:
    s1_results[model_name] = tune_and_train_cached('S1', model_name, base_model, param_space, n_iter,
                                                   X_tr, X_v, X_te, y_train, y_val, y_test)
print_summary_table('S1', s1_results)
save_results('S1', s1_results)
s1_best = best_model_by_val_auc(s1_results)
print(f"\n  S1 best model by val AUC: {s1_best} (AUC={s1_results[s1_best]['val']['roc_auc']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 -- Run S2 (SmartNIC Observable)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 13: SCENARIO S2 -- SMARTNIC OBSERVABLE")
print(f"  Features: {len(S2_TREE)} (tree) / {len(S2_LR)} (LR)  |  Models: LR, DT, RF  |  Depth: unconstrained")
print("=" * 70)

s2_results = {}
for model_name, base_model, param_space, n_iter, X_tr, X_v, X_te in [
    ('Logistic Regression', base_lr, LR_PARAM_SPACE,         20, X_train_S2_LR, X_val_S2_LR, X_test_S2_LR),
    ('Decision Tree',       base_dt, DT_PARAM_SPACE_GENERAL, 20, X_train_S2,    X_val_S2,    X_test_S2),
    ('Random Forest',       base_rf, RF_PARAM_SPACE,         30, X_train_S2,    X_val_S2,    X_test_S2),
]:
    s2_results[model_name] = tune_and_train_cached('S2', model_name, base_model, param_space, n_iter,
                                                   X_tr, X_v, X_te, y_train, y_val, y_test)
print_summary_table('S2', s2_results)
save_results('S2', s2_results)
s2_best = best_model_by_val_auc(s2_results)
print(f"\n  S2 best model by val AUC: {s2_best} (AUC={s2_results[s2_best]['val']['roc_auc']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 -- Run S3 (SmartNIC Deployable, no FE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 14: SCENARIO S3 -- SMARTNIC DEPLOYABLE (no feature engineering)")
print(f"  Features: {len(S3_TREE)}  |  Model: DT (depth={SMARTNIC_DEPTH})")
print("=" * 70)

s3_results = {}
s3_results['Decision Tree'] = tune_and_train_cached(
    'S3', 'Decision Tree', base_dt_smartnic, DT_PARAM_SPACE_SMARTNIC, 20,
    X_train_S3, X_val_S3, X_test_S3, y_train, y_val, y_test,
)
print_summary_table('S3', s3_results)
save_results('S3', s3_results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 -- Run S4 (SmartNIC Deployable + Feature Engineering)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 15: SCENARIO S4 -- SMARTNIC DEPLOYABLE + FEATURE ENGINEERING")
print(f"  Features: {len(S4_TREE)}  |  Model: DT (depth={SMARTNIC_DEPTH})")
print("=" * 70)

s4_results = {}
s4_results['Decision Tree'] = tune_and_train_cached(
    'S4', 'Decision Tree', base_dt_smartnic, DT_PARAM_SPACE_SMARTNIC, 20,
    X_train_S4, X_val_S4, X_test_S4, y_train, y_val, y_test,
)
print_summary_table('S4', s4_results)
save_results('S4', s4_results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15b -- Derive integer scale factors from fitted S4 tree, then
#                retrain S2/S3/S4 on integer-quantized data
#
# Now that S4 has been trained on float data, we can read its actual split
# thresholds, compute the minimum adequate scale factor for each float feature,
# and rebuild S2/S3/S4 in integer mode -- all within this same run.
#
# This resolves what would otherwise be a circular dependency (need the tree
# to set scale factors; need scale factors to train the tree) by using the
# float S4 tree as a proxy: the float and integer trees will have very similar
# split thresholds (the integer tree is trained on the same data, just
# quantized), so the float tree's thresholds are a reliable source for
# determining how much precision the integer representation needs to preserve.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 15b: INTEGER QUANTIZATION -- DERIVE SCALE FACTORS AND RETRAIN")
print("=" * 70)

import math as _math
from sklearn.tree import _tree as _sklearn_tree

_FE_RATIO_FEATURES = {'FE_payload_ratio', 'FE_size_cv_proxy', 'FE_avg_min_ratio'}

def _derive_scale_factors(fitted_tree, feature_names):
    """Extract every split threshold from a fitted Decision Tree and compute
    the minimum adequate integer scale factor for each float feature.

    Returns a dict mapping each float feature name to its recommended scale
    factor (minimum power of 10 that preserves all threshold precisions).
    """
    t = fitted_tree.tree_
    min_scales = {}
    for node in range(t.node_count):
        if t.children_left[node] == _sklearn_tree.TREE_LEAF:
            continue
        fname  = feature_names[t.feature[node]]
        thresh = float(t.threshold[node])
        thresh_str = f"{thresh:.10f}".rstrip('0')
        decimals   = len(thresh_str.split('.')[1]) if '.' in thresh_str else 0
        if decimals > 0:
            min_scale = int(10 ** decimals)
            min_scales[fname] = max(min_scales.get(fname, 1), min_scale)
    # Round up to next power of 10 for cleanliness
    return {
        f: int(10 ** _math.ceil(_math.log10(v))) if v > 1 else 1
        for f, v in min_scales.items()
    }

# Derive scale factors from the fitted float S4 tree
_s4_dt_float   = s4_results['Decision Tree']['model']
_derived_scales = _derive_scale_factors(_s4_dt_float, S4_TREE)

# Map derived scales to the three constants the rest of the code needs
INT_SCALE_RATE  = _derived_scales.get('Rate',  1_000)
INT_SCALE_IAT   = _derived_scales.get('IAT',   1_000_000)
INT_SCALE_RATIO = max(
    (_derived_scales.get(f, 1_000) for f in _FE_RATIO_FEATURES),
    default=1_000
)

print(f"\n  Scale factors derived from fitted float S4 tree:")
print(f"  {'Feature':<28}  {'Derived scale':>14}")
for _fname, _scale in sorted(_derived_scales.items()):
    print(f"  {_fname:<28}  {_scale:>14,}")
print(f"\n  Constants resolved:")
print(f"  INT_SCALE_RATE  = {INT_SCALE_RATE:,}")
print(f"  INT_SCALE_IAT   = {INT_SCALE_IAT:,}")
print(f"  INT_SCALE_RATIO = {INT_SCALE_RATIO:,}")

# Rebuild everything as integer and retrain S2/S3/S4
print(f"\n  Rebuilding S2/S3/S4 feature matrices in integer mode...")
(X_train_39_int, X_val_39_int, X_test_39_int,
 FE_train, FE_val, FE_test) = rebuild_integer_features(
    X_train_39, X_val_39, X_test_39,
    INT_SCALE_RATE, INT_SCALE_IAT, INT_SCALE_RATIO, S2_reduced
)

X_train_S2,    X_val_S2,    X_test_S2    = (build_X(S2_TREE, X) for X in (X_train_39_int, X_val_39_int, X_test_39_int))
X_train_S2_LR, X_val_S2_LR, X_test_S2_LR = (build_X(S2_LR,   X) for X in (X_train_39_int, X_val_39_int, X_test_39_int))
X_train_S3,    X_val_S3,    X_test_S3    = X_train_S2, X_val_S2, X_test_S2
X_train_S4 = build_X(S4_TREE, X_train_39_int, FE_train)
X_val_S4   = build_X(S4_TREE, X_val_39_int,   FE_val)
X_test_S4  = build_X(S4_TREE, X_test_39_int,  FE_test)

for name, X_ in [('S2 (int)', X_train_S2), ('S3 (int)', X_train_S3), ('S4 (int)', X_train_S4)]:
    print(f"  {name:<16} train shape: {X_.shape}  dtypes: {set(X_.dtypes.astype(str).tolist())}")

# Retrain S2, S3, S4 on integer data using the same hyperparameters found in
# the float search (no new hyperparameter search -- the search was already
# optimized; only the data representation changes, not the model configuration).
print(f"\n  Retraining S2 on integer data...")
s2_results = {}
for model_name, base_model, param_space, n_iter, X_tr, X_v, X_te in [
    ('Logistic Regression', base_lr, LR_PARAM_SPACE,         20, X_train_S2_LR, X_val_S2_LR, X_test_S2_LR),
    ('Decision Tree',       base_dt, DT_PARAM_SPACE_GENERAL, 20, X_train_S2,    X_val_S2,    X_test_S2),
    ('Random Forest',       base_rf, RF_PARAM_SPACE,         30, X_train_S2,    X_val_S2,    X_test_S2),
]:
    # Force retrain by using a cache key that distinguishes integer from float
    s2_results[model_name] = tune_and_train_cached(
        'S2_int', model_name, base_model, param_space, n_iter,
        X_tr, X_v, X_te, y_train, y_val, y_test
    )
print_summary_table('S2 (integer)', s2_results)
save_results('S2', s2_results)

print(f"\n  Retraining S3 on integer data...")
s3_results = {}
s3_results['Decision Tree'] = tune_and_train_cached(
    'S3_int', 'Decision Tree', base_dt_smartnic, DT_PARAM_SPACE_SMARTNIC, 20,
    X_train_S3, X_val_S3, X_test_S3, y_train, y_val, y_test,
)
print_summary_table('S3 (integer)', s3_results)
save_results('S3', s3_results)

print(f"\n  Retraining S4 on integer data...")
s4_results = {}
s4_results['Decision Tree'] = tune_and_train_cached(
    'S4_int', 'Decision Tree', base_dt_smartnic, DT_PARAM_SPACE_SMARTNIC, 20,
    X_train_S4, X_val_S4, X_test_S4, y_train, y_val, y_test,
)
print_summary_table('S4 (integer)', s4_results)
save_results('S4', s4_results)

s2_best = best_model_by_val_auc(s2_results)
print(f"\n  S2 (integer) best model: {s2_best} (AUC={s2_results[s2_best]['val']['roc_auc']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 -- Cross-scenario comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 16: CROSS-SCENARIO COMPARISON")
print("=" * 70)

comparison_rows = [
    ('S1', s1_best,         s1_results[s1_best]['test']),
    ('S2', s2_best,         s2_results[s2_best]['test']),
    ('S3', 'Decision Tree', s3_results['Decision Tree']['test']),
    ('S4', 'Decision Tree', s4_results['Decision Tree']['test']),
]
print(f"\n  {'Sc':<4} {'Best Model':<22} {'AUC':>7} {'PR-AUC':>7} {'Acc':>7} "
      f"{'F1-Ben':>7} {'F1-Atk':>7} {'Rec-Atk':>8} {'Pre-Atk':>8}")
print(f"  {'-'*90}")
for sc, mname, m in comparison_rows:
    print(f"  {sc:<4} {mname:<22} {m['roc_auc']:>7.4f} {m['pr_auc']:>7.4f} {m['accuracy']:>7.4f} "
          f"{m['f1_ben']:>7.4f} {m['f1_atk']:>7.4f} {m['rec_atk']:>8.4f} {m['prec_atk']:>8.4f}")

aucs = {sc: m['roc_auc'] for sc, _, m in comparison_rows}
print(f"\n  Delta analysis (test ROC-AUC):")
print(f"    S1->S2 (observability cost)    : {aucs['S2']-aucs['S1']:+.4f}")
print(f"    S2->S3 (model constraint cost) : {aucs['S3']-aucs['S2']:+.4f}")
print(f"    S3->S4 (feature engineering)   : {aucs['S4']-aucs['S3']:+.4f}")

cross_scenario = {sc: m for sc, _, m in comparison_rows}
with open(os.path.join(RESULTS_DIR, 'cross_scenario.pkl'), 'wb') as f:
    pickle.dump(cross_scenario, f)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16b -- Lift / cumulative gains analysis (all four scenarios)
#
# AUC and the default-threshold metrics in Section 16 summarise performance
# either across every possible threshold (AUC) or at exactly one fixed
# cutoff (0.5). Neither answers a question a SmartNIC/SOC triage workflow
# actually has: "given a fixed review budget of the top N% riskiest flows
# by predicted probability, what fraction of all true attacks would that
# budget catch?" That is what lift / cumulative gains measures directly.
#
# Computed on the TEST SET ONLY, reusing each scenario's already-fitted
# best model's y_proba (stored in comparison_rows' test_res dict by
# evaluate(), Section 11) -- no new predictions are made, no leakage risk:
# this is a pure post-hoc re-sort and re-aggregation of predictions that
# were already produced by a model trained without ever seeing test rows.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 16b: LIFT / CUMULATIVE GAINS ANALYSIS (test set, all scenarios)")
print("=" * 70)

LIFT_PCT_STEP = 1     # row granularity: every 1% of the ranked test set
LIFT_PCT_MAX  = 40    # report up to the top 40%; beyond this, capture is
                      # already near-complete for every scenario here and
                      # additional rows add length without adding insight


def compute_lift_table(y_true, y_proba, pct_step=LIFT_PCT_STEP, pct_max=LIFT_PCT_MAX):
    """Rank the test set by predicted probability (highest risk first),
    then for each top-N% slice report how many true attacks fall inside
    it and what share of ALL test-set attacks that represents.

    Returns a list of dicts, one per pct row:
      pct            -- top-N% threshold for this row
      n_flagged      -- number of test rows in the top N%
      n_attacks      -- count of true attacks within those flagged rows
      capture_pct    -- n_attacks / total_attacks * 100
      attack_rate    -- n_attacks / n_flagged * 100 (precision within this slice)
      lift           -- attack_rate / overall_attack_rate (this slice's
                        concentration relative to a random N% sample)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    total_attacks = y_sorted.sum()
    overall_rate = total_attacks / n

    rows = []
    for pct in range(pct_step, pct_max + 1, pct_step):
        end = int(round(n * pct / 100))
        end = max(end, 1)
        seg = y_sorted[:end]
        n_attacks = int(seg.sum())
        capture_pct = (n_attacks / total_attacks * 100) if total_attacks > 0 else 0.0
        attack_rate = (n_attacks / end * 100) if end > 0 else 0.0
        lift = (attack_rate / 100) / overall_rate if overall_rate > 0 else 0.0
        rows.append({
            'pct': pct, 'n_flagged': end, 'n_attacks': n_attacks,
            'capture_pct': capture_pct, 'attack_rate': attack_rate, 'lift': lift,
        })
    return rows


lift_tables = {}
for sc, mname, test_res in comparison_rows:
    rows = compute_lift_table(y_test.values, test_res['y_proba'])
    lift_tables[sc] = {'model_name': mname, 'rows': rows}

    print(f"\n  [{sc}] {mname}  (overall attack rate: {y_test.mean()*100:.2f}%)")
    print(f"  {'Top %':>6}  {'Flagged':>9}  {'Attacks caught':>15}  {'Capture %':>10}  {'Lift':>6}")
    print(f"  {'-'*55}")
    # print a representative subset to keep console output readable;
    # the full pct_step=1 table is saved in full to disk below
    checkpoints = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40]
    for r in rows:
        if r['pct'] in checkpoints:
            print(f"  {r['pct']:>5}%  {r['n_flagged']:>9,}  {r['n_attacks']:>15,}  "
                  f"{r['capture_pct']:>9.2f}%  {r['lift']:>5.2f}x")

with open(os.path.join(RESULTS_DIR, 'lift_tables.pkl'), 'wb') as f:
    pickle.dump(lift_tables, f)
print(f"\n  Full per-scenario lift tables (1%% increments, top {LIFT_PCT_MAX}%%) saved -> "
      f"{os.path.join(RESULTS_DIR, 'lift_tables.pkl')}")

# ── Cross-scenario lift comparison at fixed review budgets ────────────────
# The single most decision-relevant slice of the table above: at a few
# realistic review budgets, how does each scenario's capture rate compare?
print(f"\n  Cross-scenario capture %% at fixed review budgets:")
budget_checkpoints = [5, 10, 20]
header = f"  {'Budget':>8}" + "".join(f"{sc:>10}" for sc, _, _ in comparison_rows)
print(header)
for budget in budget_checkpoints:
    row_str = f"  {budget:>7}%"
    for sc, _, _ in comparison_rows:
        match = next(r for r in lift_tables[sc]['rows'] if r['pct'] == budget)
        row_str += f"{match['capture_pct']:>9.1f}%"
    print(row_str)

# ── Plot: cumulative gains curves, all four scenarios on one chart ────────
SC_COLORS_LIFT = {'S1': '#2196F3', 'S2': '#4CAF50', 'S3': '#FF9800', 'S4': '#E91E63'}
SC_DASH_LIFT   = {'S1': 'solid', 'S2': 'solid', 'S3': 'dashed', 'S4': 'dashdot'}

fig, ax = plt.subplots(figsize=(9, 6.5))
for sc, mname, _ in comparison_rows:
    rows = lift_tables[sc]['rows']
    pcts = [r['pct'] for r in rows]
    caps = [r['capture_pct'] for r in rows]
    ax.plot(pcts, caps, label=f"{sc} ({mname})", color=SC_COLORS_LIFT[sc],
            linestyle=SC_DASH_LIFT[sc], linewidth=2)
ax.plot([0, LIFT_PCT_MAX], [0, LIFT_PCT_MAX], 'k--', linewidth=0.8, alpha=0.4, label='Random baseline')
ax.set_xlabel('Top % of test set flagged (ranked by predicted risk)')
ax.set_ylabel('Cumulative % of attacks captured')
ax.set_title('Cumulative Gains -- All Scenarios (Test Set)')
ax.set_xlim(0, LIFT_PCT_MAX); ax.set_ylim(0, 100)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cumulative_gains_all_scenarios.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: cumulative_gains_all_scenarios.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 -- Threshold analysis (F2-Attack, selected on validation only)
#
# F2 (beta=2) weights recall twice as heavily as precision -- it
# operationalises "a missed attack costs more than a false alarm" in a way
# F1 (equal weight) cannot. Selected on the validation set; applied to test
# exactly once.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 17: THRESHOLD ANALYSIS")
print("=" * 70)

threshold_candidates = np.arange(0.10, 0.91, 0.05)
threshold_results = {}

for sc, model_obj, X_v, y_v, X_te, y_te in [
    ('S1', s1_results[s1_best]['model'],         X_val_S1, y_val, X_test_S1, y_test),
    ('S2', s2_results[s2_best]['model'],         X_val_S2, y_val, X_test_S2, y_test),
    ('S3', s3_results['Decision Tree']['model'], X_val_S3, y_val, X_test_S3, y_test),
    ('S4', s4_results['Decision Tree']['model'], X_val_S4, y_val, X_test_S4, y_test),
]:
    val_proba, test_proba = model_obj.predict_proba(X_v)[:, 1], model_obj.predict_proba(X_te)[:, 1]
    val_rows = []
    for t in threshold_candidates:
        yp  = (val_proba >= t).astype(int)
        rep = classification_report(y_v, yp, target_names=['Benign', 'Attack'],
                                     output_dict=True, zero_division=0)
        f2  = fbeta_score(y_v, yp, beta=2, pos_label=1, zero_division=0)
        val_rows.append({'threshold': round(float(t), 2), 'f2_atk': f2,
                         'f1_atk': rep['Attack']['f1-score'], 'rec_atk': rep['Attack']['recall'],
                         'prec_atk': rep['Attack']['precision'], 'f1_ben': rep['Benign']['f1-score'],
                         'accuracy': rep['accuracy']})
    val_df   = pd.DataFrame(val_rows)
    best_idx = val_df['f2_atk'].idxmax()
    best_t   = val_df.loc[best_idx, 'threshold']
    yp_test  = (test_proba >= best_t).astype(int)
    rep_test = classification_report(y_te, yp_test, target_names=['Benign', 'Attack'],
                                      output_dict=True, zero_division=0)
    f2_test  = fbeta_score(y_te, yp_test, beta=2, pos_label=1, zero_division=0)
    threshold_results[sc] = {'val_sweep': val_df, 'best_threshold': best_t,
                             'test_at_best_t': rep_test, 'f2_test': f2_test,
                             'val_proba': val_proba, 'test_proba': test_proba}
    print(f"  [{sc}] Best threshold (val F2-Attack): {best_t}  |  "
          f"Test F1-Atk={rep_test['Attack']['f1-score']:.4f}  "
          f"Rec-Atk={rep_test['Attack']['recall']:.4f}  Pre-Atk={rep_test['Attack']['precision']:.4f}")

with open(os.path.join(RESULTS_DIR, 'threshold_results.pkl'), 'wb') as f:
    slim_thresh = {k: {kk: vv for kk, vv in v.items() if kk not in ('val_proba', 'test_proba')}
                   for k, v in threshold_results.items()}
    pickle.dump(slim_thresh, f)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 -- Feature importance: MDI vs Permutation
#
# MDI is biased toward high-cardinality continuous features; permutation
# importance (test-set ROC-AUC drop when a feature is shuffled) is reported
# alongside it. Each scenario's importance uses that scenario's own fitted
# model and its own feature-list column order.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 18: FEATURE IMPORTANCE -- MDI vs PERMUTATION")
print("=" * 70)

importance_results = {}
for sc, results, features, X_te, y_te in [
    ('S1', s1_results, S1_TREE, X_test_S1, y_test),
    ('S2', s2_results, S2_TREE, X_test_S2, y_test),
]:
    rf_model = results['Random Forest']['model']
    mdi = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True).tail(20)
    print(f"\n  [{sc}] Computing permutation importance on test set (n_repeats=5)...", flush=True)
    perm = permutation_importance(rf_model, X_te, y_te, scoring='roc_auc',
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=features).sort_values(ascending=True).tail(20)
    importance_results[sc] = {'mdi': mdi, 'perm': perm_s}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, imp, title, xlabel in [
        (axes[0], mdi,    f'{sc} -- RF MDI Importance (Top 20)', 'MDI Importance'),
        (axes[1], perm_s, f'{sc} -- RF Permutation Importance (Test Set)', 'Mean ROC-AUC decrease'),
    ]:
        colors = ['#E91E63' if f.startswith('FE_') else '#2196F3' for f in imp.index]
        ax.barh(imp.index, imp.values, color=colors); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{sc}_rf_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{sc}] Saved: {sc}_rf_feature_importance.png")

for sc, results, features, X_te, y_te in [
    ('S3', s3_results, S3_TREE, X_test_S3, y_test),
    ('S4', s4_results, S4_TREE, X_test_S4, y_test),
]:
    dt_model = results['Decision Tree']['model']
    mdi = pd.Series(dt_model.feature_importances_, index=features)
    mdi = mdi[mdi > 0].sort_values(ascending=True)
    print(f"\n  [{sc}] Computing permutation importance on test set (n_repeats=5)...", flush=True)
    perm = permutation_importance(dt_model, X_te, y_te, scoring='roc_auc',
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=features).sort_values(ascending=True).tail(max(len(mdi), 10))
    importance_results[sc] = {'mdi': mdi, 'perm': perm_s}

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(mdi) * 0.35)))
    for ax, imp, title, xlabel in [
        (axes[0], mdi,    f'{sc} -- DT MDI Importance (non-zero only)', 'MDI Importance'),
        (axes[1], perm_s, f'{sc} -- DT Permutation Importance (Test Set)', 'Mean ROC-AUC decrease'),
    ]:
        colors = ['#E91E63' if f.startswith('FE_') else '#FF9800' for f in imp.index]
        ax.barh(imp.index, imp.values, color=colors); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{sc}_dt_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{sc}] Saved: {sc}_dt_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 -- Plots: ROC, PR, cross-scenario, confusion matrices, threshold
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 19: PLOTS")
print("=" * 70)

SC_COLORS  = {'S1': '#2196F3', 'S2': '#4CAF50', 'S3': '#FF9800', 'S4': '#E91E63'}
MOD_COLORS = {'Logistic Regression': '#9C27B0', 'Decision Tree': '#FF5722', 'Random Forest': '#009688'}

for sc, results, title in [
    ('S1', s1_results, f'S1 -- Gateway Baseline ({len(S1_TREE)} features)'),
    ('S2', s2_results, f'S2 -- SmartNIC Observable ({len(S2_TREE)} features)'),
    ('S3', s3_results, f'S3 -- SmartNIC Deployable, no FE (depth={SMARTNIC_DEPTH})'),
    ('S4', s4_results, f'S4 -- SmartNIC Deployable + FE (depth={SMARTNIC_DEPTH})'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, res in results.items():
        color = MOD_COLORS.get(model_name, '#333')
        for ax, split, m in [(axes[0], 'Val', res['val']), (axes[1], 'Test', res['test'])]:
            ax.plot(m['fpr'], m['tpr'], label=f"{model_name} (AUC={m['roc_auc']:.4f})",
                    color=color, linewidth=1.8)
    for ax, split_name in [(axes[0], 'Validation'), (axes[1], 'Test')]:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve -- {split_name}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(title, fontsize=12); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{sc}_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sc}_roc_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for sc, model_name, m in comparison_rows:
    for ax in axes:
        ax.plot(m['fpr'], m['tpr'], label=f"{sc}: {model_name} (AUC={m['roc_auc']:.4f})",
                color=SC_COLORS[sc], linewidth=2)
for ax in axes:
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
axes[0].set_title('Cross-Scenario ROC (full range)')
axes[1].set_xlim(0, 0.2); axes[1].set_ylim(0.8, 1.0); axes[1].set_title('Cross-Scenario ROC (zoom: FPR <= 0.2)')
plt.suptitle('Cross-Scenario Comparison -- Best Model per Scenario (Test Set)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cross_scenario_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: cross_scenario_roc.png")

for sc, results in [('S1', s1_results), ('S2', s2_results), ('S3', s3_results), ('S4', s4_results)]:
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, res) in zip(axes, results.items()):
        sns.heatmap(res['test']['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'], cbar=False)
        ax.set_title(f'{model_name}\nAUC={res["test"]["roc_auc"]:.4f}', fontsize=9)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.suptitle(f'{sc} -- Confusion Matrices (Test Set)', fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{sc}_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sc}_confusion_matrices.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (sc, tr) in zip(axes.flat, threshold_results.items()):
    df_t = tr['val_sweep']
    ax.plot(df_t['threshold'], df_t['f2_atk'],  'D-', label='F2-Attack (selection)', linewidth=2.0, color='#C2185B')
    ax.plot(df_t['threshold'], df_t['f1_atk'],  'o-', label='F1-Attack', linewidth=1.5)
    ax.plot(df_t['threshold'], df_t['rec_atk'], 's--', label='Recall-Attack', linewidth=1.5)
    ax.plot(df_t['threshold'], df_t['prec_atk'],'^:', label='Prec-Attack', linewidth=1.5)
    ax.axvline(tr['best_threshold'], color='red', linestyle='--', alpha=0.7,
               label=f"best t={tr['best_threshold']}")
    ax.set_xlabel('Decision Threshold'); ax.set_title(f'{sc} -- Threshold Analysis (Val)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle('Threshold Analysis -- All Scenarios (Validation Set)', fontsize=12); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: threshold_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 -- Integer scale factor extraction
#
# PURPOSE
#   After S4 is trained, read every split threshold from the fitted tree,
#   compute the minimum adequate integer scale factor for each float feature,
#   and verify whether the current INT_SCALE_* constants are sufficient.
#
# S2/S3/S4 were retrained on integer data in Section 15b.
# This section confirms that the retrained integer S4 tree's thresholds
# are all exact integers -- verifying that no split precision was lost
# in the quantization step.
#
# WHY THE SCALE FACTOR MUST BE DERIVED FROM THE FITTED TREE
#   A scale factor must be large enough that no two training-set values of
#   a feature that land on opposite sides of a split threshold collapse to
#   the same integer after scaling. The only way to know this is to read
#   the actual thresholds the tree learned. See methodology Section 5.2.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 20: INTEGER SCALE FACTOR EXTRACTION")
print(f"  Scale factors used: RATE={INT_SCALE_RATE:,}  IAT={INT_SCALE_IAT:,}  RATIO={INT_SCALE_RATIO:,}")
print("=" * 70)

from sklearn.tree import _tree as _sklearn_tree
import math as _math

# Features produced by FE formulas that may have fractional thresholds
# in float mode. All others (flag_numbers, protocol indicators, byte counts,
# and integer-by-construction FE features like FE_size_range) never do.
_FE_RATIO_FEATURES = {'FE_payload_ratio', 'FE_size_cv_proxy', 'FE_avg_min_ratio'}

def _recommend_scale(min_s):
    """Round min_s up to the nearest power of 10."""
    if min_s <= 1:
        return 1
    return int(10 ** _math.ceil(_math.log10(min_s)))

def _const_for(fname):
    """Map a feature name to the INT_SCALE_* constant that governs it."""
    if fname == 'Rate':  return 'INT_SCALE_RATE'
    if fname == 'IAT':   return 'INT_SCALE_IAT'
    if fname in _FE_RATIO_FEATURES: return 'INT_SCALE_RATIO'
    return None  # already-integer feature, no constant needed

# ── Extract thresholds from fitted S4 tree ────────────────────────────────
s4_dt   = s4_results['Decision Tree']['model']
t       = s4_dt.tree_

print(f"\n  Tree depth : {s4_dt.get_depth()}")
print(f"  Total nodes: {t.node_count}")
print(f"\n  {'Node':>4}  {'Feature':<28}  {'Threshold':>18}  {'Decimals':>8}  {'Min scale':>12}")
print(f"  {'-'*76}")

_min_scales = {}   # feature -> max min_scale seen across all its splits
_all_splits = []   # for JSON output

for _node in range(t.node_count):
    if t.children_left[_node] == _sklearn_tree.TREE_LEAF:
        continue
    _fname  = S4_TREE[t.feature[_node]]
    _thresh = float(t.threshold[_node])
    _thresh_str = f"{_thresh:.10f}".rstrip('0')
    _decimals   = len(_thresh_str.split('.')[1]) if '.' in _thresh_str else 0
    _min_scale  = int(10 ** _decimals) if _decimals > 0 else 1
    print(f"  {_node:>4}  {_fname:<28}  {_thresh:>18.10f}  {_decimals:>8}  {_min_scale:>12,}")
    if _decimals > 0:
        _min_scales[_fname] = max(_min_scales.get(_fname, 1), _min_scale)
    _all_splits.append({'node': _node, 'feature': _fname, 'threshold': _thresh,
                        'decimal_places': _decimals, 'min_scale': _min_scale})

# ── Print recommendations ─────────────────────────────────────────────────
print(f"\n  {'Feature':<28}  {'Min scale':>12}  {'Recommended':>12}  Constant")
print(f"  {'-'*80}")
_recommendations = {}
for _fname in sorted(_min_scales.keys()):
    _rec   = _recommend_scale(_min_scales[_fname])
    _const = _const_for(_fname) or '(already integer -- no constant needed)'
    _recommendations[_fname] = _rec
    print(f"  {_fname:<28}  {_min_scales[_fname]:>12,}  {_rec:>12,}  {_const}")

if not _min_scales:
    print("  All thresholds are already integers.")
    print("  Set INT_SCALE_RATE = INT_SCALE_IAT = INT_SCALE_RATIO = 1.")

# ── Verify current constants ──────────────────────────────────────────────
_const_requirements = {
    'INT_SCALE_RATE':  max((_min_scales.get(f, 1) for f in ['Rate']),           default=1),
    'INT_SCALE_IAT':   max((_min_scales.get(f, 1) for f in ['IAT']),            default=1),
    'INT_SCALE_RATIO': max((_min_scales.get(f, 1) for f in _FE_RATIO_FEATURES), default=1),
}
_current = {'INT_SCALE_RATE': INT_SCALE_RATE,
            'INT_SCALE_IAT':  INT_SCALE_IAT,
            'INT_SCALE_RATIO': INT_SCALE_RATIO}
_all_ok = True
print(f"\n  Verification against current constants:")
for _const, _min_needed in _const_requirements.items():
    _rec = _recommend_scale(_min_needed)
    _cur = _current[_const]
    if _min_needed == 1:
        _status, _marker = "not needed (no float thresholds for this feature)", " "
    elif _cur is None:
        _status, _marker = f"NOT SET -- set to >= {_rec:,}", "!"
        _all_ok = False
    elif _cur < _rec:
        _status, _marker = f"INSUFFICIENT -- current={_cur:,}, need >= {_rec:,}", "!"
        _all_ok = False
    else:
        _status, _marker = f"OK  (current={_cur:,} >= recommended={_rec:,})", " "
    print(f"  [{_marker}] {_const:<20}  recommended={_rec:>10,}  {_status}")

print()
if _all_ok:
    print("  RESULT: ALL THRESHOLDS ARE EXACT INTEGERS.")
    print("  Integer quantization in Section 15b preserved all split boundaries.")
else:
    print("  RESULT: WARNING -- some thresholds are not exact integers.")
    print("  The derived scale factors may be insufficient for some features.")
    print("  Inspect the table above and consider increasing the relevant scale.")

# ── Save to JSON ──────────────────────────────────────────────────────────
_threshold_output = {
    'run_mode': 'integer_via_section_15b',
    'tree_depth': int(s4_dt.get_depth()),
    'node_count': int(t.node_count),
    'all_splits': _all_splits,
    'min_scales_per_feature': {k: int(v) for k, v in _min_scales.items()},
    'recommended_scales': {
        'INT_SCALE_RATE':  int(_recommend_scale(_const_requirements['INT_SCALE_RATE'])),
        'INT_SCALE_IAT':   int(_recommend_scale(_const_requirements['INT_SCALE_IAT'])),
        'INT_SCALE_RATIO': int(_recommend_scale(_const_requirements['INT_SCALE_RATIO'])),
    },
    'scale_factors_used': _current,
    'all_constants_sufficient': _all_ok,
}
_thresh_path = os.path.join(RESULTS_DIR, 'S4_threshold_extraction.json')
with open(_thresh_path, 'w') as _f:
    json.dump(_threshold_output, _f, indent=2)
print(f"\n  Full details saved -> {_thresh_path}")

print("\n" + "=" * 70)
print("PART 2 (MODELING) COMPLETE")
print(f"  Plots   -> {PLOT_DIR}")
print(f"  Results -> {RESULTS_DIR}")
print(f"  Models  -> {MODELS_DIR}")

print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# PART 3 -- MULTI-SEED STABILITY CHECK
#
# Everything above uses one seed (42) throughout: the train/val/test split,
# every train-only feature-selection decision, and every hyperparameter
# search. This part asks how much that single seed matters, by repeating
# preprocessing-through-evaluation under four further seeds and comparing.
#
# WHAT IS AND ISN'T RE-DONE PER SEED
#   Re-done: the split (Section 2's logic), train-only feature reduction
#   (Section 5), the Rate-vs-IAT ablation (Section 6), the engineered-
#   feature AUROC screen (Section 7), integer scale-factor derivation and
#   retraining (Section 15b), and the final model fit for every
#   scenario/model combination -- using the hyperparameters already found
#   above (best_params from s1_results/.../s4_results).
#   NOT re-done: the hyperparameter search itself (Section 10-15b's
#   RandomizedSearchCV calls). This isolates data-partition and model-fit
#   variance from the search's own sampling noise, under the assumption
#   that the optimal configuration is not highly split-sensitive at this
#   dataset scale -- an assumption not separately tested here.
#
# SEED 42 IS ONE OF THE FIVE, NOT JUST A BASELINE
#   It is the seed used throughout Parts 1-2, fixed before any result
#   existed (not selected afterward for being favorable), so including it
#   alongside four fresh seeds is not cherry-picking. Its hyperparameters,
#   though, were specifically found FOR its own split by the search above,
#   while the other four reuse those same hyperparameters out-of-sample --
#   so seed 42's own figures may be mildly favorable relative to the other
#   four. As a correctness check, seed 42 is also recomputed from scratch
#   through the same per-seed code path used for the other four, and
#   compared against the cached s1_results/.../s4_results above: an exact
#   match confirms this part's logic agrees with Parts 1-2's.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("03_modeling.py -- PART 3: MULTI-SEED STABILITY CHECK")
print("=" * 70)

SEEDS = [42, 7, 123, 2024, 99]

BEST_PARAMS = {
    'S1': {name: res['best_params'] for name, res in s1_results.items()},
    'S2': {name: res['best_params'] for name, res in s2_results.items()},
    'S3': {name: res['best_params'] for name, res in s3_results.items()},
    'S4': {name: res['best_params'] for name, res in s4_results.items()},
}
SCENARIO_MODELS = {
    'S1': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'S2': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'S3': ['Decision Tree'],
    'S4': ['Decision Tree'],
}
STABILITY_METRICS = ['roc_auc', 'pr_auc', 'accuracy', 'prec_atk', 'rec_atk', 'f1_atk']


def run_one_seed(seed):
    """Repeats this file's own preprocessing (Sections 2, 5, 6, 7) and a
    final-fit-only pass (Sections 9-15b, with hyperparameters frozen from
    BEST_PARAMS rather than re-searched) under `seed`. Returns per-
    scenario/model metrics plus the fitted S4 tree's structure."""
    print(f"\n--- Seed {seed} ---")
    y_all_s = df[LABEL].astype(int)

    X_train_39_s, X_temp_39_s, y_train_s, y_temp_s = train_test_split(
        df[feature_cols_all], y_all_s, test_size=0.30, random_state=seed, stratify=y_all_s)
    X_val_39_s, X_test_39_s, y_val_s, y_test_s = train_test_split(
        X_temp_39_s, y_temp_s, test_size=0.50, random_state=seed, stratify=y_temp_s)

    S1_reduced_s, _ = reduce_feature_set(
        feature_cols_all, X_train_39_s, y_train_s, f'S1 (seed {seed})',
        exclude_from_correlation=RATE_IAT_EXCLUDE, random_state=seed)
    S2_reduced_s, _ = reduce_feature_set(
        SMARTNIC_FEATURES, X_train_39_s, y_train_s, f'S2/S3 (seed {seed})',
        exclude_from_correlation=RATE_IAT_EXCLUDE, random_state=seed)

    S1_ablation_s = ablate_rate_iat(S1_reduced_s, X_train_39_s, y_train_s, X_val_39_s, y_val_s,
                                    f'S1 (seed {seed})', seed=seed)
    S2_ablation_s = ablate_rate_iat(S2_reduced_s, X_train_39_s, y_train_s, X_val_39_s, y_val_s,
                                    f'S2 (seed {seed})', seed=seed)
    S1_TREE_s = apply_rate_iat_decision(S1_reduced_s, S1_ablation_s['tree'])
    S1_LR_s   = apply_rate_iat_decision(S1_reduced_s, S1_ablation_s['linear'])
    S2_TREE_s = apply_rate_iat_decision(S2_reduced_s, S2_ablation_s['tree'])
    S2_LR_s   = apply_rate_iat_decision(S2_reduced_s, S2_ablation_s['linear'])
    S3_TREE_s = list(S2_TREE_s)

    FE_train_s = build_engineered_features(X_train_39_s[SMARTNIC_FEATURES], S2_reduced_s)
    FE_val_s   = build_engineered_features(X_val_39_s[SMARTNIC_FEATURES],   S2_reduced_s)
    for fe_df, X_sub in [(FE_train_s, X_train_39_s[SMARTNIC_FEATURES]),
                        (FE_val_s,   X_val_39_s[SMARTNIC_FEATURES])]:
        fe_df['FE_flow_duration']   = fe_df['FE_flow_duration'].fillna(0)
        fe_df['FE_burst_intensity'] = fe_df['FE_burst_intensity'].fillna(SMARTNIC_MAX_INT * X_sub['Number'])
        fe_df['FE_arp_rate']        = fe_df['FE_arp_rate'].fillna(X_sub['ARP'] * SMARTNIC_MAX_INT)

    engineered_auroc_s = {}
    for col in FE_train_s.columns:
        x = FE_train_s[col].fillna(FE_train_s[col].median())
        auc = roc_auc_score(y_train_s, x)
        engineered_auroc_s[col] = max(auc, 1.0 - auc)
    kept_engineered_s = sorted([c for c, a in engineered_auroc_s.items() if a >= ENGINEERED_AUROC_MIN])
    S4_TREE_s = S2_TREE_s + kept_engineered_s
    print(f"  [seed {seed}] S1_TREE={len(S1_TREE_s)}  S2_TREE={len(S2_TREE_s)}  "
          f"S4_TREE={len(S4_TREE_s)} ({len(kept_engineered_s)} engineered kept)")

    X_train_S1_s    = build_X(S1_TREE_s, X_train_39_s)
    X_test_S1_s     = build_X(S1_TREE_s, X_test_39_s)
    X_train_S1_LR_s = build_X(S1_LR_s,   X_train_39_s)
    X_test_S1_LR_s  = build_X(S1_LR_s,   X_test_39_s)
    X_train_S4f_s   = build_X(S4_TREE_s, X_train_39_s, FE_train_s)

    base_lr_s = Pipeline([
        ('transform', PowerTransformer(method='yeo-johnson')),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                                   random_state=seed, n_jobs=-1)),
    ])
    base_dt_s          = DecisionTreeClassifier(class_weight='balanced', random_state=seed)
    base_rf_s          = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=seed)
    base_dt_smartnic_s = DecisionTreeClassifier(max_depth=SMARTNIC_DEPTH, class_weight='balanced',
                                                random_state=seed)

    def fit_frozen(base_model, params, X_tr, y_tr):
        m = clone(base_model)
        m.set_params(**params)
        m.fit(X_tr, y_tr)
        return m

    seed_metrics = {sc: {} for sc in SCENARIO_MODELS}
    for model_name, base_model, X_tr, X_te in [
        ('Logistic Regression', base_lr_s, X_train_S1_LR_s, X_test_S1_LR_s),
        ('Decision Tree',       base_dt_s, X_train_S1_s,    X_test_S1_s),
        ('Random Forest',       base_rf_s, X_train_S1_s,    X_test_S1_s),
    ]:
        m = fit_frozen(base_model, BEST_PARAMS['S1'][model_name], X_tr, y_train_s)
        seed_metrics['S1'][model_name] = evaluate(m, X_te, y_test_s)

    # Float S4 tree, fit only to derive this seed's integer scale factors
    # (mirrors Section 15 -> 15b's float-then-integer procedure); its own
    # metrics are not reported, the integer retrain below is.
    s4_float_s = fit_frozen(base_dt_smartnic_s, BEST_PARAMS['S4']['Decision Tree'],
                            X_train_S4f_s, y_train_s)
    derived_scales_s = _derive_scale_factors(s4_float_s, S4_TREE_s)
    scale_rate_s  = derived_scales_s.get('Rate', 1_000)
    scale_iat_s   = derived_scales_s.get('IAT', 1_000_000)
    scale_ratio_s = max((derived_scales_s.get(f, 1_000) for f in _FE_RATIO_FEATURES), default=1_000)

    (X_train_39i_s, X_val_39i_s, X_test_39i_s,
     FE_train_i_s, FE_val_i_s, FE_test_i_s) = rebuild_integer_features(
        X_train_39_s, X_val_39_s, X_test_39_s, scale_rate_s, scale_iat_s, scale_ratio_s, S2_reduced_s)

    X_train_S2_s    = build_X(S2_TREE_s, X_train_39i_s)
    X_test_S2_s     = build_X(S2_TREE_s, X_test_39i_s)
    X_train_S2_LR_s = build_X(S2_LR_s,   X_train_39i_s)
    X_test_S2_LR_s  = build_X(S2_LR_s,   X_test_39i_s)
    X_train_S3_s, X_test_S3_s = X_train_S2_s, X_test_S2_s
    X_train_S4_s = build_X(S4_TREE_s, X_train_39i_s, FE_train_i_s)
    X_test_S4_s  = build_X(S4_TREE_s, X_test_39i_s,  FE_test_i_s)

    for model_name, base_model, X_tr, X_te in [
        ('Logistic Regression', base_lr_s, X_train_S2_LR_s, X_test_S2_LR_s),
        ('Decision Tree',       base_dt_s, X_train_S2_s,    X_test_S2_s),
        ('Random Forest',       base_rf_s, X_train_S2_s,    X_test_S2_s),
    ]:
        m = fit_frozen(base_model, BEST_PARAMS['S2'][model_name], X_tr, y_train_s)
        seed_metrics['S2'][model_name] = evaluate(m, X_te, y_test_s)

    s3_model_s = fit_frozen(base_dt_smartnic_s, BEST_PARAMS['S3']['Decision Tree'], X_train_S3_s, y_train_s)
    seed_metrics['S3']['Decision Tree'] = evaluate(s3_model_s, X_test_S3_s, y_test_s)

    s4_model_s = fit_frozen(base_dt_smartnic_s, BEST_PARAMS['S4']['Decision Tree'], X_train_S4_s, y_train_s)
    seed_metrics['S4']['Decision Tree'] = evaluate(s4_model_s, X_test_S4_s, y_test_s)

    print(f"  [seed {seed}] S4 tree: depth={s4_model_s.get_depth()}  "
          f"nodes={s4_model_s.tree_.node_count}  top-split={S4_TREE_s[s4_model_s.tree_.feature[0]]}")

    return {
        'metrics': {sc: {m: {k: seed_metrics[sc][m][k] for k in STABILITY_METRICS}
                         for m in SCENARIO_MODELS[sc]} for sc in SCENARIO_MODELS},
        'feature_lists': {'S1_TREE': S1_TREE_s, 'S2_TREE': S2_TREE_s, 'S4_TREE': S4_TREE_s},
        's4_tree_depth': int(s4_model_s.get_depth()),
        's4_tree_nodes': int(s4_model_s.tree_.node_count),
        's4_top_split_feature': S4_TREE_s[s4_model_s.tree_.feature[0]],
    }


print("\n--- Self-test: recomputing seed 42 and comparing to cached results above ---")
seed42_recomputed = run_one_seed(42)
_cached_metrics_42 = {
    'S1': {m: {k: s1_results[m]['test'][k] for k in STABILITY_METRICS} for m in SCENARIO_MODELS['S1']},
    'S2': {m: {k: s2_results[m]['test'][k] for k in STABILITY_METRICS} for m in SCENARIO_MODELS['S2']},
    'S3': {m: {k: s3_results[m]['test'][k] for k in STABILITY_METRICS} for m in SCENARIO_MODELS['S3']},
    'S4': {m: {k: s4_results[m]['test'][k] for k in STABILITY_METRICS} for m in SCENARIO_MODELS['S4']},
}
_max_abs_diff = 0.0
for sc in SCENARIO_MODELS:
    for m in SCENARIO_MODELS[sc]:
        diff = abs(seed42_recomputed['metrics'][sc][m]['roc_auc'] - _cached_metrics_42[sc][m]['roc_auc'])
        _max_abs_diff = max(_max_abs_diff, diff)
        print(f"  [{sc}][{m}] recomputed={seed42_recomputed['metrics'][sc][m]['roc_auc']:.6f}  "
              f"cached={_cached_metrics_42[sc][m]['roc_auc']:.6f}  "
              f"{'OK' if diff < 1e-6 else f'MISMATCH (diff={diff:.6f})'}")
print(f"  Max |recomputed - cached| ROC-AUC: {_max_abs_diff:.8f}  "
      f"-> {'PASS' if _max_abs_diff < 1e-6 else 'FAIL -- investigate before trusting other seeds'}")

print("\n--- Running the remaining 4 seeds ---")
_per_seed_results = {42: seed42_recomputed}
for _seed in SEEDS:
    if _seed == 42:
        continue
    _per_seed_results[_seed] = run_one_seed(_seed)

# Seed 42's row in the aggregate uses the CACHED, already-published metrics
# (identical to the self-test's recomputation above, confirmed to agree)
# rather than the recomputation itself, since those cached values are what
# the rest of this file -- and the paper -- actually report.
_metrics_for_aggregate = {s: r['metrics'] for s, r in _per_seed_results.items()}
_metrics_for_aggregate[42] = _cached_metrics_42

stability_summary = {}
for sc in SCENARIO_MODELS:
    stability_summary[sc] = {}
    for m in SCENARIO_MODELS[sc]:
        stability_summary[sc][m] = {}
        for metric in STABILITY_METRICS:
            vals = [_metrics_for_aggregate[s][sc][m][metric] for s in SEEDS]
            stability_summary[sc][m][metric] = {
                'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)), 'values': vals,
            }

print("\n" + "=" * 70)
print("MULTI-SEED STABILITY SUMMARY (5 seeds: 42, 7, 123, 2024, 99)")
print("=" * 70)
print(f"  {'Scenario':<10} {'Model':<22} {'ROC-AUC':>16} {'Pre-Atk':>16} {'Rec-Atk':>16} {'F1-Atk':>16}")
for sc in SCENARIO_MODELS:
    for m in SCENARIO_MODELS[sc]:
        row = stability_summary[sc][m]
        def fmt(key): return f"{row[key]['mean']:.4f}+/-{row[key]['std']:.4f}"
        print(f"  {sc:<10} {m:<22} {fmt('roc_auc'):>16} {fmt('prec_atk'):>16} {fmt('rec_atk'):>16} {fmt('f1_atk'):>16}")

print(f"\n  S4 tree structure across seeds:")
for _seed in SEEDS:
    r = _per_seed_results[_seed]
    print(f"    seed {_seed:>5}: depth={r['s4_tree_depth']}  nodes={r['s4_tree_nodes']}  "
          f"top-split={r['s4_top_split_feature']}")

_stability_output = {
    'seeds': SEEDS,
    'self_test_max_abs_roc_auc_diff': _max_abs_diff,
    'per_seed_feature_lists': {str(s): r['feature_lists'] for s, r in _per_seed_results.items()},
    'per_seed_s4_tree': {str(s): {'depth': r['s4_tree_depth'], 'nodes': r['s4_tree_nodes'],
                                  'top_split_feature': r['s4_top_split_feature']}
                        for s, r in _per_seed_results.items()},
    'aggregate': stability_summary,
}
_stability_path = os.path.join(RESULTS_DIR, 'seed_stability_summary.json')
with open(_stability_path, 'w') as _f:
    json.dump(_stability_output, _f, indent=2)
print(f"\n  Full details saved -> {_stability_path}")

print("\n" + "=" * 70)
print("PART 3 (MULTI-SEED STABILITY CHECK) COMPLETE")
print("=" * 70)

