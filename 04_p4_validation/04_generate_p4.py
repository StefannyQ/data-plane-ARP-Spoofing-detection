#!/usr/bin/env python3
"""
04_generate_p4.py -- Generate P4 programs, test harnesses, and a resource
summary for the S3 and S4 Decision Trees

PURPOSE
-------
Reads each scenario's integer-trained Decision Tree and the preprocessing
summary, then generates, for S3 and S4 in turn:

  p4_validation/
    arp_detector_s3.p4, table_entries_s3.txt, test_harness_s3.py,
        flow_test_vectors_s3.csv, README_s3.md        -- S3 (no feature
        engineering; the scenario recommended as the operational default)
    arp_detector.p4,    table_entries.txt,    test_harness.py,
        flow_test_vectors.csv,    README.md            -- S4 (adds the
        two engineered features its tree actually splits on)

Both scenarios share one test split and one set of integer scale factors
(derived once from the fitted S4 tree in 03_modeling.py Section 15b and
reused for S2/S3/S4 alike), so the dataset is loaded and quantized once
and reused for both. Each scenario gets its own custom ethertype pair so
the two programs' traffic can never be cross-confused on the same VM.

After both are generated, a final section reads the resulting .p4 files
back and reports each one's resource footprint -- program size, tables,
actions, header width, and register count -- directly from the generated
artifact rather than from an estimate, and saves it to
p4_resource_summary.json alongside the modeling results.

WHAT THE GENERATED P4 PROGRAMS DO
----------------------------------
Each test-set flow is represented as a custom Ethernet frame carrying the
scenario's feature values in a custom header. Each P4 program:
  1. Parses the custom header
  2. Reads any FE_* engineered feature values directly from the header --
     they are precomputed in Python using the same integer arithmetic as
     training, not computed in the data plane (see the design note above
     tree_to_p4_apply() for why)
  3. Applies the fitted decision tree as nested if/else comparisons inside
     a single action, reached via one table keyed on a constant ethertype
     match (one table total, regardless of tree depth)
  4. Sets metadata.verdict = 0 (Benign) or 1 (Attack)
  5. Reflects the packet back out port 1 (veth1) with verdict in a result
     header -- a fixed port, not the ingress port the packet arrived on

The test harness sends each flow vector as one crafted packet, reads the
reflected verdict, and compares it to the Python model's prediction on the
same row. A 100% match rate means the P4 encoding is faithful.

USAGE
-----
  python 04_generate_p4.py

Then copy the p4_validation/ folder to your P4 VM and follow README_s3.md
and README.md for the respective scenario.

PATHS  (match 03_modeling.py)
-----
"""

import os, sys, io, json, pickle, re, textwrap
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths (must match 03_modeling.py) ────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, 'modeling', 'results')
MODELS_DIR  = os.path.join(BASE, 'modeling', 'models')
DATA_PATH   = os.path.join(BASE, 'data', 'final_dataset.csv')
OUT_DIR     = os.path.join(BASE, 'p4_validation')

SUMMARY_PATH    = os.path.join(RESULTS_DIR, 'preprocessing_summary.json')
THRESHOLD_PATH  = os.path.join(RESULTS_DIR, 'S4_threshold_extraction.json')

# Number of test-set rows to include in each scenario's test vectors file.
# Set to None to use the full test set (~210,863 rows).
# Set to an integer (e.g. 2000) for a faster stratified sample.
N_TEST_VECTORS = None  # None = full test set
RANDOM_STATE   = 42

SMARTNIC_MAX_INT = 2**32 - 1

print("=" * 70)
print("04_generate_p4.py -- P4 Program Generator (S3 and S4)")
print("=" * 70)

for path, label in [(SUMMARY_PATH, 'preprocessing_summary.json'),
                    (THRESHOLD_PATH, 'S4_threshold_extraction.json')]:
    if not os.path.exists(path):
        sys.exit(f"\nERROR: {path} not found.\nRun 03_modeling.py first.")

with open(SUMMARY_PATH) as f:
    summary = json.load(f)
with open(THRESHOLD_PATH) as f:
    _threshold_info = json.load(f)

# Scale factors are derived once, from the fitted S4 tree (03_modeling.py
# Section 15b), and reused as-is for S2, S3, and S4 -- all three share one
# integer-quantized feature matrix, so there is nothing scenario-specific
# to derive here.
_scale_factors  = _threshold_info['scale_factors_used']
INT_SCALE_RATE  = _scale_factors['INT_SCALE_RATE']
INT_SCALE_IAT   = _scale_factors['INT_SCALE_IAT']
INT_SCALE_RATIO = _scale_factors['INT_SCALE_RATIO']

# S2_reduced -- needed to restrict FE_protocol_diversity/FE_no_app_layer to
# exactly the columns that survived Section 5's train-only statistical
# reduction, matching build_engineered_features() in 03_modeling.py exactly.
S2_reduced = summary['S2_diagnostics']['reduced']

print(f"\n  Scale factors: RATE={INT_SCALE_RATE:,}  IAT={INT_SCALE_IAT:,}  RATIO={INT_SCALE_RATIO:,}")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared dataset: one test split, quantized once, reused by every scenario ──
df = pd.read_csv(DATA_PATH)
LABEL = 'label'
feature_cols_all = [c for c in df.columns if c != LABEL]
y_all = df[LABEL].astype(int)

_, X_temp, _, y_temp = train_test_split(
    df[feature_cols_all], y_all, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y_all
)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)


def quantize(X):
    """Mirrors apply_integer_quantization() in 03_modeling.py term for
    term, so these test vectors are built exactly the way every scenario's
    integer model was trained."""
    X = X.copy()
    sentinel = X['Rate'] == SMARTNIC_MAX_INT
    X['Rate'] = (X['Rate'].where(sentinel, X['Rate'] * INT_SCALE_RATE)).astype(np.int64)
    X.loc[sentinel, 'Rate'] = SMARTNIC_MAX_INT
    X['IAT'] = (X['IAT'] * INT_SCALE_IAT).astype(np.int64)
    for col in ['Min', 'Max', 'Header_Length', 'AVG', 'Tot sum', 'Tot size',
                'Variance', 'Number', 'ARP', 'TCP', 'UDP', 'ICMP', 'IGMP',
                'HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP',
                'syn_flag_number', 'ack_flag_number', 'fin_flag_number',
                'rst_flag_number', 'psh_flag_number',
                'ece_flag_number', 'cwr_flag_number', 'Protocol Type', 'Time_To_Live']:
        if col in X.columns:
            X[col] = X[col].astype(np.int64)
    return X


X_test_int = quantize(X_test)


def build_fe(X):
    """Mirrors build_engineered_features(..., integer_mode=True) plus the
    caller-side sentinel fillna in rebuild_integer_features(), both in
    03_modeling.py, term for term. Only called for scenarios whose tree
    actually uses an FE_* feature (S4); S3 has none, so it never calls
    this at all."""
    fe = pd.DataFrame(index=X.index)
    safe_rate = X['Rate'].replace(SMARTNIC_MAX_INT, np.nan)
    fe['FE_burst_intensity']    = (safe_rate * X['Number']).fillna(
                                      np.int64(SMARTNIC_MAX_INT) * X['Number']).astype(np.int64)
    fe['FE_flow_duration']      = (X['Number'] // safe_rate.replace(0, np.nan)).fillna(0).astype(np.int64)
    fe['FE_size_cv_proxy']      = (X['Variance'] // (X['AVG'] + 1)).astype(np.int64)
    proto_cols_full = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP', 'TCP', 'UDP', 'ARP', 'ICMP', 'IGMP']
    proto_cols = [c for c in proto_cols_full if c in S2_reduced]
    fe['FE_protocol_diversity'] = (X[proto_cols] > 0).sum(axis=1).astype(np.int64)
    fe['FE_size_range']         = (X['Max'] - X['Min']).astype(np.int64)
    fe['FE_payload_ratio']      = (((X['Tot size'] - X['Header_Length']) * INT_SCALE_RATIO)
                                    // (X['Tot size'] + 1)).astype(np.int64)
    fe['FE_header_ratio']       = ((X['Header_Length'] * 100) // (X['Tot size'] + 1)).astype(np.int64)
    app_layer_cols_full = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP']
    app_cols = [c for c in app_layer_cols_full if c in S2_reduced]
    fe['FE_no_app_layer']       = ((X[app_cols].sum(axis=1)) == 0).astype(np.int64)
    fe['FE_avg_min_ratio']      = (X['AVG'] // (X['Min'] + 1)).astype(np.int64)
    fe['FE_arp_rate']           = (X['ARP'] * safe_rate).fillna(
                                      X['ARP'] * np.int64(SMARTNIC_MAX_INT)).astype(np.int64)
    return fe


# ── Helper: walk a fitted tree into a flat node list ──────────────────────────
def get_tree_nodes(tree_model, feature_names):
    """Return list of dicts describing every node in the tree."""
    t = tree_model.tree_
    nodes = []
    def walk(node, depth=0, parent=None, branch=None):
        is_leaf = t.children_left[node] == _tree.TREE_LEAF
        class_counts = t.value[node][0]
        predicted_class = int(np.argmax(class_counts))
        nodes.append({
            'id': node,
            'depth': depth,
            'parent': parent,
            'branch': branch,           # 'left' (<= threshold) or 'right' (>)
            'is_leaf': is_leaf,
            'feature': feature_names[t.feature[node]] if not is_leaf else None,
            'threshold': int(t.threshold[node]) if not is_leaf else None,
            'left_child':  int(t.children_left[node])  if not is_leaf else None,
            'right_child': int(t.children_right[node]) if not is_leaf else None,
            'predicted_class': predicted_class,
            'class_label': 'Benign' if predicted_class == 0 else 'Attack',
            'n_samples': int(t.n_node_samples[node]),
        })
        if not is_leaf:
            walk(t.children_left[node],  depth+1, node, 'left')
            walk(t.children_right[node], depth+1, node, 'right')
    walk(0)
    return nodes


# DESIGN NOTE -- why P4 does not compute FE_* itself
#   For S4, only 2 of its 8 FE_* features (FE_burst_intensity,
#   FE_payload_ratio) are ever used by the tree's splits, and computing
#   them in the data plane would require source raw columns (Number, Tot
#   size) that are not part of the tree's own feature list at all -- they
#   were absorbed into FE_* during feature engineering and never sent as
#   their own header fields. Rather than widen the header just to
#   re-derive values Python has already computed correctly (and verified
#   against the cached model), both scenarios send every FE_* value they
#   use precomputed, exactly as they appear in their flow_test_vectors
#   CSV. P4 only does the match-action comparisons -- which is the actual
#   thing this experiment validates: does the tree's topology, re-expressed
#   as P4 if/else logic, reproduce the trained model's verdicts. S3 has no
#   engineered features at all, so this note is moot for it -- every one
#   of its header fields is a directly observable raw feature.
def tree_to_p4_apply(nodes):
    node_map = {n['id']: n for n in nodes}

    def field_ref(fname):
        clean = fname.replace(' ', '_').replace('-', '_')
        return f"hdr.flow.{clean}"

    def emit_node(node_id, depth=0):
        n = node_map[node_id]
        pad = '    ' * depth
        if n['is_leaf']:
            return f"{pad}meta.verdict = {n['predicted_class']};  // {n['class_label']}"
        f = field_ref(n['feature'])
        left  = emit_node(n['left_child'],  depth+1)
        right = emit_node(n['right_child'], depth+1)
        return (
            f"{pad}if ({f} <= {n['threshold']}) {{\n"
            f"{left}\n"
            f"{pad}}} else {{\n"
            f"{right}\n"
            f"{pad}}}"
        )

    return emit_node(0)


def generate_scenario_p4(scenario, results_path, model_path, file_suffix,
                         ethertype_flow, ethertype_result):
    """Generates arp_detector<file_suffix>.p4, table_entries<file_suffix>.txt,
    flow_test_vectors<file_suffix>.csv, test_harness<file_suffix>.py, and
    README<file_suffix or nothing>.md for one fitted scenario tree, into
    OUT_DIR. Returns a dict of resource-relevant facts (tree depth, node
    counts, header field count) consumed by the resource-summary section
    at the end of this script.
    """
    print("\n" + "=" * 70)
    print(f"SCENARIO {scenario}")
    print("=" * 70)

    if not os.path.exists(results_path):
        sys.exit(f"\nERROR: {results_path} not found.\nRun 03_modeling.py first.")
    if not os.path.exists(model_path):
        sys.exit(f"\nERROR: {model_path} not found.\nRun 03_modeling.py first.")

    # save_results() in 03_modeling.py strips the fitted model object out
    # of the slim *_results.pkl and joblib-dumps it separately -- the
    # model is never inside <scenario>_results.pkl itself.
    tree_model = joblib.load(model_path)
    t          = tree_model.tree_
    tree_feature_list = summary['feature_lists'][f'{scenario}_TREE']

    print(f"\n  Tree depth : {tree_model.get_depth()}")
    print(f"  Nodes      : {t.node_count}")
    print(f"  Features   : {len(tree_feature_list)}  ->  {tree_feature_list}")

    nodes = get_tree_nodes(tree_model, tree_feature_list)
    leaves = [n for n in nodes if n['is_leaf']]
    internals = [n for n in nodes if not n['is_leaf']]

    print(f"\n  Internal nodes: {len(internals)}  |  Leaves: {len(leaves)}")
    print(f"\n  Tree structure:")
    for n in nodes:
        indent = '  ' * n['depth']
        if n['is_leaf']:
            print(f"    {indent}[Leaf {n['id']}] -> {n['class_label']} (n={n['n_samples']:,})")
        else:
            print(f"    {indent}[Node {n['id']}] if {n['feature']} <= {n['threshold']:,}: "
                  f"goto {n['left_child']} else goto {n['right_child']}")

    FE_FEATURES  = [f for f in tree_feature_list if f.startswith('FE_')]
    RAW_FEATURES = [f for f in tree_feature_list if not f.startswith('FE_')]
    print(f"\n  Raw features ({len(RAW_FEATURES)}): {RAW_FEATURES}")
    print(f"  FE features  ({len(FE_FEATURES)}):  {FE_FEATURES}")

    # Every field the tree's splits reference -- raw features plus the
    # FE_* features actually used -- gets one bit<64> slot in the header,
    # in this fixed order. The test harness packs its CSV columns in the
    # exact same order, so the two stay in lockstep without separate
    # bookkeeping.
    #
    # bit<64> throughout -- with the derived scale factors (commonly 1e10
    # or more), a scaled Rate value alone can reach ~1e14 and
    # FE_burst_intensity (Rate * Number) can reach ~1e17, both far beyond
    # bit<32>'s ~4.3e9 ceiling. bit<64> removes the overflow/truncation
    # risk regardless of which scale factor a given run derives.
    fe_in_tree = sorted(set(
        n['feature'] for n in internals if n['feature'] and n['feature'].startswith('FE_')
    ))
    HEADER_FIELDS = RAW_FEATURES + fe_in_tree

    raw_header_fields = '\n    '.join(
        f"bit<64> {f.replace(' ','_').replace('-','_').replace('/','_')};" for f in HEADER_FIELDS
    )

    p4_filename = f"arp_detector{file_suffix}.p4"
    p4_program = f"""\
/* {p4_filename}
 * Generated by 04_generate_p4.py from the fitted {scenario} Decision Tree
 * P4_16 -- compiles with p4c, runs on BMv2 simple_switch
 *
 * WHAT THIS DOES
 *   Receives crafted Ethernet frames carrying a custom 'flow' header with
 *   integer-scaled feature values (one frame per flow), all precomputed in
 *   Python -- including any FE_* features the tree actually splits on.
 *   Applies the fitted depth-{tree_model.get_depth()} Decision Tree as nested if/else
 *   logic, sets a verdict (0=Benign, 1=Attack), and reflects the packet
 *   back with a 'result' header prepended.
 *
 * SCALE FACTORS (applied when 04_generate_p4.py builds the test vectors)
 *   Rate : x{INT_SCALE_RATE:,}   IAT : x{INT_SCALE_IAT:,}   Ratios: x{INT_SCALE_RATIO:,}
 *
 * CUSTOM ETHERTYPE
 *   {ethertype_flow:#06x} = flow feature frame (input to classifier)
 *   {ethertype_result:#06x} = result frame (reflected back with verdict)
 */

#include <core.p4>
#include <v1model.p4>

// ── Custom ethertypes ─────────────────────────────────────────────────────────
const bit<16> TYPE_FLOW   = {ethertype_flow:#06x};
const bit<16> TYPE_RESULT = {ethertype_result:#06x};
const bit<32> MAX_INT_SENTINEL = {SMARTNIC_MAX_INT};

// ── Headers ───────────────────────────────────────────────────────────────────
header ethernet_t {{
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}}

// One 64-bit slot per field the tree's splits reference, both precomputed
// in Python exactly as they appear in the test vectors CSV -- P4 does not
// compute FE_* itself (see the design note above tree_to_p4_apply()).
// All values are pre-scaled integers (see scale factors above).
header flow_t {{
    {raw_header_fields}
}}

// Result header: prepended to the reflected packet.
header result_t {{
    bit<8>  verdict;    // 0 = Benign, 1 = Attack
    bit<8>  pad;
    bit<16> reserved;
}}

struct headers {{
    ethernet_t ethernet;
    flow_t     flow;
    result_t   result;
}}

// ── Metadata ──────────────────────────────────────────────────────────────────
struct metadata {{
    bit<8>  verdict;
}}

// ── Parser ────────────────────────────────────────────────────────────────────
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {{
    state start {{
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {{
            TYPE_FLOW: parse_flow;
            default: accept;
        }}
    }}
    state parse_flow {{
        packet.extract(hdr.flow);
        transition accept;
    }}
}}

// ── Checksum verification (none needed for custom headers) ────────────────────
control MyVerifyChecksum(inout headers hdr, inout metadata meta) {{
    apply {{ }}
}}

// ── Ingress ───────────────────────────────────────────────────────────────────
control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {{

    action drop() {{
        mark_to_drop(standard_metadata);
    }}

    action classify() {{
        // ── Apply fitted decision tree (all fields arrive precomputed) ───────
{textwrap.indent(tree_to_p4_apply(nodes), '        ')}

        // ── Add result header and send out port 1 (veth1) ────────────────────
        hdr.result.setValid();
        hdr.result.verdict  = meta.verdict;
        hdr.result.pad      = 0;
        hdr.result.reserved = 0;
        hdr.ethernet.etherType = TYPE_RESULT;

        // Send out port 1 (veth1). Scapy listens on veth1 for replies.
        // Packets sent into veth0 (port 0) by Scapy arrive at BMv2 on port 0.
        // BMv2 sends the result out port 1 (veth1); Scapy sniffs veth1.
        standard_metadata.egress_spec = 1;
    }}

    table flow_classifier {{
        key = {{
            hdr.ethernet.etherType : exact;
        }}
        actions = {{
            classify;
            drop;
        }}
        default_action = drop();
        const entries = {{
            TYPE_FLOW : classify();
        }}
    }}

    apply {{
        if (hdr.ethernet.isValid()) {{
            flow_classifier.apply();
        }}
    }}
}}

// ── Egress (passthrough) ──────────────────────────────────────────────────────
control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {{
    apply {{ }}
}}

// ── Checksum update (none needed) ─────────────────────────────────────────────
control MyComputeChecksum(inout headers hdr, inout metadata meta) {{
    apply {{ }}
}}

// ── Deparser ──────────────────────────────────────────────────────────────────
control MyDeparser(packet_out packet, in headers hdr) {{
    apply {{
        packet.emit(hdr.ethernet);
        packet.emit(hdr.result);   // result header first (before flow data)
        packet.emit(hdr.flow);
    }}
}}

// ── Switch instantiation ──────────────────────────────────────────────────────
V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
"""

    p4_path = os.path.join(OUT_DIR, p4_filename)
    with open(p4_path, 'w', encoding='utf-8') as f:
        f.write(p4_program)
    print(f"\n[1/5] Generated: {p4_path}")

    # ── 2. table_entries -- empty, tree uses const entries ───────────────
    entries_filename = f"table_entries{file_suffix}.txt"
    startup_commands = f"""\
# {entries_filename}
# No runtime table entries needed -- the decision tree is encoded as
# const entries directly in {p4_filename}.
#
# BMv2 startup command (run inside the VM):
#   sudo simple_switch --interface 0@veth0 --interface 1@veth1 \\
#       arp_detector{file_suffix}.json
#
# The compiled JSON is produced by:
#   p4c --target bmv2 --arch v1model {p4_filename}
"""
    entries_path = os.path.join(OUT_DIR, entries_filename)
    with open(entries_path, 'w', encoding='utf-8') as f:
        f.write(startup_commands)
    print(f"[2/5] Generated: {entries_path}")

    # ── 3. Flow test vectors ──────────────────────────────────────────────
    n_desc = "full test set" if N_TEST_VECTORS is None else f"{N_TEST_VECTORS} rows, stratified"
    print(f"\n[3/5] Building test vectors ({n_desc})...")

    if FE_FEATURES:
        FE_test = build_fe(X_test_int)
        X_scenario = pd.concat([X_test_int[RAW_FEATURES], FE_test[FE_FEATURES]], axis=1)[tree_feature_list]
    else:
        X_scenario = X_test_int[tree_feature_list]

    py_preds = tree_model.predict(X_scenario)

    ben_idx = X_scenario.index[y_test.loc[X_scenario.index] == 0]
    atk_idx = X_scenario.index[y_test.loc[X_scenario.index] == 1]
    rng = np.random.RandomState(RANDOM_STATE)
    if N_TEST_VECTORS is None:
        sample_idx = X_scenario.index  # full test set
    else:
        n_each = N_TEST_VECTORS // 2
        sample_idx = np.concatenate([
            rng.choice(ben_idx, min(n_each, len(ben_idx)), replace=False),
            rng.choice(atk_idx, min(n_each, len(atk_idx)), replace=False),
        ])
        rng.shuffle(sample_idx)

    X_sample = X_scenario.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]
    pred_sample = tree_model.predict(X_sample)

    vectors_filename = f"flow_test_vectors{file_suffix}.csv"
    vectors_path = os.path.join(OUT_DIR, vectors_filename)
    X_sample_out = X_sample.copy()
    X_sample_out['true_label']  = y_sample.values
    X_sample_out['python_pred'] = pred_sample
    X_sample_out.to_csv(vectors_path, index=False)
    print(f"[3/5] Generated: {vectors_path}  ({len(X_sample_out)} rows, "
          f"{(pred_sample==0).sum()} benign, {(pred_sample==1).sum()} attack predictions)")

    # ── 4. Test harness ────────────────────────────────────────────────────
    header_feat_list = repr(HEADER_FIELDS)
    harness_filename = f"test_harness{file_suffix}.py"
    results_csv_name = f"p4_validation_results{file_suffix}.csv"

    harness = f"""\
#!/usr/bin/env python3
\"\"\"
{harness_filename} -- P4 validation test harness for {scenario}
Generated by 04_generate_p4.py

PURPOSE
-------
For each row in {vectors_filename}, crafts one Ethernet frame carrying
that row's feature values in the custom flow header, sends it through the
running BMv2 switch, and reads back the verdict from the result header.
Compares every P4 verdict against the Python model's prediction stored in
the CSV (column 'python_pred').

A 100% match rate means the P4 decision tree encoding is faithful to the
Python model.

USAGE (inside the P4 VM, after simple_switch is running)
-----
  python3 {harness_filename} [--vectors {vectors_filename}]
                          [--iface veth0]
                          [--timeout 2.0]

DEPENDENCIES (pre-installed on the P4 tutorial VM)
------------
  scapy, pandas, numpy
\"\"\"

import argparse, sys, time, struct
import pandas as pd
import numpy as np

try:
    from scapy.all import Ether, sendp, sniff, Raw, conf
    conf.verb = 0
except ImportError:
    sys.exit("scapy not found. Install with: pip install scapy")

# ── Constants (must match {p4_filename}) ────────────────────────────────────
TYPE_FLOW   = {ethertype_flow:#06x}
TYPE_RESULT = {ethertype_result:#06x}
IFACE_DEFAULT = "veth0"
TIMEOUT_DEFAULT = 2.0   # seconds to wait for reflected packet

# Feature order in the custom header, all precomputed, 64-bit each.
HEADER_FIELDS = {header_feat_list}

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="P4 ARP detector test harness ({scenario})")
parser.add_argument('--vectors',  default='{vectors_filename}')
parser.add_argument('--iface',    default=IFACE_DEFAULT)
parser.add_argument('--timeout',  type=float, default=TIMEOUT_DEFAULT)
parser.add_argument('--max-rows', type=int,   default=None,
                    help="Stop after this many rows (default: all)")
args = parser.parse_args()

# ── Load test vectors ─────────────────────────────────────────────────────────
df = pd.read_csv(args.vectors)
if args.max_rows:
    df = df.head(args.max_rows)
print(f"Loaded {{len(df)}} test vectors from {{args.vectors}}")
print(f"Sending on interface: {{args.iface}}")

# ── Build a custom flow header from one row ───────────────────────────────────
def build_flow_header(row):
    \"\"\"Pack feature values into 8-byte big-endian fields, in HEADER_FIELDS
    order (matches the bit<64> fields in flow_t in {p4_filename}).\"\"\"
    fields = []
    for feat in HEADER_FIELDS:
        val = int(row.get(feat, 0))
        # Clamp to uint64 range
        val = max(0, min(val, 0xFFFFFFFFFFFFFFFF))
        fields.append(struct.pack('!Q', val))
    return b''.join(fields)

# ── Send one packet and receive the reflected result ──────────────────────────
def send_and_receive(flow_payload, iface, timeout):
    \"\"\"Send a TYPE_FLOW packet on iface (veth0) and receive the result on
    veth1. BMv2 egresses on port 1 (veth1), so Scapy must listen there.\"\"\"
    pkt = Ether(type=TYPE_FLOW) / Raw(load=flow_payload)

    # Start sniffing on veth1 BEFORE sending, so we don't miss the reply
    from scapy.all import AsyncSniffer
    sniffer = AsyncSniffer(
        iface='veth1',
        lfilter=lambda p: p.haslayer(Ether) and p[Ether].type == TYPE_RESULT,
        count=1,
        timeout=timeout
    )
    sniffer.start()

    sendp(pkt, iface=iface, verbose=False)

    sniffer.join(timeout=timeout + 0.5)
    captured = sniffer.results

    if not captured:
        return None

    # Result header layout: verdict(1B) + pad(1B) + reserved(2B) = 4 bytes
    raw = bytes(captured[0][Ether].payload)
    if len(raw) >= 1:
        return raw[0]   # verdict byte
    return None

# ── Main loop ─────────────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("P4 VALIDATION ({scenario}) -- sending test flows")
print("=" * 60)

results = []
n_match = 0
n_miss  = 0
n_timeout = 0

for i, row in df.iterrows():
    py_pred  = int(row['python_pred'])
    payload  = build_flow_header(row)
    p4_verdict = send_and_receive(payload, args.iface, args.timeout)

    if p4_verdict is None:
        n_timeout += 1
        status = 'TIMEOUT'
    elif p4_verdict == py_pred:
        n_match += 1
        status = 'MATCH'
    else:
        n_miss += 1
        status = f'MISMATCH (P4={{p4_verdict}}, Python={{py_pred}})'

    results.append({{'row': i, 'python_pred': py_pred,
                     'p4_verdict': p4_verdict, 'status': status}})

    if (len(results)) % 100 == 0:
        print(f"  {{len(results):>5}} / {{len(df)}}  "
              f"match={{n_match}}  mismatch={{n_miss}}  timeout={{n_timeout}}")

# ── Summary ────────────────────────────────────────────────────────────────────
total = len(results)
print("\\n" + "=" * 60)
print(f"RESULTS ({scenario}, {{total}} flows)")
print("=" * 60)
print(f"  Match    : {{n_match:>6}} / {{total}}  ({{n_match/total*100:.2f}}%)")
print(f"  Mismatch : {{n_miss:>6}} / {{total}}")
print(f"  Timeout  : {{n_timeout:>6}} / {{total}}")

if n_miss > 0:
    print("\\n  MISMATCHES (first 10):")
    shown = 0
    for r in results:
        if r['status'].startswith('MISMATCH'):
            print(f"    row={{r['row']}}  python={{r['python_pred']}}  p4={{r['p4_verdict']}}")
            shown += 1
            if shown >= 10: break

if n_match == total:
    print("\\n  RESULT: PERFECT MATCH -- P4 encoding is faithful to the Python model.")
elif n_timeout == total:
    print("\\n  RESULT: ALL TIMEOUTS -- is simple_switch running? Is the interface correct?")
else:
    print(f"\\n  RESULT: {{n_miss}} mismatches -- inspect the rows above to debug.")

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv('{results_csv_name}', index=False)
print(f"\\n  Full results saved -> {results_csv_name}")
"""

    harness_path = os.path.join(OUT_DIR, harness_filename)
    with open(harness_path, 'w', encoding='utf-8') as f:
        f.write(harness)
    print(f"[4/5] Generated: {harness_path}")

    # ── 5. README ──────────────────────────────────────────────────────────
    readme_filename = "README.md" if not file_suffix else f"README{file_suffix}.md"

    readme = f"""\
# P4 Validation -- {scenario} Decision Tree

Generated by `04_generate_p4.py` from the fitted {scenario} Decision Tree.

## What this validates

The {scenario} Decision Tree (depth={tree_model.get_depth()}, integer-quantized) is re-expressed as a
P4_16 program and run on BMv2. Each test flow is sent as a crafted packet;
the P4 program classifies it and reflects the verdict back. If every verdict
matches the Python model's prediction on the same row, the P4 encoding is
faithful and the model is confirmed deployable on a programmable SmartNIC.

## Files

| File | Purpose |
|---|---|
| `{p4_filename}` | P4_16 program (compile and run on BMv2) |
| `{vectors_filename}` | {len(X_sample_out):,} test-set rows ({n_desc}) with Python predictions |
| `{harness_filename}` | Sends flows through BMv2 and compares verdicts |
| `{entries_filename}` | BMv2 startup notes (no runtime entries needed) |

Uses ethertype pair {ethertype_flow:#06x}/{ethertype_result:#06x}, distinct
from the other scenario's, so the two validations can never be
cross-confused if both sets of files are present on the same VM.

## Scale factors used

| Feature | Scale factor | Meaning |
|---|---|---|
| Rate | x{INT_SCALE_RATE:,} | raw pkt/s × {INT_SCALE_RATE:,} |
| IAT | x{INT_SCALE_IAT:,} | seconds × {INT_SCALE_IAT:,} |
| Ratios (FE_payload_ratio, FE_size_cv_proxy) | x{INT_SCALE_RATIO:,} | × {INT_SCALE_RATIO:,} |

## Step-by-step (inside the P4 VM)

### 1. Copy this folder to the VM

From your Windows machine, copy the `p4_validation/` folder to the VM.
The easiest way is to use the VirtualBox shared folder feature, or:
```
scp -r p4_validation/ p4@<vm-ip>:/home/p4/
```

Or just drag-and-drop if the VM guest additions are installed.

### 2. Set up virtual interfaces

BMv2 needs a pair of virtual Ethernet interfaces:
```bash
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 up
sudo ip link set veth1 up
```

### 3. Compile the P4 program

```bash
cd /home/p4/p4_validation
p4c --target bmv2 --arch v1model --std p4-16 {p4_filename}
```

This produces `arp_detector{file_suffix}.json`.

### 4. Start the BMv2 switch

```bash
sudo simple_switch --interface 0@veth0 --interface 1@veth1 \\
    arp_detector{file_suffix}.json &
```

Wait 2-3 seconds for it to start.

### 5. Run the test harness

`{vectors_filename}` covers {n_desc} ({len(X_sample_out):,} rows). The
harness sends one packet and blocks on a capture per row, sequentially --
if you're running the full test set, expect this to take several hours,
not minutes. Progress prints every 100 rows so you can confirm it's moving.
Consider running it inside `tmux`/`screen` (or with `nohup ... &`) so it
survives a dropped VM console session, and pass `--max-rows N` first to
sanity-check on a small slice before committing to the full run.

```bash
python3 {harness_filename} --iface veth0 --vectors {vectors_filename}
```

### 6. Read the results

Expected output if the encoding is correct:
```
RESULT: PERFECT MATCH -- P4 encoding is faithful to the Python model.
```

Full per-row results are saved to `{results_csv_name}`.

## Troubleshooting

**All timeouts:** BMv2 is not running, or the interface name is wrong.
Try `ip link show` to list interfaces and pass the correct one with `--iface`.

**Mismatches:** The P4 tree logic diverges from Python at specific rows.
Check the mismatch rows in `{results_csv_name}` -- compare the
feature values against the tree structure printed by `04_generate_p4.py`
to identify which branch is being taken incorrectly.

**Compile error in p4c:** Check the p4c version matches (`p4c --version`).
This program was generated for p4c 1.2.x with BMv2 simple_switch 1.15.x.
"""

    readme_path = os.path.join(OUT_DIR, readme_filename)
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"[5/5] Generated: {readme_path}")

    return {
        'scenario': scenario,
        'p4_path': p4_path,
        'tree_depth': int(tree_model.get_depth()),
        'node_count': int(t.node_count),
        'header_fields': len(HEADER_FIELDS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generate S3 first (the scenario recommended as the operational default),
# then S4 (the scenario with feature engineering).
# ─────────────────────────────────────────────────────────────────────────────
s3_info = generate_scenario_p4(
    scenario='S3',
    results_path=os.path.join(RESULTS_DIR, 'S3_results.pkl'),
    model_path=os.path.join(MODELS_DIR, 'S3_Decision_Tree.joblib'),
    file_suffix='_s3',
    ethertype_flow=0x1236,
    ethertype_result=0x1237,
)

s4_info = generate_scenario_p4(
    scenario='S4',
    results_path=os.path.join(RESULTS_DIR, 'S4_results.pkl'),
    model_path=os.path.join(MODELS_DIR, 'S4_Decision_Tree.joblib'),
    file_suffix='',
    ethertype_flow=0x1234,
    ethertype_result=0x1235,
)

print(f"\n{'='*70}")
print(f"ALL FILES GENERATED -> {OUT_DIR}")
print(f"{'='*70}")
print(f"\n  Next step: copy the p4_validation/ folder to your P4 VM.")
print(f"  Then follow README_s3.md (S3) and README.md (S4) inside that folder.")

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE SUMMARY -- static analysis of both generated .p4 files: program
# size, tables, actions, header width, and register count. None of this
# requires compiling or running anything on BMv2, let alone physical
# hardware -- it is a property of the generated source text itself.
#
# Registers are checked explicitly (via a `register<` search) rather than
# assumed absent: the per-flow state footprint constraint (Section III)
# rules out any feature needing per-source state, so zero registers here
# is a design property made falsifiable, not merely asserted.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("RESOURCE SUMMARY")
print(f"{'='*70}")

def analyze_p4_resources(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    header_fields = re.findall(r'bit<64>\s+\w+;', text)
    tables   = re.findall(r'^\s*table\s+(\w+)\s*\{', text, re.MULTILINE)
    actions  = re.findall(r'^\s*action\s+(\w+)\s*\(', text, re.MULTILINE)
    registers = re.findall(r'\bregister\s*<', text)
    return {
        'file': os.path.basename(path),
        'lines': len(text.splitlines()),
        'bytes': len(text.encode('utf-8')),
        'header_fields': len(header_fields),
        'header_bytes': len(header_fields) * 8,
        'tables': len(tables),
        'actions': len(actions),
        'registers': len(registers),
    }

resource_summary = {
    'S3': analyze_p4_resources(s3_info['p4_path']),
    'S4': analyze_p4_resources(s4_info['p4_path']),
}

header = f"{'Metric':<22} {'S3':>14} {'S4':>14}"
print(header)
print("-" * len(header))
rows = [
    ('Program size (lines)', 'lines'),
    ('Program size (bytes)', 'bytes'),
    ('Tables',               'tables'),
    ('Actions',               'actions'),
    ('Header fields',         'header_fields'),
    ('Header width (bytes)',  'header_bytes'),
    ('Registers (stateful)',  'registers'),
]
for label, key in rows:
    print(f"{label:<22} {resource_summary['S3'][key]:>14,} {resource_summary['S4'][key]:>14,}")

resource_path = os.path.join(RESULTS_DIR, 'p4_resource_summary.json')
with open(resource_path, 'w', encoding='utf-8') as f:
    json.dump(resource_summary, f, indent=2)
print(f"\nSaved: {resource_path}")
