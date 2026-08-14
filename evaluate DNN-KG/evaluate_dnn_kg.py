import os
import argparse
import re
import warnings
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate the DNN-KG model using repeated five-fold "
        "out-of-fold cross-validation."
    )
)

parser.add_argument(
    "--data",
    default="SAF-Hydrogenation reaction.xlsx",
    help="Input Excel file containing the hydrogenation dataset.",
)

parser.add_argument(
    "--output-dir",
    default="DNN_fatty_repeated5fold_results",
    help="Directory for evaluation outputs.",
)

args = parser.parse_args()

EXCEL_PATH = args.data
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FEEDSTOCK_CATEGORY = "Fatty Acids & Esters"

# Repeated 5-fold:
# 20 repeats x 5 folds = 100 independently trained fold models
N_REPEATS = 20
N_SPLITS = 5
BASE_SEED = 42

# Original training settings
PRE_EPOCHS = 80
FT_EPOCHS = 50
BATCH_SIZE = 32

PRE_LR = 1e-3
FT_LR = 5e-4

LAMBDA_FT = 3.0
TARGET_COVER = 0.60

TOP_K_CONTEXT = 5

# Keep original weighted fine-tuning by default.
USE_WEIGHTED_FINETUNE = True



# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(BASE_SEED)


# ============================================================
# 3. READ EXCEL
# ============================================================

df = pd.read_excel(EXCEL_PATH)

print("=" * 82)
print("FATTY ACIDS & ESTERS — DNN-KG + REPEATED 5-FOLD OOF")
print("=" * 82)
print("Original data shape:", df.shape)


# ============================================================
# 4. PARSING FUNCTIONS
# ============================================================

def parse_numeric(series):
    """
    Extract first numerical value.

    Example:
        '95.6 %' -> 95.6
    """
    s = series.astype(str).str.extract(r'(\d+\.?\d*)')[0]
    return pd.to_numeric(s, errors="coerce")


def parse_temperature_to_celsius(series):
    """
    Convert reaction temperature to °C.
    Ranges are replaced by their mean.
    Kelvin is converted to Celsius.
    """
    def extract_one(text):
        if text is None:
            return np.nan

        t = str(text).strip()

        if (
            t == ""
            or t.lower() == "nan"
            or "not reported" in t.lower()
        ):
            return np.nan

        nums = re.findall(r'(\d+\.?\d*)', t)

        if not nums:
            return np.nan

        vals = [float(v) for v in nums]

        if re.search(r'k', t, flags=re.IGNORECASE):
            vals = [v - 273.15 for v in vals]

        return float(np.mean(vals))

    return series.apply(extract_one)


def parse_pressure_to_bar(series):
    """
    Convert pressure to bar.
    Ranges are replaced by their mean.
    """
    def extract_one(text):
        if text is None:
            return np.nan

        t = str(text).strip().lower()

        if (
            t == ""
            or t == "nan"
            or "not reported" in t
        ):
            return np.nan

        if "ambient" in t or "atmospher" in t:
            return 1.0

        nums = re.findall(r'(\d+\.?\d*)', t)

        if not nums:
            return np.nan

        vals = [float(v) for v in nums]

        if "mpa" in t:
            factor = 10.0
        elif "kpa" in t:
            factor = 0.01
        elif "psig" in t or "psi" in t:
            factor = 0.0689476
        elif "atm" in t:
            factor = 1.01325
        else:
            factor = 1.0

        vals_bar = [v * factor for v in vals]

        return float(np.mean(vals_bar))

    return series.apply(extract_one)


def parse_time_to_hour(series):
    """
    Convert reaction time to hours.
    Ranges are replaced by their mean.
    """
    def extract_one(text):
        if text is None:
            return np.nan

        t = str(text).strip().lower()

        if (
            t == ""
            or t == "nan"
            or "not reported" in t
        ):
            return np.nan

        nums = re.findall(r'(\d+\.?\d*)', t)

        if not nums:
            return np.nan

        vals = [float(v) for v in nums]

        if (
            "hour" in t
            or "hr" in t
            or re.search(r'(?<![a-z])h(?![a-z])', t)
        ):
            factor = 1.0

        elif "min" in t:
            factor = 1.0 / 60.0

        elif (
            "sec" in t
            or re.search(r'(?<![a-z])s(?![a-z])', t)
        ):
            factor = 1.0 / 3600.0

        else:
            factor = 1.0

        vals_h = [v * factor for v in vals]

        return float(np.mean(vals_h))

    return series.apply(extract_one)


# ============================================================
# 5. PARSE NUMERIC COLUMNS
# ============================================================

df["conv_num"] = parse_numeric(
    df["Conversion rate"]
).clip(0, 100)

df["sel_num"] = parse_numeric(
    df["Product selectivity"]
).clip(0, 100)

df["temp_num"] = parse_temperature_to_celsius(
    df["Reaction temperature"]
)

df["pres_num"] = parse_pressure_to_bar(
    df["Reaction pressure"]
)

df["time_num"] = parse_time_to_hour(
    df["Reaction time"]
)


# ============================================================
# 6. FATTY-ACID/ESTER SUPERVISED SUBSET
# ============================================================

fat_all = df[
    df["Feedstock category"]
    .astype(str)
    .str.strip()
    .eq(TARGET_FEEDSTOCK_CATEGORY)
].copy()

# At least one real target is required for supervised training.
label_available = (
    fat_all["conv_num"].notna()
    | fat_all["sel_num"].notna()
)

df_clean = (
    fat_all
    .loc[label_available]
    .copy()
    .reset_index()
    .rename(columns={"index": "original_row_index"})
)

df_clean["conv_frac"] = (
    df_clean["conv_num"] / 100.0
)

df_clean["sel_frac"] = (
    df_clean["sel_num"] / 100.0
)

df_clean["conv_mask"] = (
    df_clean["conv_num"]
    .notna()
    .astype(np.float32)
)

df_clean["sel_mask"] = (
    df_clean["sel_num"]
    .notna()
    .astype(np.float32)
)

n_samples = len(df_clean)

print("\n" + "=" * 82)
print("FATTY-ONLY DATA AVAILABILITY")
print("=" * 82)

print(
    "Total Fatty Acids & Esters records:",
    len(fat_all)
)

print(
    "Supervised rows (>=1 target):",
    len(df_clean)
)

print(
    "Conversion labels:",
    int(df_clean["conv_mask"].sum())
)

print(
    "Selectivity labels:",
    int(df_clean["sel_mask"].sum())
)

print(
    "Both labels:",
    int(
        (
            (df_clean["conv_mask"] == 1)
            &
            (df_clean["sel_mask"] == 1)
        ).sum()
    )
)

print(
    "Conversion only:",
    int(
        (
            (df_clean["conv_mask"] == 1)
            &
            (df_clean["sel_mask"] == 0)
        ).sum()
    )
)

print(
    "Selectivity only:",
    int(
        (
            (df_clean["conv_mask"] == 0)
            &
            (df_clean["sel_mask"] == 1)
        ).sum()
    )
)

print(
    "Temperature missing:",
    int(df_clean["temp_num"].isna().sum())
)

print(
    "Pressure missing:",
    int(df_clean["pres_num"].isna().sum())
)

print(
    "Time missing:",
    int(df_clean["time_num"].isna().sum())
)


# ============================================================
# 7. SEMANTIC / KG REPRESENTATION
# ============================================================

def clean_node_text(x):
    s = str(x).strip()

    if s.lower() in {
        "nan",
        "none",
        ""
    }:
        return None

    return s


def build_reaction_text(row):
    """
    Preserve the semantic fields used by the original model.
    """
    parts = [
        str(row["Reaction mode"]),
        str(row["Feedstock category"]),
        str(row["Feedstock"]),
        str(row["Catalyst"]),
        str(row["Product category"]),
        str(row["Product"]),
    ]

    return " | ".join(parts)


def get_node_texts_from_rows(frame):
    """
    Build KG-node text pool from TRAINING rows only.
    """
    vals = pd.unique(
        pd.concat([
            frame["Feedstock category"].astype(str),
            frame["Feedstock"].astype(str),
            frame["Catalyst"].astype(str),
            frame["Product category"].astype(str),
            frame["Product"].astype(str),
        ])
    )

    vals = [
        clean_node_text(v)
        for v in vals
    ]

    vals = [
        v
        for v in vals
        if v is not None
    ]

    return vals


print("\nLoading SentenceTransformer...")
emb_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

df_clean["reaction_text"] = df_clean.apply(
    build_reaction_text,
    axis=1
)

# Fixed external sentence embedding.
# It does not use conversion/selectivity labels.
sample_emb_all = emb_model.encode(
    df_clean["reaction_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

EMBED_DIM = sample_emb_all.shape[1]

print(
    "Sentence embedding shape:",
    sample_emb_all.shape
)


# ============================================================
# 8. NODE EMBEDDING CACHE
# ============================================================

# Pre-encode all unique node strings only for computational efficiency.
# During CV, the CONTEXT POOL itself still contains TRAINING-fold nodes only.

all_fatty_node_texts = get_node_texts_from_rows(
    df_clean
)

all_node_embeddings = emb_model.encode(
    all_fatty_node_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

node_emb_cache = {
    txt: emb
    for txt, emb in zip(
        all_fatty_node_texts,
        all_node_embeddings
    )
}


def get_cached_node_embedding(text):
    if text not in node_emb_cache:
        node_emb_cache[text] = emb_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype(np.float32)

    return node_emb_cache[text]


def build_context_from_pool(
    sample_embeddings,
    pool_texts,
    top_k=TOP_K_CONTEXT
):
    """
    For each sample:
        cosine similarity between sample embedding and TRAINING KG nodes
        -> top-k node vectors
        -> mean vector

    Embeddings are normalized, so dot product == cosine similarity.
    """
    if len(pool_texts) == 0:
        raise ValueError(
            "Training KG node pool is empty."
        )

    pool_emb = np.vstack([
        get_cached_node_embedding(txt)
        for txt in pool_texts
    ]).astype(np.float32)

    k = min(
        top_k,
        len(pool_emb)
    )

    contexts = []

    for emb_vec in sample_embeddings:
        sims = (
            pool_emb
            @ emb_vec
        )

        if k == len(pool_emb):
            top_idx = np.arange(
                len(pool_emb)
            )
        else:
            top_idx = np.argpartition(
                sims,
                -k
            )[-k:]

        context_vec = (
            pool_emb[top_idx]
            .mean(axis=0)
        )

        contexts.append(
            context_vec
        )

    return np.vstack(
        contexts
    ).astype(np.float32)


# ============================================================
# 9. RAW FEATURES + TARGETS + MASKS
# ============================================================

num_raw_all = df_clean[
    [
        "temp_num",
        "pres_num",
        "time_num"
    ]
].to_numpy(
    dtype=np.float32
)

y_all_nan = df_clean[
    [
        "conv_frac",
        "sel_frac"
    ]
].to_numpy(
    dtype=np.float32
)

target_mask_all = df_clean[
    [
        "conv_mask",
        "sel_mask"
    ]
].to_numpy(
    dtype=np.float32
)

# Missing targets are stored as 0 only for tensor construction.
# masked loss guarantees they do not contribute to training.
y_all_filled = np.nan_to_num(
    y_all_nan,
    nan=0.0
).astype(np.float32)


# ============================================================
# 10. ORIGINAL DNN ARCHITECTURE
# ============================================================

INPUT_DIM = (
    3
    + EMBED_DIM
)

OUTPUT_DIM = 2


class SAFDNN(nn.Module):
    """
    Original architecture:
        input -> 128 -> 64 -> 2
        ReLU / ReLU / Sigmoid
    """
    def __init__(
        self,
        in_dim,
        out_dim
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                in_dim,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                out_dim
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 11. MASKED MSE
# ============================================================

def masked_mse_loss(
    pred,
    target,
    target_mask,
    sample_weight=None
):
    sq = (
        pred
        - target
    ) ** 2

    effective_mask = target_mask

    if sample_weight is not None:
        effective_mask = (
            effective_mask
            * sample_weight.reshape(
                -1,
                1
            )
        )

    numerator = (
        sq
        * effective_mask
    ).sum()

    denominator = (
        effective_mask
        .sum()
        .clamp(min=1.0)
    )

    return (
        numerator
        / denominator
    )


# ============================================================
# 12. NUMERIC PREPROCESSING — TRAINING FOLD ONLY
# ============================================================

def fit_numeric_preprocessor(
    train_num_raw
):
    """
    Median imputation + StandardScaler fitted using training fold only.
    """
    medians = np.nanmedian(
        train_num_raw,
        axis=0
    )

    # This fallback should almost never be needed.
    global_medians = np.nanmedian(
        num_raw_all,
        axis=0
    )

    medians = np.where(
        np.isnan(medians),
        global_medians,
        medians
    )

    train_imputed = np.where(
        np.isnan(train_num_raw),
        medians,
        train_num_raw
    )

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_imputed
    ).astype(np.float32)

    return (
        medians.astype(np.float32),
        scaler,
        train_scaled
    )


def transform_numeric(
    raw_num,
    medians,
    scaler
):
    imputed = np.where(
        np.isnan(raw_num),
        medians,
        raw_num
    )

    return scaler.transform(
        imputed
    ).astype(np.float32)


# ============================================================
# 13. STRATIFICATION
# ============================================================
#
# We want each fold to contain similar proportions of:
#   - Both-label samples
#   - Conversion-only samples
#   - Selectivity-only samples
#
# For BOTH-label samples we additionally preserve broad target pattern:
#   low/high Conversion x low/high Selectivity.
#
# For single-label groups, we DO NOT split high/low because some
# resulting cells would contain fewer than 5 samples.
#
# With the current fatty dataset this produces strata with all n >= 5.
# If future data changes and any stratum becomes too small, the code
# automatically falls back to label-availability strata only.
# ============================================================

conv_median = float(
    df_clean.loc[
        df_clean["conv_mask"] == 1,
        "conv_num"
    ].median()
)

sel_median = float(
    df_clean.loc[
        df_clean["sel_mask"] == 1,
        "sel_num"
    ].median()
)


def build_joint_strata(frame):
    labels = []

    for _, row in frame.iterrows():

        has_conv = (
            row["conv_mask"] == 1
        )

        has_sel = (
            row["sel_mask"] == 1
        )

        if (
            has_conv
            and has_sel
        ):
            cbin = int(
                row["conv_num"]
                >= conv_median
            )

            sbin = int(
                row["sel_num"]
                >= sel_median
            )

            labels.append(
                f"B_{cbin}{sbin}"
            )

        elif has_conv:
            labels.append(
                "C_only"
            )

        else:
            labels.append(
                "S_only"
            )

    labels = np.asarray(labels)

    counts = pd.Series(
        labels
    ).value_counts()

    # Safety fallback
    if (
        len(counts) < 2
        or counts.min() < N_SPLITS
    ):
        print(
            "\nJoint stratification has a cell "
            f"with fewer than {N_SPLITS} samples."
        )
        print(
            "Falling back to label-availability strata only."
        )

        labels = []

        for _, row in frame.iterrows():

            has_conv = (
                row["conv_mask"] == 1
            )

            has_sel = (
                row["sel_mask"] == 1
            )

            if (
                has_conv
                and has_sel
            ):
                labels.append(
                    "Both"
                )

            elif has_conv:
                labels.append(
                    "Conversion_only"
                )

            else:
                labels.append(
                    "Selectivity_only"
                )

        labels = np.asarray(labels)

    return labels


strata = build_joint_strata(
    df_clean
)

strata_counts = (
    pd.Series(strata)
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 82)
print("STRATIFICATION")
print("=" * 82)

print(
    f"Conversion median = "
    f"{conv_median:.2f}%"
)

print(
    f"Selectivity median = "
    f"{sel_median:.2f}%"
)

print("\nStratum counts:")
print(
    strata_counts.to_string()
)

if strata_counts.min() < N_SPLITS:
    raise ValueError(
        "At least one final stratum contains fewer "
        "than 5 samples. Reduce N_SPLITS or simplify strata."
    )


# ============================================================
# 14. TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    train_idx,
    test_idx,
    seed
):
    """
    Train original DNN-KG on one training fold and predict one held-out fold.
    """

    set_seed(seed)

    # --------------------------------------------------------
    # 14.1 Numeric preprocessing — training fold only
    # --------------------------------------------------------

    (
        medians,
        scaler,
        train_num_scaled
    ) = fit_numeric_preprocessor(
        num_raw_all[train_idx]
    )

    test_num_scaled = transform_numeric(
        num_raw_all[test_idx],
        medians,
        scaler
    )

    # --------------------------------------------------------
    # 14.2 KG context — training node pool only
    # --------------------------------------------------------

    train_frame = df_clean.iloc[
        train_idx
    ]

    train_pool_texts = get_node_texts_from_rows(
        train_frame
    )

    train_ctx = build_context_from_pool(
        sample_emb_all[train_idx],
        train_pool_texts,
        top_k=TOP_K_CONTEXT
    )

    test_ctx = build_context_from_pool(
        sample_emb_all[test_idx],
        train_pool_texts,
        top_k=TOP_K_CONTEXT
    )

    X_train = np.hstack([
        train_num_scaled,
        train_ctx
    ]).astype(np.float32)

    X_test = np.hstack([
        test_num_scaled,
        test_ctx
    ]).astype(np.float32)

    y_train = y_all_filled[
        train_idx
    ]

    m_train = target_mask_all[
        train_idx
    ]

    # --------------------------------------------------------
    # 14.3 Fresh original SAFDNN
    # --------------------------------------------------------

    model = SAFDNN(
        INPUT_DIM,
        OUTPUT_DIM
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PRE_LR
    )

    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    m_train_t = torch.tensor(
        m_train,
        dtype=torch.float32
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_ds = TensorDataset(
        X_train_t,
        y_train_t,
        m_train_t
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    # --------------------------------------------------------
    # 14.4 Original 80-epoch pre-training
    # --------------------------------------------------------

    for _ in range(
        PRE_EPOCHS
    ):
        model.train()

        for (
            xb,
            yb,
            mb
        ) in train_loader:

            optimizer.zero_grad()

            pred = model(xb)

            loss = masked_mse_loss(
                pred,
                yb,
                mb
            )

            loss.backward()

            optimizer.step()

    # --------------------------------------------------------
    # 14.5 Original weighted fine-tuning
    # --------------------------------------------------------

    if USE_WEIGHTED_FINETUNE:

        combo_counts = (
            train_frame
            .groupby([
                "Feedstock category",
                "Product category"
            ])
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False
            )
        )

        combo_counts[
            "cum_ratio"
        ] = (
            combo_counts["count"]
            .cumsum()
            /
            combo_counts["count"]
            .sum()
        )

        top_k_combo = (
            int(
                (
                    combo_counts[
                        "cum_ratio"
                    ]
                    <= TARGET_COVER
                ).sum()
            )
            + 1
        )

        combo_top = (
            combo_counts
            .head(top_k_combo)
        )

        weights = np.ones(
            len(train_idx),
            dtype=np.float32
        )

        train_frame_reset = (
            train_frame
            .reset_index(drop=True)
        )

        for _, combo_row in combo_top.iterrows():

            combo_mask = (
                (
                    train_frame_reset[
                        "Feedstock category"
                    ].to_numpy()
                    ==
                    combo_row[
                        "Feedstock category"
                    ]
                )
                &
                (
                    train_frame_reset[
                        "Product category"
                    ].to_numpy()
                    ==
                    combo_row[
                        "Product category"
                    ]
                )
            )

            weights[
                combo_mask
            ] = LAMBDA_FT

        weights_t = torch.tensor(
            weights,
            dtype=torch.float32
        )

        train_ds_ft = TensorDataset(
            X_train_t,
            y_train_t,
            m_train_t,
            weights_t
        )

        generator_ft = (
            torch.Generator()
            .manual_seed(
                seed
                + 10000
            )
        )

        train_loader_ft = DataLoader(
            train_ds_ft,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator_ft
        )

        optimizer_ft = torch.optim.Adam(
            model.parameters(),
            lr=FT_LR
        )

        for _ in range(
            FT_EPOCHS
        ):
            model.train()

            for (
                xb,
                yb,
                mb,
                wb
            ) in train_loader_ft:

                optimizer_ft.zero_grad()

                pred = model(xb)

                loss = masked_mse_loss(
                    pred,
                    yb,
                    mb,
                    sample_weight=wb
                )

                loss.backward()

                optimizer_ft.step()

    # --------------------------------------------------------
    # 14.6 Held-out prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        pred_test = model(
            torch.tensor(
                X_test,
                dtype=torch.float32
            )
        ).cpu().numpy()

    return pred_test


# ============================================================
# 15. REPEATED 5-FOLD OOF
# ============================================================

print("\n" + "=" * 82)
print("STARTING REPEATED 5-FOLD OOF")
print("=" * 82)

print(
    f"Repeats            : {N_REPEATS}"
)

print(
    f"Folds per repeat   : {N_SPLITS}"
)

print(
    f"Total fold models  : "
    f"{N_REPEATS * N_SPLITS}"
)

print(
    f"Input dimension    : {INPUT_DIM}"
)

print(
    "DNN architecture   : "
    "387-ish -> 128 -> 64 -> 2"
)

print(
    f"Weighted fine-tune : "
    f"{USE_WEIGHTED_FINETUNE}"
)

print("=" * 82 + "\n")


per_repeat_rows = []
all_oof_prediction_rows = []

# Sum predictions across repeats for final averaged OOF plots.
pred_sum = np.zeros(
    (
        n_samples,
        2
    ),
    dtype=np.float64
)

pred_count = np.zeros(
    (
        n_samples,
        2
    ),
    dtype=np.int32
)


for repeat_id in range(
    1,
    N_REPEATS + 1
):

    repeat_seed = (
        BASE_SEED
        + repeat_id
        * 1000
    )

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=repeat_seed
    )

    # Every sample should receive exactly one OOF prediction per repeat.
    repeat_oof_pred = np.full(
        (
            n_samples,
            2
        ),
        np.nan,
        dtype=np.float32
    )

    repeat_fold_id = np.full(
        n_samples,
        -1,
        dtype=int
    )

    for (
        fold_id,
        (
            train_idx,
            test_idx
        )
    ) in enumerate(
        skf.split(
            np.zeros(n_samples),
            strata
        ),
        start=1
    ):

        fold_seed = (
            repeat_seed
            + fold_id
        )

        pred_test = train_one_fold(
            train_idx=train_idx,
            test_idx=test_idx,
            seed=fold_seed
        )

        repeat_oof_pred[
            test_idx
        ] = pred_test

        repeat_fold_id[
            test_idx
        ] = fold_id

        # Store all individual held-out predictions.
        for (
            local_pos,
            sample_idx
        ) in enumerate(
            test_idx
        ):

            true_conv = (
                y_all_nan[
                    sample_idx,
                    0
                ]
            )

            true_sel = (
                y_all_nan[
                    sample_idx,
                    1
                ]
            )

            all_oof_prediction_rows.append({
                "Repeat": repeat_id,
                "Fold": fold_id,
                "Sample_index": int(sample_idx),
                "Original_row_index":
                    int(
                        df_clean.loc[
                            sample_idx,
                            "original_row_index"
                        ]
                    ),

                "Experimental_conversion":
                    (
                        np.nan
                        if np.isnan(
                            true_conv
                        )
                        else float(
                            true_conv
                            * 100.0
                        )
                    ),

                "Predicted_conversion":
                    float(
                        pred_test[
                            local_pos,
                            0
                        ]
                        * 100.0
                    ),

                "Experimental_selectivity":
                    (
                        np.nan
                        if np.isnan(
                            true_sel
                        )
                        else float(
                            true_sel
                            * 100.0
                        )
                    ),

                "Predicted_selectivity":
                    float(
                        pred_test[
                            local_pos,
                            1
                        ]
                        * 100.0
                    ),
            })

    # Safety check: every sample must have exactly one OOF prediction.
    if np.isnan(
        repeat_oof_pred
    ).any():
        raise RuntimeError(
            f"Repeat {repeat_id}: "
            "some samples did not receive an OOF prediction."
        )

    # ========================================================
    # IMPORTANT:
    # Pool ALL 5 folds before calculating metrics.
    # ========================================================

    conv_valid = (
        target_mask_all[
            :,
            0
        ]
        == 1
    )

    sel_valid = (
        target_mask_all[
            :,
            1
        ]
        == 1
    )

    conv_true = (
        y_all_nan[
            conv_valid,
            0
        ]
        * 100.0
    )

    conv_pred = (
        repeat_oof_pred[
            conv_valid,
            0
        ]
        * 100.0
    )

    sel_true = (
        y_all_nan[
            sel_valid,
            1
        ]
        * 100.0
    )

    sel_pred = (
        repeat_oof_pred[
            sel_valid,
            1
        ]
        * 100.0
    )

    conv_mae = mean_absolute_error(
        conv_true,
        conv_pred
    )

    conv_rmse = np.sqrt(
        mean_squared_error(
            conv_true,
            conv_pred
        )
    )

    conv_r2 = r2_score(
        conv_true,
        conv_pred
    )

    sel_mae = mean_absolute_error(
        sel_true,
        sel_pred
    )

    sel_rmse = np.sqrt(
        mean_squared_error(
            sel_true,
            sel_pred
        )
    )

    sel_r2 = r2_score(
        sel_true,
        sel_pred
    )

    per_repeat_rows.append({
        "Repeat": repeat_id,

        "Conversion_n":
            int(
                conv_valid.sum()
            ),

        "Conversion_MAE":
            conv_mae,

        "Conversion_RMSE":
            conv_rmse,

        "Conversion_R2":
            conv_r2,

        "Selectivity_n":
            int(
                sel_valid.sum()
            ),

        "Selectivity_MAE":
            sel_mae,

        "Selectivity_RMSE":
            sel_rmse,

        "Selectivity_R2":
            sel_r2,
    })

    # Accumulate predictions for averaged OOF prediction.
    for sample_idx in range(
        n_samples
    ):
        if (
            target_mask_all[
                sample_idx,
                0
            ]
            == 1
        ):
            pred_sum[
                sample_idx,
                0
            ] += repeat_oof_pred[
                sample_idx,
                0
            ]

            pred_count[
                sample_idx,
                0
            ] += 1

        if (
            target_mask_all[
                sample_idx,
                1
            ]
            == 1
        ):
            pred_sum[
                sample_idx,
                1
            ] += repeat_oof_pred[
                sample_idx,
                1
            ]

            pred_count[
                sample_idx,
                1
            ] += 1

    print(
        f"[Repeat {repeat_id:02d}/{N_REPEATS}] "
        f"Conversion: "
        f"MAE={conv_mae:6.2f}, "
        f"RMSE={conv_rmse:6.2f}, "
        f"R²={conv_r2:7.3f} | "
        f"Selectivity: "
        f"MAE={sel_mae:6.2f}, "
        f"RMSE={sel_rmse:6.2f}, "
        f"R²={sel_r2:7.3f}"
    )


# ============================================================
# 16. MEAN ± SD ACROSS 20 POOLED-OOF REPEATS
# ============================================================

per_repeat_metrics = pd.DataFrame(
    per_repeat_rows
)


def make_mean_sd_summary(
    target_name,
    prefix
):
    return {
        "Target": target_name,

        "N_labels":
            int(
                per_repeat_metrics[
                    f"{prefix}_n"
                ].iloc[0]
            ),

        "MAE_mean":
            per_repeat_metrics[
                f"{prefix}_MAE"
            ].mean(),

        "MAE_SD":
            per_repeat_metrics[
                f"{prefix}_MAE"
            ].std(ddof=1),

        "RMSE_mean":
            per_repeat_metrics[
                f"{prefix}_RMSE"
            ].mean(),

        "RMSE_SD":
            per_repeat_metrics[
                f"{prefix}_RMSE"
            ].std(ddof=1),

        "R2_mean":
            per_repeat_metrics[
                f"{prefix}_R2"
            ].mean(),

        "R2_SD":
            per_repeat_metrics[
                f"{prefix}_R2"
            ].std(ddof=1),

        "MAE_median":
            per_repeat_metrics[
                f"{prefix}_MAE"
            ].median(),

        "RMSE_median":
            per_repeat_metrics[
                f"{prefix}_RMSE"
            ].median(),

        "R2_median":
            per_repeat_metrics[
                f"{prefix}_R2"
            ].median(),
    }


summary_mean_sd = pd.DataFrame([
    make_mean_sd_summary(
        "Conversion",
        "Conversion"
    ),

    make_mean_sd_summary(
        "Selectivity",
        "Selectivity"
    ),
])


print("\n" + "=" * 82)
print("REPEATED 5-FOLD POOLED-OOF: MEAN ± SD")
print("=" * 82)

for _, row in summary_mean_sd.iterrows():

    print(
        f"{row['Target']}: "
        f"MAE = "
        f"{row['MAE_mean']:.2f} "
        f"± {row['MAE_SD']:.2f}; "
        f"RMSE = "
        f"{row['RMSE_mean']:.2f} "
        f"± {row['RMSE_SD']:.2f}; "
        f"R² = "
        f"{row['R2_mean']:.3f} "
        f"± {row['R2_SD']:.3f}"
    )


# ============================================================
# 17. AVERAGED OOF PREDICTIONS ACROSS ALL 20 REPEATS
# ============================================================

avg_pred = np.full(
    (
        n_samples,
        2
    ),
    np.nan,
    dtype=np.float64
)

for target_idx in range(
    2
):
    valid_count = (
        pred_count[
            :,
            target_idx
        ]
        > 0
    )

    avg_pred[
        valid_count,
        target_idx
    ] = (
        pred_sum[
            valid_count,
            target_idx
        ]
        /
        pred_count[
            valid_count,
            target_idx
        ]
    )


# Conversion
conv_eval_mask = (
    target_mask_all[
        :,
        0
    ]
    == 1
)

conv_true_avg = (
    y_all_nan[
        conv_eval_mask,
        0
    ]
    * 100.0
)

conv_pred_avg = (
    avg_pred[
        conv_eval_mask,
        0
    ]
    * 100.0
)

agg_conv_mae = mean_absolute_error(
    conv_true_avg,
    conv_pred_avg
)

agg_conv_rmse = np.sqrt(
    mean_squared_error(
        conv_true_avg,
        conv_pred_avg
    )
)

agg_conv_r2 = r2_score(
    conv_true_avg,
    conv_pred_avg
)


# Selectivity
sel_eval_mask = (
    target_mask_all[
        :,
        1
    ]
    == 1
)

sel_true_avg = (
    y_all_nan[
        sel_eval_mask,
        1
    ]
    * 100.0
)

sel_pred_avg = (
    avg_pred[
        sel_eval_mask,
        1
    ]
    * 100.0
)

agg_sel_mae = mean_absolute_error(
    sel_true_avg,
    sel_pred_avg
)

agg_sel_rmse = np.sqrt(
    mean_squared_error(
        sel_true_avg,
        sel_pred_avg
    )
)

agg_sel_r2 = r2_score(
    sel_true_avg,
    sel_pred_avg
)


averaged_oof_metrics = pd.DataFrame({
    "Target": [
        "Conversion",
        "Selectivity"
    ],

    "N_labels": [
        int(
            conv_eval_mask.sum()
        ),
        int(
            sel_eval_mask.sum()
        )
    ],

    "MAE": [
        agg_conv_mae,
        agg_sel_mae
    ],

    "RMSE": [
        agg_conv_rmse,
        agg_sel_rmse
    ],

    "R2": [
        agg_conv_r2,
        agg_sel_r2
    ],
})


print("\n" + "=" * 82)
print("AVERAGED OOF PREDICTIONS ACROSS ALL REPEATS")
print("=" * 82)

print(
    averaged_oof_metrics
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 18. AVERAGED OOF PREDICTION TABLE
# ============================================================

avg_prediction_table = pd.DataFrame({
    "Sample_index":
        np.arange(n_samples),

    "Original_row_index":
        df_clean[
            "original_row_index"
        ].to_numpy(),

    "Conversion_observed":
        target_mask_all[
            :,
            0
        ].astype(int),

    "Conversion_OOF_repeats":
        pred_count[
            :,
            0
        ],

    "Experimental_conversion":
        y_all_nan[
            :,
            0
        ]
        * 100.0,

    "Mean_OOF_predicted_conversion":
        avg_pred[
            :,
            0
        ]
        * 100.0,

    "Selectivity_observed":
        target_mask_all[
            :,
            1
        ].astype(int),

    "Selectivity_OOF_repeats":
        pred_count[
            :,
            1
        ],

    "Experimental_selectivity":
        y_all_nan[
            :,
            1
        ]
        * 100.0,

    "Mean_OOF_predicted_selectivity":
        avg_pred[
            :,
            1
        ]
        * 100.0,

    "Temperature_C":
        df_clean[
            "temp_num"
        ].to_numpy(),

    "Pressure_bar":
        df_clean[
            "pres_num"
        ].to_numpy(),

    "Time_h":
        df_clean[
            "time_num"
        ].to_numpy(),
})


for col in [
    "DOI",
    "Title",
    "Reaction mode",
    "Feedstock category",
    "Feedstock",
    "Catalyst",
    "Product category",
    "Product"
]:
    if col in df_clean.columns:
        avg_prediction_table[
            col
        ] = (
            df_clean[
                col
            ]
            .astype(str)
            .to_numpy()
        )


# ============================================================
# 19. DATA AVAILABILITY SUMMARY
# ============================================================

availability_summary = pd.DataFrame({
    "Item": [
        "Raw records",
        "Fatty Acids & Esters records",
        "Fatty supervised rows (>=1 target)",
        "Conversion labels",
        "Selectivity labels",
        "Both labels",
        "Conversion only",
        "Selectivity only",
        "Temperature missing among fatty supervised rows",
        "Pressure missing among fatty supervised rows",
        "Time missing among fatty supervised rows",
        "Repeated CV repeats",
        "Folds per repeat",
        "Total fold models"
    ],

    "Count": [
        len(df),
        len(fat_all),
        len(df_clean),

        int(
            df_clean[
                "conv_mask"
            ].sum()
        ),

        int(
            df_clean[
                "sel_mask"
            ].sum()
        ),

        int(
            (
                (
                    df_clean[
                        "conv_mask"
                    ]
                    == 1
                )
                &
                (
                    df_clean[
                        "sel_mask"
                    ]
                    == 1
                )
            ).sum()
        ),

        int(
            (
                (
                    df_clean[
                        "conv_mask"
                    ]
                    == 1
                )
                &
                (
                    df_clean[
                        "sel_mask"
                    ]
                    == 0
                )
            ).sum()
        ),

        int(
            (
                (
                    df_clean[
                        "conv_mask"
                    ]
                    == 0
                )
                &
                (
                    df_clean[
                        "sel_mask"
                    ]
                    == 1
                )
            ).sum()
        ),

        int(
            df_clean[
                "temp_num"
            ].isna().sum()
        ),

        int(
            df_clean[
                "pres_num"
            ].isna().sum()
        ),

        int(
            df_clean[
                "time_num"
            ].isna().sum()
        ),

        N_REPEATS,
        N_SPLITS,
        N_REPEATS * N_SPLITS
    ]
})


# ============================================================
# 20. EXPORT EXCEL
# ============================================================

excel_output = os.path.join(
    OUTPUT_DIR,
    "DNN_KG_repeated5fold_evaluation.xlsx"
)

with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:

    summary_mean_sd.to_excel(
        writer,
        sheet_name="Mean_SD_metrics",
        index=False
    )

    averaged_oof_metrics.to_excel(
        writer,
        sheet_name="Averaged_OOF_metrics",
        index=False
    )

    per_repeat_metrics.to_excel(
        writer,
        sheet_name="Per_repeat_pooled_OOF",
        index=False
    )

    avg_prediction_table.to_excel(
        writer,
        sheet_name="Averaged_OOF_predictions",
        index=False
    )

    pd.DataFrame(
        all_oof_prediction_rows
    ).to_excel(
        writer,
        sheet_name="All_OOF_predictions",
        index=False
    )

    availability_summary.to_excel(
        writer,
        sheet_name="Data_availability",
        index=False
    )

    pd.DataFrame({
        "Stratum":
            strata_counts.index,

        "Count":
            strata_counts.values
    }).to_excel(
        writer,
        sheet_name="Strata_summary",
        index=False
    )


print(
    "\nSaved evaluation workbook:"
)

print(
    excel_output
)


# ============================================================
# 21. PLOTS
# ============================================================

def plot_avg_oof(
    y_true,
    y_pred,
    target_name,
    mae,
    rmse,
    r2,
    filename
):
    fig, ax = plt.subplots(
        figsize=(
            6.0,
            5.6
        )
    )

    ax.scatter(
        y_true,
        y_pred,
        s=55,
        edgecolors="black",
        linewidths=0.5
    )

    lo = min(
        0.0,
        float(
            np.min(y_true)
        ),
        float(
            np.min(y_pred)
        )
    )

    hi = max(
        100.0,
        float(
            np.max(y_true)
        ),
        float(
            np.max(y_pred)
        )
    )

    ax.plot(
        [
            lo,
            hi
        ],
        [
            lo,
            hi
        ],
        "--",
        linewidth=1.2
    )

    ax.set_xlim(
        lo,
        hi
    )

    ax.set_ylim(
        lo,
        hi
    )

    ax.set_xlabel(
        (
            f"Experimental "
            f"{target_name.lower()} (%)"
        ),
        fontsize=13
    )

    ax.set_ylabel(
        (
            f"Predicted "
            f"{target_name.lower()} (%)"
        ),
        fontsize=13
    )

    ax.text(
        0.05,
        0.95,
        (
            f"MAE = {mae:.2f}\n"
            f"RMSE = {rmse:.2f}\n"
            f"$R^2$ = {r2:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=12
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=600,
        bbox_inches="tight"
    )

    # Avoid Windows GUI-thread errors.
    plt.close(fig)


conv_plot = os.path.join(
    OUTPUT_DIR,
    "DNN_fatty_only_avgOOF_conversion.png"
)

sel_plot = os.path.join(
    OUTPUT_DIR,
    "DNN_fatty_only_avgOOF_selectivity.png"
)


plot_avg_oof(
    conv_true_avg,
    conv_pred_avg,
    "Conversion",
    agg_conv_mae,
    agg_conv_rmse,
    agg_conv_r2,
    conv_plot
)


plot_avg_oof(
    sel_true_avg,
    sel_pred_avg,
    "Selectivity",
    agg_sel_mae,
    agg_sel_rmse,
    agg_sel_r2,
    sel_plot
)


print("\nSaved plots:")
print(conv_plot)
print(sel_plot)

