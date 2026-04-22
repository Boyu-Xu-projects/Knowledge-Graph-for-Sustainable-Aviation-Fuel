import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ---------- 第 2 步：读取 excel ----------
path = r"C:\Users\xuboy\saf_llm_ner_project\SAF-prediction\SAF-prediction-trend surface analysis\SAF-Hydrogenation reaction.xlsx"

df = pd.read_excel(path)

print("原始数据形状:", df.shape)
print(df.head())
print(df.columns)


# ---------- 第 3 步：把转化率、选择性、温度、压力、时间变成数字（温度统一 °C，压力统一 bar，时间统一 h） ----------

import re
import numpy as np

def parse_numeric(series):
    """从字符串中提取第一个数字，示例：
    '95.6 %' → 95.6
    '250-350 bar' → 250
    """
    s = series.astype(str).str.extract(r'(\d+\.?\d*)')[0]
    return pd.to_numeric(s, errors='coerce')


# ------------------------- 温度：统一转换为 °C -------------------------
def parse_temperature_to_celsius(series):
    """
    温度统一转换为摄氏度（°C）
    支持：
    - '200 °C', '200°C', '80 C', '350℃'
    - '300–350 °C' → 取平均
    - '323K', '323 K', '473K（高温）' → Kelvin → °C
    """

    def extract_celsius(text):
        if text is None:
            return np.nan

        t = str(text).strip()
        if t == "":
            return np.nan

        nums = re.findall(r'(\d+\.?\d*)', t)
        if not nums:
            return np.nan

        vals = [float(v) for v in nums]

        # 判断是否 Kelvin（K/k），不要求空格
        is_kelvin = bool(re.search(r'k', t, flags=re.IGNORECASE))

        if is_kelvin:
            vals_c = [v - 273.15 for v in vals]
        else:
            vals_c = vals   # 默认 °C

        return float(np.mean(vals_c))

    return series.apply(extract_celsius)



# ------------------------- 压力：统一转换为 bar -------------------------
def parse_pressure_to_bar(series):
    """
    将 Reaction pressure 统一转换为 bar。
    支持：
    - '10 bar'
    - '2 MPa' / '2MPa'        → 20 bar
    - '200 kPa' / '200kPa'    → 2 bar
    - '300 psi', '300 psig'   → 20.7 bar
    - '1 atm'                 → 1.013 bar
    - 'ambient pressure'      → 1 bar
    - 区间：'2–3 MPa' → 25 bar
    """

    def extract_bar(text):
        if text is None:
            return np.nan

        t = str(text).strip().lower()
        if t == "":
            return np.nan

        # 特殊：ambient / atmospheric
        if "ambient" in t or "atmospher" in t:
            return 1.0

        nums = re.findall(r'(\d+\.?\d*)', t)
        if not nums:
            return np.nan

        vals = [float(v) for v in nums]

        # 判断单位
        if "mpa" in t:
            factor = 10.0
        elif "kpa" in t:
            factor = 0.01
        elif "psig" in t or "psi" in t:
            factor = 0.0689476
        elif "atm" in t:
            factor = 1.01325
        elif "bar" in t:
            factor = 1.0
        else:
            factor = 1.0  # 默认单位：bar

        vals_bar = [v * factor for v in vals]
        return float(np.mean(vals_bar))

    return series.apply(extract_bar)



# ------------------------- 时间：全部转换为小时 h -------------------------
def parse_time_to_hour(series):
    """
    Reaction time → 全部转换成小时（h）
    支持：
    - '2 h', '2h', '1.5 h'
    - '30 min', '30 mins', '90min'
    - '360 s', '360 sec'
    - 区间 '2–3 h', '30–60 min'
    """

    def extract_hour(text):
        if text is None:
            return np.nan

        t = str(text).strip().lower()
        if t == "":
            return np.nan

        nums = re.findall(r'(\d+\.?\d*)', t)
        if not nums:
            return np.nan

        vals = [float(v) for v in nums]

        # 判断单位
        if "hour" in t or "hr" in t or "h" in t:
            factor = 1.0
        elif "min" in t:
            factor = 1.0 / 60.0
        elif "sec" in t or "s" in t:
            factor = 1.0 / 3600.0
        else:
            factor = 1.0  # 默认按小时

        vals_h = [v * factor for v in vals]
        return float(np.mean(vals_h))

    return series.apply(extract_hour)
    
# ------------------------- molar rate (Reactant molar ratio) 解析 -------------------------
def parse_molar_rate_with_unit(series: pd.Series):
    """
    解析 Reactant molar ratio -> molar_value（数值） + molar_unit（单位标签）

    注意：表里可能混有 mol/mol, v/v, Nm3/m3 等，数值不可直接混在一个尺度里比较。
    这里先把数值提出来，并保留单位标签。
    """

    def extract_one(text):
        if text is None:
            return (np.nan, "unknown")

        raw = str(text).strip()
        if raw == "" or raw.lower() == "nan":
            return (np.nan, "unknown")

        low = raw.lower()

        # 过滤：明显不是你要的 molar ratio
        if "not reported" in low:
            return (np.nan, "unknown")
        if "gas composition" in low:
            return (np.nan, "unknown")
        if "(wt/wt)" in low or "wt/wt" in low:
            return (np.nan, "unknown")

        # -------- 单位标签（用于分组/后续换算）--------
        unit = "unknown"
        if ("mol/mol" in low) or ("mol mol" in low):
            unit = "mol/mol"
        elif "v/v" in low:
            unit = "v/v"
        else:
            # 放宽 Nm3/m3 的识别：只要包含 nm + m3/m³ 结构
            if "nm" in low and ("m3/m3" in low or "m³/m³" in low or "m3/m³" in low or "m³/m3" in low):
                unit = "Nm3/m3"

        # 尽量从 H2: 开始截取，避免 CO2:H2:Ar 这种
        h2_pos = low.find("h2:")
        seg = raw if h2_pos < 0 else raw[h2_pos:]
        seg_low = seg.lower()

        # -------- 1) 区间：a-b / a to b / a–b --------
        m_range = re.search(r'(\d+\.?\d*)\s*(?:-|–|to)\s*(\d+\.?\d*)', seg_low)
        if m_range:
            a = float(m_range.group(1))
            b = float(m_range.group(2))
            return (float((a + b) / 2.0), unit)

        # -------- 2) 冒号形式：a:b 可能是“比值”也可能是“范围写法” --------
        m_ratio = re.search(r'(\d+\.?\d*)\s*:\s*(\d+\.?\d*)', seg_low)
        if m_ratio:
            a = float(m_ratio.group(1))
            b = float(m_ratio.group(2))

            # 关键修正：当单位是 v/v 或 Nm3/m3 且两边都很大，优先按“范围”处理
            # 例如 800:1200 (v/v) 更像 800–1200
            if unit in ("v/v", "Nm3/m3") and a >= 50 and b >= 50:
                return (float((a + b) / 2.0), unit)

            # 否则按比值处理：a/b
            if b != 0:
                return (float(a / b), unit)
            return (np.nan, unit)

        # -------- 3) 单值：直接取数字均值 --------
        nums = re.findall(r'(\d+\.?\d*)', seg_low)
        if nums:
            vals = [float(x) for x in nums]
            return (float(np.mean(vals)), unit)

        return (np.nan, unit)

    tmp = series.apply(extract_one)
    molar_value = tmp.apply(lambda x: x[0])
    molar_unit  = tmp.apply(lambda x: x[1])
    return molar_value, molar_unit


def convert_molar_to_mol_per_L(molar_value, molar_unit):
    """
    把不同单位的 molar_value 统一成 mol H2 / L oil（mol/L_oil）

    - Nm3/m3: 认为 Nm3 是标准状态体积。
      默认：0°C, 1 atm -> 1 Nm3 = 44.615 mol
      mol/L = (Nm3/m3 * 44.615) / 1000

    - v/v: 近似把 v/v 当成“标准气体体积/液体体积”，按 Nm3/m3 同样处理
      （注意：如果你的 v/v 不是标准体积，会有系统误差）

    - mol/mol: 无法转成 mol/L（缺少油的密度 & 摩尔质量），返回 NaN
    """
    if pd.isna(molar_value) or molar_unit is None:
        return np.nan

    MOL_PER_NM3 = 44.615  # 0°C, 1 atm 的近似值，可统一修改

    if molar_unit == "Nm3/m3":
        return (float(molar_value) * MOL_PER_NM3) / 1000.0

    if molar_unit == "v/v":
        return (float(molar_value) * MOL_PER_NM3) / 1000.0

    return np.nan


# ------------------------- 开始解析 df -------------------------

df["conv_num"] = parse_numeric(df["Conversion rate"])
df["sel_num"]  = parse_numeric(df["Product selectivity"])

df["temp_num"] = parse_temperature_to_celsius(df["Reaction temperature"])
df["pres_num"] = parse_pressure_to_bar(df["Reaction pressure"])
df["time_num"] = parse_time_to_hour(df["Reaction time"])

df["molar_value"], df["molar_unit"] = parse_molar_rate_with_unit(df["Reactant molar ratio"])

df["molar_mol_per_L"] = [
    convert_molar_to_mol_per_L(v, u) for v, u in zip(df["molar_value"], df["molar_unit"])
]




# 丢掉必需字段缺失的数据（用于训练）
df_clean = df.dropna(subset=["conv_num", "sel_num", "temp_num", "pres_num", "time_num"]).copy()

# 百分比裁剪 + 归一化（为 Sigmoid 输出准备）
df_clean["conv_num"] = df_clean["conv_num"].clip(0, 100)
df_clean["sel_num"]  = df_clean["sel_num"].clip(0, 100)

df_clean["conv_frac"] = df_clean["conv_num"] / 100.0
df_clean["sel_frac"]  = df_clean["sel_num"] / 100.0

print("清洗后有效样本数:", len(df_clean))
print(df_clean[["Feedstock category","Product category","conv_num","sel_num","temp_num","pres_num","time_num"]].head())


# ---------- 新增：构造不含 "not reported"（仅检查 3 个条件列）的 excel-2 ----------

cols_to_check = ["Reaction time", "Reaction temperature", "Reaction pressure"]

mask_bad = df[cols_to_check].astype(str).apply(
    lambda col: col.str.contains("not reported", case=False, na=False)
).any(axis=1)

df2 = df[~mask_bad].copy()
print("删掉 not reported 后的 excel-2 行数:", len(df2))

# 可选：导出 excel
df2.to_excel(
    r"C:\Users\xuboy\saf_llm_ner_project\SAF-prediction\SAF-Hydrogenation reaction_excel2_no_not_reported.xlsx",
    index=False
)


# 对 df2 做和前面一样的数值清洗（再来一遍）
df2["conv_num"] = parse_numeric(df2["Conversion rate"])
df2["sel_num"]  = parse_numeric(df2["Product selectivity"])

df2["temp_num"] = parse_temperature_to_celsius(df2["Reaction temperature"])
df2["pres_num"] = parse_pressure_to_bar(df2["Reaction pressure"])
df2["time_num"] = parse_time_to_hour(df2["Reaction time"])

df2["molar_value"], df2["molar_unit"] = parse_molar_rate_with_unit(df2["Reactant molar ratio"])
df2["molar_mol_per_L"] = [
    convert_molar_to_mol_per_L(v, u) for v, u in zip(df2["molar_value"], df2["molar_unit"])
]



df2_clean = df2.dropna(subset=["conv_num", "sel_num", "temp_num", "pres_num", "time_num"]).copy()
print("excel-2 清洗后有效样本数:", len(df2_clean))


df2_clean["conv_num"] = df2_clean["conv_num"].clip(0, 100)
df2_clean["sel_num"]  = df2_clean["sel_num"].clip(0, 100)

df2_clean["conv_frac"] = df2_clean["conv_num"] / 100.0
df2_clean["sel_frac"]  = df2_clean["sel_num"] / 100.0

print("excel-2 清洗后有效样本数:", len(df2_clean))



# ---------- 第 4 步：构建 KG 节点 + 文本 embedding ----------

# 把原料类别、原料、催化剂、产物类别、产物都当成 KG 节点文本
node_texts = pd.unique(
    pd.concat([
        df_clean["Feedstock category"].astype(str),
        df_clean["Feedstock"].astype(str),
        df_clean["Catalyst"].astype(str),
        df_clean["Product category"].astype(str),
        df_clean["Product"].astype(str),
    ])
)
# 去掉 "nan"
node_texts = [t for t in node_texts if t.lower() != "nan"]

print("KG 节点数:", len(node_texts))

# ① 这里一定要先创建 emb_model
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# 再用 emb_model 对 KG 节点做 embedding
node_emb = emb_model.encode(
    node_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ---------- 第 5 步：为每条反应构造 reaction_text + context embedding ----------

def build_reaction_text(row):
    parts = [
        str(row["Reaction mode"]),       # ★ 新加：反应模式
        str(row["Feedstock category"]),
        str(row["Feedstock"]),
        str(row["Catalyst"]),
        str(row["Product category"]),
        str(row["Product"]),
    ]
    return " | ".join(parts)

df_clean["reaction_text"] = df_clean.apply(build_reaction_text, axis=1)

sample_emb = emb_model.encode(
    df_clean["reaction_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

top_k = 5
context_list = []

for emb in sample_emb:
    sims = cosine_similarity(emb.reshape(1, -1), node_emb)[0]
    top_idx = sims.argsort()[-top_k:]
    ctx_vec = node_emb[top_idx].mean(axis=0)
    context_list.append(ctx_vec)

context_emb = np.vstack(context_list)
print("context_emb 形状:", context_emb.shape)



# ---------- 第 6 步：构造 DNN 输入 X_all 和 输出 y_all ----------

# 数值条件特征（温度 + 压力 + 时间）
num_feats = df_clean[["temp_num", "pres_num", "time_num"]].to_numpy()

scaler = StandardScaler()
num_scaled = scaler.fit_transform(num_feats)

X_all = np.hstack([num_scaled, context_emb])


### ★★★ Sigmoid 版：y 用 0–1 小数而不是 0–100 百分比
y_all = df_clean[["conv_frac","sel_frac"]].to_numpy()


print("X_all 形状:", X_all.shape)
print("y_all 形状:", y_all.shape)

# ---------- 新增：基于 excel-2 (df2_clean) 构造 X_all_2, y_all_2 ----------

# 先给 df2_clean 也构造 reaction_text
df2_clean["reaction_text"] = df2_clean.apply(build_reaction_text, axis=1)

# 文本 embedding
sample_emb_2 = emb_model.encode(
    df2_clean["reaction_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

# 基于原来的 KG 节点 embedding (node_emb) 构造 context_emb_2
context_list_2 = []
top_k = 5  # 和前面保持一致
for emb_vec in sample_emb_2:
    sims = cosine_similarity(emb_vec.reshape(1, -1), node_emb)[0]
    top_idx = sims.argsort()[-top_k:]
    ctx_vec = node_emb[top_idx].mean(axis=0)
    context_list_2.append(ctx_vec)

context_emb_2 = np.vstack(context_list_2)
print("excel-2 的 context_emb_2 形状:", context_emb_2.shape)

# 数值特征（温度 + 压力），注意用“原来在 df_clean 上 fit 的 scaler”做 transform
num_feats_2 = df2_clean[["temp_num", "pres_num", "time_num"]].to_numpy()
num_scaled_2 = scaler.transform(num_feats_2)

X_all_2 = np.hstack([num_scaled_2, context_emb_2])
y_all_2 = df2_clean[["conv_frac", "sel_frac"]].to_numpy()

print("excel-2 的 X_all_2 形状:", X_all_2.shape)
print("excel-2 的 y_all_2 形状:", y_all_2.shape)


# ---------- 第 7 步：定义 PyTorch DNN 模型 + PRE-TRAIN ----------

input_dim  = X_all.shape[1]
output_dim = 2   # [conv, sel]

class SAFDNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Sigmoid()   # ★★★ Sigmoid 版：输出严格在 [0,1]
        )
    def forward(self, x):
        return self.net(x)


model = SAFDNN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 把 numpy 转成 torch tensor
X_all_t = torch.tensor(X_all, dtype=torch.float32)
y_all_t = torch.tensor(y_all, dtype=torch.float32)

train_ds = TensorDataset(X_all_t, y_all_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 预训练若干 epoch
num_epochs_pre = 80

for epoch in range(num_epochs_pre):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    total_loss /= len(train_loader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f"[Pre-train] Epoch {epoch+1}/{num_epochs_pre}  MSE={total_loss:.2f}")

# 保存预训练好的模型参数
torch.save(model.state_dict(), "saf_dnn_pretrained.pt")
print("预训练完成并保存为 saf_dnn_pretrained.pt")

# ---------- 第 8 步（改）：按覆盖 60% 样本数选择重点组合 ----------

combo_counts = (
    df2_clean
    .groupby(["Feedstock category", "Product category"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

# ★ 目标：重点组合覆盖 60% 的样本
target_cover = 0.60

combo_counts["cum_ratio"] = combo_counts["count"].cumsum() / combo_counts["count"].sum()

# 自动确定 K
top_k_combo = int((combo_counts["cum_ratio"] <= target_cover).sum()) + 1

combo_topk = combo_counts.head(top_k_combo)

print(f"\n[excel-2] 覆盖 {target_cover*100:.0f}% 样本的重点组合数 K = {top_k_combo}")
print(combo_topk)

# 在 df2_clean 里为这些组合构造 mask
mask_ft = pd.Series(False, index=df2_clean.index)

for _, row in combo_topk.iterrows():
    f_cat = row["Feedstock category"]
    p_cat = row["Product category"]
    mask_ft |= (
        (df2_clean["Feedstock category"] == f_cat) &
        (df2_clean["Product category"]   == p_cat)
    )

df_ft = df2_clean[mask_ft].copy()
print("\n[excel-2] 用于微调的重点样本数:", len(df_ft))
print(df_ft[["Feedstock category", "Product category"]].drop_duplicates())

# ---------- 第 9 步：Weighted Fine-tuning：在 excel-2 的所有样本上继续训练，重点 12 组合权重大 ----------

# 从预训练参数开始微调
model.load_state_dict(torch.load("saf_dnn_pretrained.pt"))

lambda_ft = 3.0  # 重点组合样本的 loss 权重
sample_weights = np.ones(len(df2_clean), dtype=np.float32)
sample_weights[mask_ft.values] = lambda_ft

print("[excel-2] 权重 = 1.0 的样本数:", (sample_weights == 1.0).sum())
print(f"[excel-2] 权重 = {lambda_ft} 的样本数:", (sample_weights == lambda_ft).sum())

optimizer_ft = torch.optim.Adam(model.parameters(), lr=5e-4)

# 使用 excel-2 的 X_all_2 / y_all_2
X_all_2_t = torch.tensor(X_all_2, dtype=torch.float32)
y_all_2_t = torch.tensor(y_all_2, dtype=torch.float32)
w_all_2_t = torch.tensor(sample_weights, dtype=torch.float32)

train_ds_ft = TensorDataset(X_all_2_t, y_all_2_t, w_all_2_t)
train_loader_ft = DataLoader(train_ds_ft, batch_size=32, shuffle=True)

num_epochs_ft = 50

for epoch in range(num_epochs_ft):
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for xb, yb, wb in train_loader_ft:
        optimizer_ft.zero_grad()
        pred = model(xb)

        mse = (pred - yb) ** 2
        mse_per_sample = mse.mean(dim=1)

        loss = (mse_per_sample * wb).mean()
        loss.backward()
        optimizer_ft.step()

        total_loss   += (mse_per_sample * wb).sum().item()
        total_weight += wb.sum().item()

    avg_loss = total_loss / total_weight
    if (epoch + 1) % 10 == 0:
        print(f"[Fine-tune(excel-2, weighted-top{top_k_combo})] Epoch {epoch+1}/{num_epochs_ft}  weighted MSE={avg_loss:.4f}")

torch.save(model.state_dict(), "saf_dnn_finetuned.pt")  # 你可以保持同名
print("在 excel-2 上加权微调完成并保存为 saf_dnn_finetuned.pt")

# ---------- 第 10 步：可视化预测结果（温度–收率 & 压力–收率） ----------
# 使用 excel-2 (df2 / df2_clean)，在 excel-2 里统计 Feedstock，只要样本数 >=2 就画

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 10.0 基本 mask 和“典型氢化压力” pres_default（bar）

# 在 excel-2 (df2) 里筛选 Reaction mode = Hydrogenation reaction
mode_mask_raw = (df2["Reaction mode"].astype(str) == "Hydrogenation reaction")

# 在 excel-2 清洗后 (df2_clean) 里，找出有数值压力的 Hydrogenation，用来求默认压力
hydro_mask_clean = (df2_clean["Reaction mode"].astype(str) == "Hydrogenation reaction")

pres_default = df2_clean.loc[hydro_mask_clean, "pres_num"].mean()
time_default = df2_clean.loc[hydro_mask_clean, "time_num"].mean()
temp_default = df2_clean.loc[hydro_mask_clean, "temp_num"].mean()

print(f"[excel-2] temp_default (°C) = {temp_default:.2f}")
print(f"[excel-2] pres_default (bar) = {pres_default:.2f}")
print(f"[excel-2] time_default (h)   = {time_default:.2f}")


# ============================================================
# 10.0.1 统计 excel-2 中可视化用的 Feedstock（Hydrogenation, n>=2）
# ============================================================

# 只看 Hydrogenation reaction
mode_mask_raw = (df2["Reaction mode"].astype(str) == "Hydrogenation reaction")

fs_counts = (
    df2.loc[mode_mask_raw, "Feedstock"]
    .astype(str)
    .value_counts()
)

valid_feedstocks = fs_counts[fs_counts >= 2].index.tolist()

print(
    "[excel-2] Hydrogenation 下样本数 >=2 的 Feedstock：",
    valid_feedstocks
)

if len(valid_feedstocks) == 0:
    raise ValueError(
        "在 excel-2 中，Hydrogenation 反应下没有任何 Feedstock 的样本数 >= 2"
    )

# KG context 中相似节点数量
top_k_ctx = 5

# 加载 fine-tuned 模型（只需一次）
model.load_state_dict(torch.load("saf_dnn_finetuned.pt"))
model.eval()

# ============================================================
# 10.1a 3D scatter + SEMANTIC trend plane (temperature, data-driven)
# Selectivity (%) – Conversion (%) – Reaction temperature (°C)
# ============================================================

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

target_cat = "Fatty Acids & Esters"

# ------------------------------------------------------------
# 10.1a-1 过滤原始 df（temperature reported）
# ------------------------------------------------------------
mask_cat = (df["Feedstock category"].astype(str) == target_cat)
mask_temp_reported = ~df["Reaction temperature"].astype(str).str.contains(
    "not reported", case=False, na=False
)

sub_raw = df.loc[mask_cat & mask_temp_reported].copy()
sub_raw = sub_raw[~sub_raw["temp_num"].isna()].copy()

if len(sub_raw) == 0:
    raise ValueError(f"[10.1a] {target_cat} 没有可用 temperature 数据")

# ------------------------------------------------------------
# 10.1a-2 填补 pressure / time（使用 hydrogenation 平均值）
# ------------------------------------------------------------
sub_raw["pres_num_filled"] = sub_raw["pres_num"].fillna(pres_default)
sub_raw["time_num_filled"] = sub_raw["time_num"].fillna(time_default)

# ------------------------------------------------------------
# 10.1a-3 KG context embedding
# ------------------------------------------------------------
sub_raw["reaction_text"] = sub_raw.apply(build_reaction_text, axis=1)

sample_emb = emb_model.encode(
    sub_raw["reaction_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

context_list = []
for emb_vec in sample_emb:
    sims = cosine_similarity(emb_vec.reshape(1, -1), node_emb)[0]
    top_idx = sims.argsort()[-top_k_ctx:]
    context_list.append(node_emb[top_idx].mean(axis=0))

context_emb = np.vstack(context_list)

# ------------------------------------------------------------
# 10.1a-4 模型预测（0–1 → %）
# ------------------------------------------------------------
temps_real = sub_raw["temp_num"].to_numpy()
pres  = sub_raw["pres_num_filled"].to_numpy()
time  = sub_raw["time_num_filled"].to_numpy()

num_feats = np.column_stack([temps_real, pres, time])
num_scaled = scaler.transform(num_feats)
X_pred = np.hstack([num_scaled, context_emb])

with torch.no_grad():
    y_pred = model(torch.tensor(X_pred, dtype=torch.float32)).numpy()

conv_pct = np.clip(y_pred[:, 0], 0.0, 1.0) * 100.0
sel_pct  = np.clip(y_pred[:, 1], 0.0, 1.0) * 100.0

conv_adj = conv_pct.copy()
sel_adj  = sel_pct.copy()

# ------------------------------------------------------------
# 催化剂名称
# ------------------------------------------------------------
catalyst_names = (
    sub_raw["Catalyst"]
    .fillna("Unknown catalyst")
    .astype(str)
    .str.strip()
    .replace({"": "Unknown catalyst"})
    .values
)

# ============================================================
# 趋势面：线性回归平面（斜面）
# ============================================================
from sklearn.linear_model import LinearRegression

# 用 DNN 预测的 (conv_adj, sel_adj) 拟合真实温度 temps_real
X_plane = np.column_stack([
    conv_adj,      # 转化率 (%)
    sel_adj        # 选择性 (%)
])

reg_temp = LinearRegression()
reg_temp.fit(X_plane, temps_real)

# 生成网格用于画图
conv_grid = np.linspace(conv_adj.min(), conv_adj.max(), 40)
sel_grid  = np.linspace(sel_adj.min(),  sel_adj.max(),  40)
CONV, SEL = np.meshgrid(conv_grid, sel_grid)

X_grid = np.column_stack([CONV.ravel(), SEL.ravel()])
TEMP_pred = reg_temp.predict(X_grid).reshape(CONV.shape)

# ============================================================
# Matplotlib 3D（序号 + 彩色 list）
# ============================================================
fig = plt.figure(figsize=(12.5, 8))
ax = fig.add_subplot(111, projection="3d")

# ---- 趋势平面（改为半透明，alpha=0.3）----
ax.plot_surface(
    SEL, CONV, TEMP_pred,
    cmap="viridis",
    alpha=0.3,          # ★ 改成半透明
    linewidth=0
)

# ---- 散点 ----
sc = ax.scatter(
    sel_adj, conv_adj, temps_real,
    c=temps_real,
    cmap="viridis",
    s=65,
    edgecolor="k",
    linewidth=0.3
)

# ============================================================
# ★ 1) conversion 排序 → 序号
# ============================================================
order_conv_desc = np.argsort(-conv_adj)
rank = np.empty_like(order_conv_desc)
rank[order_conv_desc] = np.arange(1, len(conv_adj) + 1)

# ============================================================
# 导出散点图「序号 - 名称」到 Excel
# ============================================================

scatter_label_df = pd.DataFrame({
    "Scatter_rank": rank,
    "Catalyst_name": catalyst_names
})

# 按图中显示顺序（conversion 从高到低）
scatter_label_df = scatter_label_df.loc[order_conv_desc].reset_index(drop=True)

scatter_label_df.to_excel(
    "scatter_labels_temperature.xlsx",
    index=False
)

print("已导出：scatter_labels_temperature.xlsx")

# ============================================================
# ★ 2) 散点右上角：序号（黑色，字号调小）
# ============================================================
dx, dy, dz = 0.6, 0.6, 5
for x, y, z, r in zip(sel_adj, conv_adj, temps_real, rank):
    ax.text(
        x + dx, y + dy, z + dz,
        str(r),
        fontsize=10,      # ★ 从 14 改为 10，字号小一些
        color="black",
    )

# ============================================================
# ★ 3) 右侧 list：序号黑色 + 名称颜色 = 散点颜色（字号调小）
# ============================================================
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(
    vmin=temps_real.min(),
    vmax=temps_real.max()
)

x0 = 0.68
y0 = 0.80
line_h = 0.015

for k, i in enumerate(order_conv_desc):
    y = y0 - k * line_h
    if y < 0.05:
        break

    color_i = cmap(norm(temps_real[i]))

    # 序号（黑色，字号调小）
    fig.text(
        x0, y,
        f"{rank[i]}.",
        ha="left",
        va="center",
        fontsize=7,     
        color="black",
        family="monospace"
    )

    # 名称（与散点同色，字号调小）
    fig.text(
        x0 + 0.02, y,
        catalyst_names[i],
        ha="left",
        va="center",
        fontsize=7,      
        color=color_i,
        family="monospace"
    )
    
# ---- 坐标轴 ----
ax.set_xlabel("Predicted selectivity (%)", fontsize=14)
ax.set_ylabel("Predicted conversion (%)", fontsize=14)
ax.view_init(elev=20, azim=60)

ax.tick_params(axis="both", labelsize=14)
ax.tick_params(axis="z", labelsize=14)

# 直接用原始温度作为 Z 轴刻度
ax.set_zlabel("Reaction temperature (°C)", fontsize=14)
z_ticks_real = np.linspace(temps_real.min(), temps_real.max(), 5)
ax.set_zticks(z_ticks_real)
ax.set_zticklabels([f"{t:.0f}" for t in z_ticks_real], fontsize=14)

# ---- 左侧 colorbar ----
cax = fig.add_axes([0.08, 0.25, 0.02, 0.5])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label(
    "Reaction temperature (°C)",
    rotation=90,
    labelpad=12,
    fontsize=14
)

cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()

# ============================================================
# 10.1c 3D scatter + LINEAR trend plane (time, ECDF-scaled Z-axis)
# Selectivity (%) – Conversion (%) – Reaction time (h)
# ============================================================

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

target_cat = "Fatty Acids & Esters"

# ------------------------------------------------------------
# 1) 过滤原始 df：time reported
# ------------------------------------------------------------
mask_cat = (df["Feedstock category"].astype(str) == target_cat)
mask_time_reported = ~df["Reaction time"].astype(str).str.contains(
    "not reported", case=False, na=False
)

sub_time = df.loc[mask_cat & mask_time_reported].copy()
sub_time = sub_time[~sub_time["time_num"].isna()].copy()

if len(sub_time) == 0:
    raise ValueError(f"[10.1c] {target_cat} 没有可用 time 数据")

# ------------------------------------------------------------
# 2) 填补 temperature / pressure（与 10.1a 一致）
# ------------------------------------------------------------
sub_time["temp_num_filled"] = sub_time["temp_num"].fillna(temp_default)
sub_time["pres_num_filled"] = sub_time["pres_num"].fillna(pres_default)

# ------------------------------------------------------------
# 3) KG context embedding
# ------------------------------------------------------------
sub_time["reaction_text"] = sub_time.apply(build_reaction_text, axis=1)

sample_emb = emb_model.encode(
    sub_time["reaction_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

context_list = []
for emb_vec in sample_emb:
    sims = cosine_similarity(emb_vec.reshape(1, -1), node_emb)[0]
    top_idx = sims.argsort()[-top_k_ctx:]
    context_list.append(node_emb[top_idx].mean(axis=0))

context_emb = np.vstack(context_list)

# ------------------------------------------------------------
# 4) 模型预测
# ------------------------------------------------------------
temps = sub_time["temp_num_filled"].to_numpy()
pres  = sub_time["pres_num_filled"].to_numpy()
time_real = sub_time["time_num"].to_numpy()

num_feats = np.column_stack([temps, pres, time_real])
num_scaled = scaler.transform(num_feats)
X_pred = np.hstack([num_scaled, context_emb])

with torch.no_grad():
    y_pred = model(torch.tensor(X_pred, dtype=torch.float32)).numpy()

conv_pct = np.clip(y_pred[:, 0], 0.0, 1.0) * 100.0
sel_pct  = np.clip(y_pred[:, 1], 0.0, 1.0) * 100.0

conv_adj = conv_pct.copy()
sel_adj  = sel_pct.copy()

# ------------------------------------------------------------
# 5) 催化剂名称
# ------------------------------------------------------------
catalyst_names = (
    sub_time["Catalyst"]
    .fillna("Unknown catalyst")
    .astype(str)
    .str.strip()
    .replace({"": "Unknown catalyst"})
    .values
)

# ============================================================
# 6) ECDF warp（time 轴，用于 Z 轴显示）
# ============================================================
order = np.argsort(time_real)
time_sorted = time_real[order]
time_ecdf = np.linspace(0.0, 1.0, len(time_sorted))
time_warp = np.interp(time_real, time_sorted, time_ecdf)

# ============================================================
# 7) 趋势面：线性回归平面（直接在 warp 空间拟合）
# ============================================================
# 用 DNN 预测的 (conv_adj, sel_adj) 拟合 warp 后的时间 time_warp
X_plane = np.column_stack([
    conv_adj,      # 转化率 (%)
    sel_adj        # 选择性 (%)
])

reg_time_warp = LinearRegression()
reg_time_warp.fit(X_plane, time_warp)

# 生成网格用于画图
conv_grid = np.linspace(conv_adj.min(), conv_adj.max(), 40)
sel_grid  = np.linspace(sel_adj.min(),  sel_adj.max(),  40)
CONV, SEL = np.meshgrid(conv_grid, sel_grid)

X_grid = np.column_stack([CONV.ravel(), SEL.ravel()])
TIME_pred_warp = reg_time_warp.predict(X_grid).reshape(CONV.shape)

# ============================================================
# 8) Matplotlib 3D（序号 + 彩色 list）
# ============================================================
fig = plt.figure(figsize=(12.5, 8))
ax = fig.add_subplot(111, projection="3d")

# ---- 趋势平面（半透明，直接用 warp 空间的预测值）----
ax.plot_surface(
    SEL, CONV, TIME_pred_warp,
    cmap="plasma",
    alpha=0.3,          # 半透明
    linewidth=0
)

# ---- 散点（使用 warp 后的时间）----
sc = ax.scatter(
    sel_adj, conv_adj, time_warp,
    c=time_real,
    cmap="plasma",
    s=60,
    edgecolor="k",
    linewidth=0.3
)

# ============================================================
# ★ 1) conversion 排序 → 序号
# ============================================================
order_conv_desc = np.argsort(-conv_adj)
rank = np.empty_like(order_conv_desc)
rank[order_conv_desc] = np.arange(1, len(conv_adj) + 1)

# ============================================================
# 导出散点图「序号 - 名称」到 Excel
# ============================================================
scatter_label_df = pd.DataFrame({
    "Scatter_rank": rank,
    "Catalyst_name": catalyst_names
})

# 按图中显示顺序（conversion 从高到低）
scatter_label_df = scatter_label_df.loc[order_conv_desc].reset_index(drop=True)

scatter_label_df.to_excel(
    "scatter_labels_time.xlsx",
    index=False
)

print("已导出：scatter_labels_time.xlsx")

# ============================================================
# ★ 2) 散点右上角：序号（黑色，字号调小）
# ============================================================
dx, dy, dz = 0.6, 0.6, 0.02
for x, y, z, r in zip(sel_adj, conv_adj, time_warp, rank):
    ax.text(
        x + dx, y + dy, z + dz,
        str(r),
        fontsize=10,
        color="black",
    )

# ============================================================
# ★ 3) 右侧 list（序号黑色 + 名称颜色 = 散点颜色，字号调小，间距调窄）
# ============================================================
cmap = plt.get_cmap("plasma")
norm = mpl.colors.Normalize(
    vmin=time_real.min(),
    vmax=time_real.max()
)

x0 = 0.68
y0 = 0.80
line_h = 0.015

for k, i in enumerate(order_conv_desc):
    y = y0 - k * line_h
    if y < 0.05:
        break

    color_i = cmap(norm(time_real[i]))

    fig.text(
        x0, y,
        f"{rank[i]}.",
        ha="left",
        va="center",
        fontsize=7,
        color="black",
        family="monospace"
    )

    fig.text(
        x0 + 0.02, y,
        catalyst_names[i],
        ha="left",
        va="center",
        fontsize=7,
        color=color_i,
        family="monospace"
    )

# ---- 坐标轴 ----
ax.set_xlabel("Predicted selectivity (%)", fontsize=14)
ax.set_ylabel("Predicted conversion (%)", fontsize=14)
ax.view_init(elev=20, azim=60)

ax.tick_params(axis="both", labelsize=14)
ax.tick_params(axis="z", labelsize=14)

# ---- Z 轴 ticks（真实时间，但显示在 warp 位置上）----
z_ticks_real = np.array([0, 10, 25, 50, 100, 150, 200])
z_ticks_warp = np.interp(z_ticks_real, time_sorted, time_ecdf)
keep = np.insert(np.diff(z_ticks_warp) > 0.05, 0, True)
ax.set_zticks(z_ticks_warp[keep])
ax.set_zticklabels([str(int(t)) for t in z_ticks_real[keep]], fontsize=14)
ax.zaxis.set_label_position("upper") 
ax.tick_params(axis="z", pad=6)

# ---- 左侧 colorbar ----
cax = fig.add_axes([0.08, 0.25, 0.02, 0.5])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("Reaction time (h)", rotation=90, labelpad=12, fontsize=14)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()

# ============================================================
# 10.2 3D scatter + LINEAR trend plane (pressure, ECDF-scaled Z-axis)
# Selectivity (%) – Conversion (%) – Reaction pressure (bar)
# ============================================================

from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

target_cat = "Fatty Acids & Esters"

# ------------------------------------------------------------
# 10.2-1 过滤原始 df（pressure reported）
# ------------------------------------------------------------
mask_cat = (df["Feedstock category"].astype(str) == target_cat)
mask_pres_reported = ~df["Reaction pressure"].astype(str).str.contains(
    "not reported", case=False, na=False
)

sub_pres = df.loc[mask_cat & mask_pres_reported].copy()
sub_pres = sub_pres[~sub_pres["pres_num"].isna()].copy()

if len(sub_pres) == 0:
    raise ValueError(f"[10.2] {target_cat} 没有可用 pressure 数据")

# ------------------------------------------------------------
# 10.2-2 填补 temperature / time（与 10.1a 相同）
# ------------------------------------------------------------
sub_pres["temp_num_filled"] = sub_pres["temp_num"].fillna(temp_default)
sub_pres["time_num_filled"] = sub_pres["time_num"].fillna(time_default)

# ------------------------------------------------------------
# 10.2-3 KG context embedding
# ------------------------------------------------------------
sub_pres["reaction_text"] = sub_pres.apply(build_reaction_text, axis=1)

sample_emb = emb_model.encode(
    sub_pres["reaction_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

context_list = []
for emb_vec in sample_emb:
    sims = cosine_similarity(emb_vec.reshape(1, -1), node_emb)[0]
    top_idx = sims.argsort()[-top_k_ctx:]
    context_list.append(node_emb[top_idx].mean(axis=0))

context_emb = np.vstack(context_list)

# ------------------------------------------------------------
# 10.2-4 模型预测（0–1 → %）
# ------------------------------------------------------------
temps = sub_pres["temp_num_filled"].to_numpy()
pres_real = sub_pres["pres_num"].to_numpy()  # 真实的 pressure
time  = sub_pres["time_num_filled"].to_numpy()

num_feats = np.column_stack([temps, pres_real, time])
num_scaled = scaler.transform(num_feats)
X_pred = np.hstack([num_scaled, context_emb])

with torch.no_grad():
    y_pred = model(torch.tensor(X_pred, dtype=torch.float32)).numpy()

conv_pct = np.clip(y_pred[:, 0], 0.0, 1.0) * 100.0
sel_pct  = np.clip(y_pred[:, 1], 0.0, 1.0) * 100.0

conv_adj = conv_pct.copy()
sel_adj  = sel_pct.copy()

# ------------------------------------------------------------
# 10.2-5 催化剂名称
# ------------------------------------------------------------
catalyst_names = (
    sub_pres["Catalyst"]
    .fillna("Unknown catalyst")
    .astype(str)
    .str.strip()
    .replace({"": "Unknown catalyst"})
    .values
)

# ============================================================
# 10.2-6 ECDF warp（pressure 轴，用于 Z 轴显示）
# ============================================================
order = np.argsort(pres_real)
pres_sorted = pres_real[order]
pres_ecdf = np.linspace(0.0, 1.0, len(pres_sorted))
pres_warp = np.interp(pres_real, pres_sorted, pres_ecdf)

# ============================================================
# 10.2-7 线性趋势平面（直接在 warp 空间拟合）
# ============================================================
# 用 DNN 预测的 (conv_adj, sel_adj) 拟合 warp 后的压力 pres_warp
X_plane = np.column_stack([
    conv_adj,      # 转化率 (%)
    sel_adj        # 选择性 (%)
])

reg_pres_warp = LinearRegression()
reg_pres_warp.fit(X_plane, pres_warp)

# 生成网格用于画图
conv_grid = np.linspace(conv_adj.min(), conv_adj.max(), 40)
sel_grid  = np.linspace(sel_adj.min(),  sel_adj.max(),  40)
CONV, SEL = np.meshgrid(conv_grid, sel_grid)

X_grid = np.column_stack([CONV.ravel(), SEL.ravel()])
PRES_pred_warp = reg_pres_warp.predict(X_grid).reshape(CONV.shape)

# ============================================================
# 10.2-8 Matplotlib 3D（序号 + 右侧 list）
# ============================================================
fig = plt.figure(figsize=(12.5, 8))
ax = fig.add_subplot(111, projection="3d")

# ---- 趋势平面（半透明，用 warp 空间的预测值）----
ax.plot_surface(
    SEL, CONV, PRES_pred_warp,
    cmap="cividis",
    alpha=0.3,          # 半透明
    linewidth=0
)

# ---- 散点（使用 warp 后的压力）----
sc = ax.scatter(
    sel_adj, conv_adj, pres_warp,
    c=pres_real,
    cmap="cividis",
    s=60,
    edgecolor="k",
    linewidth=0.3
)

# ============================================================
# ★ 1) conversion 排序 → 序号
# ============================================================
order_conv_desc = np.argsort(-conv_adj)
rank = np.empty_like(order_conv_desc)
rank[order_conv_desc] = np.arange(1, len(conv_adj) + 1)

# ============================================================
# 导出散点图「序号 - 名称」到 Excel
# ============================================================
scatter_label_df = pd.DataFrame({
    "Scatter_rank": rank,
    "Catalyst_name": catalyst_names
})

# 按图中显示顺序（conversion 从高到低）
scatter_label_df = scatter_label_df.loc[order_conv_desc].reset_index(drop=True)

scatter_label_df.to_excel(
    "scatter_labels_pressure.xlsx",
    index=False
)

print("已导出：scatter_labels_pressure.xlsx")

# ============================================================
# ★ 2) 散点右上角：黑色序号（字号调小）
# ============================================================
dx, dy, dz = 0.6, 0.6, 0.02

for x, y, z, r in zip(sel_adj, conv_adj, pres_warp, rank):
    ax.text(
        x + dx,
        y + dy,
        z + dz,
        str(r),
        fontsize=10,
        color="black",
    )

# ============================================================
# ★ 3) 右侧 list：序号黑色 + 名称颜色 = pressure（字号调小，间距调窄）
# ============================================================
cmap = plt.get_cmap("cividis")
norm = mpl.colors.Normalize(vmin=pres_real.min(), vmax=pres_real.max())

x0 = 0.68
y0 = 0.80
line_h = 0.015

for k, i in enumerate(order_conv_desc):
    y = y0 - k * line_h
    if y < 0.05:
        break

    color_i = cmap(norm(pres_real[i]))

    # 序号（黑色，字号调小）
    fig.text(
        x0, y,
        f"{rank[i]}.",
        ha="left",
        va="center",
        fontsize=7,
        color="black",
        family="monospace"
    )

    # 名称（与散点同色，字号调小，间距调窄）
    fig.text(
        x0 + 0.02, y,
        catalyst_names[i],
        ha="left",
        va="center",
        fontsize=7,
        color=color_i,
        family="monospace"
    )

# ---- 坐标轴 ----
ax.set_xlabel("Predicted selectivity (%)", fontsize=14)
ax.set_ylabel("Predicted conversion (%)", fontsize=14)

ax.view_init(elev=20, azim=60)
ax.tick_params(axis="both", labelsize=14)
ax.tick_params(axis="z", labelsize=14)

# ---- Z 轴 ticks（真实压力，但显示在 warp 位置上）----
# 选择有代表性的压力值作为刻度
z_ticks_real = np.array([1, 5, 10, 20, 30, 40, 50, 60, 80, 100])
# 只保留在数据范围内的刻度
z_ticks_real = z_ticks_real[(z_ticks_real >= pres_real.min()) & (z_ticks_real <= pres_real.max())]
z_ticks_warp = np.interp(z_ticks_real, pres_sorted, pres_ecdf)
# 去掉彼此太近的 tick（ECDF 空间）
keep = np.insert(np.diff(z_ticks_warp) > 0.05, 0, True)
ax.set_zticks(z_ticks_warp[keep])
ax.set_zticklabels([f"{int(t)}" for t in z_ticks_real[keep]], fontsize=14)
ax.zaxis.set_label_position("upper") 
ax.tick_params(axis="z", pad=6)

# ---- 左侧 colorbar ----
cax = fig.add_axes([0.08, 0.25, 0.02, 0.5])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("Reaction pressure (bar)", rotation=90, labelpad=12, fontsize=14)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()