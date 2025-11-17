import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 1. 指定一个支持中文的字体（Windows 一般有 SimHei 或 Microsoft YaHei）
rcParams['font.sans-serif'] = ['SimHei', 'FangSong_GB2312']          # 或 ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False           # 解决负号显示为方块的问题


# ==================== 1. 基本设置：文件与实验常数 ====================

# 数据文件名（和脚本放在同一文件夹下）
DATA_PATH =input("请输入数据文件路径 (例如: experiment_data.csv): ")

# --- 实验常数：请按你自己的实验填写 ---
GD = float(input("请输入Supporting Drying 称量 GD（g）: "))       # 支撑架重量 GD，g（请改成你自己的）
GC = float(input("请输入Supporting 绝干物料量 GC"))   # 绝干物料量 GC，g（请改成你自己的）
S = float(input("请输入干燥面积 S（m^2）: "))    # 干燥面积 S，m^2（示例：0.17×0.080×2，可按实际修改）

# 孔板流量计参数（按装置说明书）
C0 = 0.65          # 流量系数
d0 = 0.035         # 孔径，m
A0 = np.pi * (d0 ** 2) / 4  # 孔板开孔面积，m^2

# 洞道截面积（用于求空气流速）
A_tunnel = 0.15 * 0.20   # m^2（说明书示例：0.15×0.20，如有不同请改）

# 湿球温度对应的汽化潜热（近似，按 2 400~2 450 kJ/kg 取值）
r_tw = 2430e3      # J/kg

# 列名（与你的表保持一致）
COL_TIME = "time_min"
COL_GT = "GT_g"
COL_TW = "湿球温度摄氏度"
COL_DP = "压差（kpa）"
COL_TD = "干球温度"
COL_TIN = "入口温度"

# 输出文件名
OUT_EXCEL = "干燥数据_处理结果.xlsx"
FIG_X_T = "干燥曲线_X_t.png"
FIG_U_X = "干燥速率曲线_U_Xav.png"


# ==================== 2. 读入原始数据 ====================

def load_raw_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到数据文件：{p.resolve()}")

    if p.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    # 简单检查
    required_cols = [COL_TIME, COL_GT, COL_TW, COL_DP, COL_TD, COL_TIN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据文件中缺少必要列：'{col}'")

    return df


# ==================== 3. 计算 X, X_av, U 等 ====================

def compute_moisture_and_rate(df: pd.DataFrame) -> pd.DataFrame:
    """计算：
       - Gi：被干燥物料重量，g
       - X：干基含水量，kg水/kg绝干物料
       - X_av：平均含水量
       - U：干燥速率，kg/(m2·s)
    """
    df = df.copy()

    # 时间：min -> s
    df["time_s"] = df[COL_TIME] * 60.0

    # 被干燥物料重量 Gi = GT - GD（g）
    df["Gi_g"] = df[COL_GT] - GD

    # 干基含水量 X = (Gi - GC) / GC （g/g == kg/kg）
    df["X"] = (df["Gi_g"] - GC) / GC

    # 平均含水量 X_av：相邻两点平均
    X = df["X"].to_numpy()
    X_av = np.empty_like(X)
    X_av[:-1] = 0.5 * (X[:-1] + X[1:])
    X_av[-1] = np.nan
    df["X_av"] = X_av

    # 干燥速率 U = Δm / (S * Δt)
    Gi_g = df["Gi_g"].to_numpy()
    time_s = df["time_s"].to_numpy()
    U = np.empty_like(time_s, dtype=float)
    U[:] = np.nan

    for i in range(1, len(df)):
        dm_kg = (Gi_g[i - 1] - Gi_g[i]) / 1000.0  # kg
        dt_s = time_s[i] - time_s[i - 1]
        if dt_s <= 0:
            U[i] = np.nan
        else:
            U[i] = dm_kg / (S * dt_s)  # kg/(m2·s)

    df["U_kg_m2_s"] = U
    df["U_x1e4"] = df["U_kg_m2_s"] * 1e4  # 为画图方便，乘 10^-4

    return df


# ==================== 4. 自动识别恒速段、求 Xc、Uc ====================

def find_constant_rate_segment(df: pd.DataFrame):
    """
    简单自动识别恒速段：
    - 在 U 的最大值附近，找 U >= 0.9 * U_max 的连续区间
    - 取其中最长的一段作为恒速段
    返回：恒速段索引列表、Uc（平均）、Xc（恒速段末点的 X）
    """
    U = df["U_kg_m2_s"].to_numpy()
    X = df["X"].to_numpy()

    valid = ~np.isnan(U)
    if valid.sum() < 3:
        return [], np.nan, np.nan

    U_valid = U[valid]
    idx_valid = np.where(valid)[0]

    U_max = np.nanmax(U_valid)
    if U_max <= 0:
        return [], np.nan, np.nan

    threshold = 0.9 * U_max
    mask = U_valid >= threshold

    longest_len = 0
    longest_start = None
    current_start = None

    for i, flag in enumerate(mask):
        if flag and current_start is None:
            current_start = i
        elif (not flag) and current_start is not None:
            length = i - current_start
            if length > longest_len:
                longest_len = length
                longest_start = current_start
            current_start = None

    # 末尾收尾
    if current_start is not None:
        length = len(mask) - current_start
        if length > longest_len:
            longest_len = length
            longest_start = current_start

    if longest_start is None or longest_len <= 1:
        return [], np.nan, np.nan

    const_valid_idx = np.arange(longest_start, longest_start + longest_len)
    const_idx = idx_valid[const_valid_idx]

    Uc = np.nanmean(U[const_idx])
    Xc = X[const_idx[-1]]

    return const_idx.tolist(), Uc, Xc


# ==================== 5. 计算 h、流量、流速 ====================

def compute_h_and_flow(df: pd.DataFrame, const_idx, Uc):
    """用恒速段的平均温度和压差计算：
       - 对流传热系数 h
       - 孔板处体积流量 Q0
       - 干燥室内体积流量 Q
       - 流速 v
    """
    if not const_idx or np.isnan(Uc):
        return np.nan, np.nan, np.nan, np.nan

    # 恒速段平均温度、压差
    t_dry = df.loc[const_idx, COL_TD].mean()
    t_wet = df.loc[const_idx, COL_TW].mean()
    t0 = df.loc[const_idx, COL_TIN].mean()
    deltaP_kPa = df.loc[const_idx, COL_DP].mean()
    deltaP = deltaP_kPa * 1e3  # kPa -> Pa

    # 对流传热系数 h = Uc * r_tw / (t_dry - t_wet)
    if t_dry == t_wet:
        h = np.nan
    else:
        h = Uc * r_tw / (t_dry - t_wet)

    # 空气密度（近似理想气体，0℃时 1.293 kg/m3）
    rho0 = 1.293 * 273.15 / (273.15 + t0)

    # 孔板处体积流量 Q0 = C0 * A0 * sqrt(2ΔP / rho0)
    Q0 = C0 * A0 * np.sqrt(2 * deltaP / rho0)

    # 温度修正到干燥器内：Q = Q0 * T0 / Td
    T0_K = 273.15 + t0
    Td_K = 273.15 + t_dry
    Q = Q0 * T0_K / Td_K

    # 洞道内平均流速
    v = Q / A_tunnel

    return h, Q0, Q, v


# ==================== 6. 绘图 ====================

def plot_curves(df: pd.DataFrame, const_idx, Xc, Uc):
    # --- 干燥曲线 X-t ---
    plt.figure()
    plt.plot(df[COL_TIME], df["X"], marker="o")
    plt.xlabel("时间 t / min")
    plt.ylabel("干基含水量 X / kg水·kg⁻¹绝干物料")
    plt.title("干燥曲线 X-t")

    # 标出临界点
    if const_idx and not np.isnan(Xc):
        t_c = df.loc[const_idx[-1], COL_TIME]
        plt.axvline(t_c, linestyle="--")
        plt.text(t_c, Xc, f"  Xc={Xc:.3f}", va="bottom")

    plt.tight_layout()
    plt.savefig(FIG_X_T, dpi=300)

    # --- 干燥速率曲线 U-X_av ---
    plt.figure()
    mask = ~df["X_av"].isna() & ~df["U_kg_m2_s"].isna()
    plt.plot(df.loc[mask, "X_av"],
             df.loc[mask, "U_kg_m2_s"] * 1e4,
             marker="o")
    plt.xlabel("平均含水量 X_av / kg水·kg⁻¹绝干物料")
    plt.ylabel("干燥速率 U × 10⁴ / kg·m⁻²·s⁻¹")
    plt.title("干燥速率曲线 U-X_av")

    # 标出恒速段
    if const_idx:
        plt.scatter(df.loc[const_idx, "X_av"],
                    df.loc[const_idx, "U_kg_m2_s"] * 1e4)

    plt.tight_layout()
    plt.savefig(FIG_U_X, dpi=300)


# ==================== 7. 主程序 ====================

def main():
    # 1) 读入数据
    df_raw = load_raw_data(DATA_PATH)

    # 2) 计算含水量、干燥速率
    df = compute_moisture_and_rate(df_raw)

    # 3) 恒速段、Xc、Uc
    const_idx, Uc, Xc = find_constant_rate_segment(df)
    df["is_const_rate"] = False
    if const_idx:
        df.loc[const_idx, "is_const_rate"] = True

    # 4) 计算 h、流量、流速
    h, Q0, Q, v = compute_h_and_flow(df, const_idx, Uc)

    # 5) 输出到 Excel（一个 sheet：数据 + 结果）
    with pd.ExcelWriter(OUT_EXCEL) as writer:
        df.to_excel(writer, index=False, sheet_name="处理数据")

        summary = pd.DataFrame({
            "参数": ["临界含水量 Xc", "恒速干燥速率 Uc",
                   "对流传热系数 h", "孔板处体积流量 Q0",
                   "干燥室内体积流量 Q", "流速 v"],
            "数值": [Xc,
                   Uc,
                   h,
                   Q0,
                   Q,
                   v],
            "单位": ["kg水/kg绝干物料",
                   "kg/(m²·s)",
                   "W/(m²·K)",
                   "m³/s",
                   "m³/s",
                   "m/s"]
        })
        summary.to_excel(writer, index=False, sheet_name="计算结果")

    # 6) 控制台打印关键结果
    print(f"数据处理完成，已保存到：{OUT_EXCEL}\n")
    print("=== 关键结果 ===")
    if not np.isnan(Xc):
        print(f"临界含水量 Xc = {Xc:.4f} kg水/kg绝干物料")
    else:
        print("未能自动识别临界含水量 Xc")

    if not np.isnan(Uc):
        print(f"恒速干燥速率 Uc = {Uc:.4e} kg/(m²·s)")
    else:
        print("未能自动识别恒速干燥速率 Uc")

    if not np.isnan(h):
        print(f"对流传热系数 h = {h:.2f} W/(m²·K)")
    else:
        print("未能计算对流传热系数 h（请检查数据）")

    print(f"孔板处体积流量 Q0 = {Q0:.4e} m³/s")
    print(f"干燥室内体积流量 Q = {Q:.4e} m³/s")
    print(f"流速 v = {v:.4f} m/s")

    # 7) 画图并保存
    plot_curves(df, const_idx, Xc, Uc)
    print(f"已生成图像：{FIG_X_T}, {FIG_U_X}")


if __name__ == "__main__":
    main()
