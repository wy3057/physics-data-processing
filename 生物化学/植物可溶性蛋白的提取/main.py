# -*- coding: utf-8 -*-
"""
实验2：植物可溶性蛋白的提取及考马斯亮蓝法测定蛋白质浓度
数据处理自动化脚本（根据实验报告中的具体数据编写）

功能：
1. 根据表1中标准蛋白体积和 A595，自动计算标准蛋白浓度(µg/mL)、拟合标准曲线。
2. 绘制标准曲线图像并保存为 standard_curve.png。
3. 根据表2中的样品 A595，利用标准曲线计算样品蛋白浓度：
   - 比色体系中的蛋白浓度 (µg/mL)
   - 按 0.1 mL 样品 + 5 mL 显色剂 的稀释倍数换算回“原提取液中的蛋白浓度” (mg/mL)。
4. 将所有结果输出为 Excel 文件 protein_results.xlsx。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体（选择一种你系统有的）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# 如果上面不行，可以改成：['Microsoft YaHei'] 或 ['SimSun']

# 解决坐标轴负号显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# ================================
# 一、根据实验报告设置的固定参数
# ================================

# 标准蛋白溶液浓度：1 mg/mL（结晶牛血清蛋白）  →  1000 µg/mL
STOCK_CONC_MG_PER_ML = 1.0      # mg/mL
STOCK_CONC_UG_PER_ML = STOCK_CONC_MG_PER_ML * 1000  # 1000 µg/mL

# 反应体系总体积：0.1 mL（蛋白溶液或样品 + NaCl）+ 5 mL 考马斯亮蓝试剂
REACTION_TOTAL_VOLUME_ML = 0.1 + 5.0  # = 5.1 mL

# 样品加入体积：0.1 mL
SAMPLE_VOLUME_ML = 0.1

# 从比色体系浓度换算到“原样品提取液”的稀释倍数
DILUTION_FACTOR = REACTION_TOTAL_VOLUME_ML / SAMPLE_VOLUME_ML  # 5.1 / 0.1 = 51

# ================================
# 二、表1：标准曲线数据（来自实验报告）
# ================================

# 试管编号 0~6，对应 1 mg/mL 标准蛋白溶液加入体积（mL）
standard_tube_ids = np.array([0, 1, 2, 3, 4, 5, 6])
standard_volumes_ml = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

# 含 0.15 mol/L NaCl 的体积 0.1 mL（体系中蛋白溶液+NaCl总体积恒为 0.1 mL）
nacl_volumes_ml = 0.1 - standard_volumes_ml  # 0.1, 0.09, ... 0.04

# 表1中给出的 A595nm（只有一组数据）
A595_std = np.array([0.000, 0.230, 0.487, 0.527, 0.627, 0.729, 0.807])

# ================================
# 三、表2：未知样品 A595 数据（来自实验报告）
# ================================
# 对照（空白）A595 = 0，仅作为说明用，这里标准和样品数据都已经默认扣除空白
A595_blank = 0.0

# 各样品的 A595 值（如果以后有平行测定，只需要把列表写成多个值即可）
samples_abs = {
    "样品1": [0.281],
    "样品2": [0.284],
    "样品3": [0.340],
}


# ================================
# 四、函数定义
# ================================

def build_standard_dataframe():
    """
    根据标准蛋白体积和 A595 构建标准曲线数据表：
    - 计算蛋白加入量 (µg)
    - 计算反应体系中蛋白浓度 (µg/mL) = 蛋白量 / 5.1 mL
    """
    # 加入的蛋白质量 (µg)
    protein_amount_ug = standard_volumes_ml * STOCK_CONC_UG_PER_ML

    # 反应体系中蛋白浓度 (µg/mL)
    protein_conc_ug_per_ml = protein_amount_ug / REACTION_TOTAL_VOLUME_ML

    df_std = pd.DataFrame({
        "试管编号": standard_tube_ids,
        "标准蛋白体积 (mL)": standard_volumes_ml,
        "0.15 mol/L NaCl 体积 (mL)": nacl_volumes_ml,
        "蛋白加入量 (µg)": protein_amount_ug,
        "反应体系中蛋白浓度 (µg/mL)": protein_conc_ug_per_ml,
        "A595nm": A595_std
    })

    return df_std


def fit_standard_curve(df_std):
    """
    对标准曲线做线性回归：
    以 反应体系中蛋白浓度 (µg/mL) 为横坐标 x，
      A595nm 为纵坐标 y，
    拟合方程：A = k * C + b
    返回：k, b, R^2
    """
    x = df_std["反应体系中蛋白浓度 (µg/mL)"].values
    y = df_std["A595nm"].values

    # 一元线性拟合：返回斜率 k 和截距 b
    k, b = np.polyfit(x, y, 1)

    # 计算决定系数 R^2
    y_pred = k * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return k, b, r2


def plot_standard_curve(df_std, k, b, r2, out_file="standard_curve.png"):
    """
    绘制标准曲线图像并保存为 PNG 文件。
    横坐标：反应体系中蛋白浓度 (µg/mL)
    纵坐标：A595nm
    """
    x = df_std["反应体系中蛋白浓度 (µg/mL)"].values
    y = df_std["A595nm"].values

    x_line = np.linspace(0, x.max() * 1.1, 100)
    y_line = k * x_line + b

    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(x, y, label="标准点")
    plt.plot(x_line, y_line, label="线性拟合", linewidth=1.5)

    # 注意：用 mathtext 写 µ 和上标
    plt.xlabel(r"蛋白浓度 C ($\mu$g/mL, 反应体系)")
    plt.ylabel("A595nm")
    plt.title("蛋白质标准曲线（考马斯亮蓝法）")

    # 在图中标记线性方程与 R^2（也用 mathtext）
    eq_text = rf"A = {k:.4f} \times C + {b:.4f}\n$R^2$ = {r2:.4f}"
    plt.text(0.05 * x.max(), 0.80 * y.max(),
             eq_text, fontsize=10,
             bbox=dict(facecolor="white", alpha=0.7))

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def build_sample_dataframe(k, b):
    """
    根据样品 A595 和标准曲线 A = kC + b 计算样品浓度。

    对每个样品：
    1. 求 A 平均值（如果有多次平行测定）。
    2. 根据 C = (A - b) / k 计算“比色体系中的蛋白浓度 C_assay (µg/mL)”。
    3. 再乘以稀释倍数 51，得到“原样品提取液中的蛋白浓度 C_sample (µg/mL)”。
    4. 同时给出 mg/mL 形式，方便实验报告书写。
    """
    records = []

    for name, A_list in samples_abs.items():
        A_array = np.array(A_list, dtype=float)
        A_mean = A_array.mean()
        A_std = A_array.std(ddof=1) if len(A_array) > 1 else np.nan

        # 利用标准曲线求比色体系中蛋白浓度（µg/mL）
        C_assay_ug_per_ml = (A_mean - b) / k

        # 换算回原样品提取液中的蛋白浓度
        C_sample_ug_per_ml = C_assay_ug_per_ml * DILUTION_FACTOR
        C_sample_mg_per_ml = C_sample_ug_per_ml / 1000.0

        records.append({
            "样品名称": name,
            "A595各次测定": ", ".join(f"{v:.3f}" for v in A_array),
            "A595平均值": A_mean,
            "A595标准差": A_std,
            "比色体系中蛋白浓度 (µg/mL)": C_assay_ug_per_ml,
            "原提取液中蛋白浓度 (µg/mL)": C_sample_ug_per_ml,
            "原提取液中蛋白浓度 (mg/mL)": C_sample_mg_per_ml
        })

    df_samples = pd.DataFrame(records)
    return df_samples


def save_to_excel(df_std, df_samples, k, b, r2,
                  out_file="protein_results.xlsx"):
    """
    将标准曲线数据、拟合参数、样品结果写入 Excel 文件
    """
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        # 标准曲线数据
        df_std.to_excel(writer, sheet_name="标准曲线数据", index=False)

        # 标准曲线拟合参数
        df_params = pd.DataFrame({
            "参数": ["斜率 k", "截距 b", "R²",
                   "横坐标含义", "反应体系总体积 (mL)", "样品加入体积 (mL)", "稀释倍数"],
            "数值": [
                k,
                b,
                r2,
                "蛋白浓度 C (µg/mL, 反应体系)",
                REACTION_TOTAL_VOLUME_ML,
                SAMPLE_VOLUME_ML,
                DILUTION_FACTOR
            ]
        })
        df_params.to_excel(writer, sheet_name="标准曲线参数", index=False)

        # 样品结果
        df_samples.to_excel(writer, sheet_name="样品结果", index=False)


def main():
    out_dir = Path(".")
    out_dir.mkdir(exist_ok=True)

    # 1) 标准曲线数据
    df_std = build_standard_dataframe()
    print("=== 标准曲线数据 ===")
    print(df_std)
    print()

    # 2) 拟合标准曲线
    k, b, r2 = fit_standard_curve(df_std)
    print(f"线性回归结果：A = {k:.4f} × C + {b:.4f}")
    print(f"R² = {r2:.4f}\n")

    # 3) 绘制标准曲线图像
    curve_img = out_dir / "standard_curve.png"
    plot_standard_curve(df_std, k, b, r2, out_file=str(curve_img))
    print(f"标准曲线图已保存为：{curve_img.resolve()}\n")

    # 4) 样品浓度计算
    df_samples = build_sample_dataframe(k, b)
    print("=== 样品结果 ===")
    print(df_samples)
    print()

    # 5) 导出 Excel
    excel_file = out_dir / "protein_results.xlsx"
    save_to_excel(df_std, df_samples, k, b, r2, out_file=str(excel_file))
    print(f"结果已导出到：{excel_file.resolve()}")


if __name__ == "__main__":
    main()
