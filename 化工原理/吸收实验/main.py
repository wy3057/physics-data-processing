# -*- coding: utf-8 -*-
"""
填料吸收塔实验数据自动处理（Excel 表格输入版）

使用前准备：
1. 同目录下放一个 Excel 文件：experiment_data.xlsx
2. 其中包含三个工作表：
   - dry          ：干塔流体力学
   - wet          ：湿塔流体力学
   - mass_transfer：传质实验数据

表头格式见脚本开头注释。
运行结果：
- 终端打印 3 个处理后的数据表
- 生成图像：
  - dp_u_curve.png  : 干/湿塔 (ΔP/Z)-u 曲线
  - Kxa_vs_L.png    : 传质系数 Kxa-L 曲线
- 如需导出 Excel，可在 main 里打开相应 to_excel 语句。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 1. 实验装置参数（可按需要修改） =====================

COLUMN_DIAMETER_M = 0.10   # 塔径 D, m
PACKING_HEIGHT_M = 0.66    # 填料高度 Z, m

R = 8.314                  # J/(mol·K)
ATM_PRESSURE_KPA = 101.3   # kPa
WATER_DENSITY_KG_M3 = 1000.0
MW_WATER_KG_PER_KMOL = 18.0

# 亨利常数表（CO2-水）
HENRY_TEMP_C = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60], dtype=float)
HENRY_E_1E5 = np.array([0.738, 0.888, 1.05, 1.24, 1.44, 1.66, 1.88,
                        2.12, 2.36, 2.60, 2.87, 3.46], dtype=float)
HENRY_E_KPA = HENRY_E_1E5 * 1e5   # 实际 E (kPa)


# ===================== 2. 工具函数 =====================

def column_cross_section_area(diameter_m: float) -> float:
    """塔截面积 A (m2)"""
    return np.pi * diameter_m ** 2 / 4.0


def henry_constant_CO2_water(T_C: float) -> float:
    """插值 CO2 在水中的亨利常数 E (kPa)"""
    return float(np.interp(T_C, HENRY_TEMP_C, HENRY_E_KPA))


def volumetric_flow_Lmin_to_m3h(V_L_min: float) -> float:
    """体积流量 L/min -> m3/h"""
    return V_L_min / 1000.0 * 60.0


def gas_molar_flux_from_vol_flow(V_m3_h: float,
                                 T_C: float,
                                 P_kPa: float,
                                 area_m2: float) -> float:
    """体积流量 V (m3/h) -> 摩尔通量 G (kmol/m2·h)"""
    T_K = T_C + 273.15
    P_Pa = P_kPa * 1000.0
    n_dot_mol_h = P_Pa * V_m3_h / (R * T_K)
    n_dot_kmol_h = n_dot_mol_h / 1000.0
    return n_dot_kmol_h / area_m2


def liquid_molar_flux_from_vol_flow(V_L_h: float, area_m2: float) -> float:
    """水的体积流量 V (L/h) -> 摩尔通量 L (kmol/m2·h)"""
    V_m3_h = V_L_h / 1000.0
    mass_kg_h = V_m3_h * WATER_DENSITY_KG_M3
    n_dot_kmol_h = mass_kg_h / MW_WATER_KG_PER_KMOL
    return n_dot_kmol_h / area_m2


def y_to_Y(y: float) -> float:
    """摩尔分率 y -> 溶质无惰组分摩尔比 Y"""
    return y / (1.0 - y)


def Y_to_y(Y: float) -> float:
    """溶质无惰组分摩尔比 Y -> 摩尔分率 y"""
    return Y / (1.0 + Y)


def compute_A(L: float, G: float, m: float) -> float:
    """吸收因数 A"""
    return L / (m * G)


def integral_NOL_numeric(G: float, L: float, Y1: float, Y2: float,
                         E_kPa: float, P_kPa: float,
                         n_steps: int = 2000) -> float:
    """
    数值积分计算 NOL:
        NOL = ∫ dX / (X* - X)
    采用操作线 & 平衡线关系，见之前说明。
    """
    X2 = 0.0
    X1 = G * (Y1 - Y2) / L  # 全塔物料衡算

    xs = np.linspace(X2 + 1e-12, X1, n_steps)

    def Y_of_X(X):
        return Y2 + (L / G) * (X - X2)

    def X_star(Y):
        y = Y_to_y(Y)
        x_star = y * P_kPa / E_kPa
        return x_star

    integrand = []
    for x in xs:
        Y = Y_of_X(x)
        x_eq = X_star(Y)
        dx = x_eq - x
        dx = max(dx, 1e-16)
        integrand.append(1.0 / dx)

    NOL = np.trapz(integrand, xs)
    return float(NOL)


# ===================== 3. 流体力学数据处理 =====================

def process_hydrodynamics(dry_df: pd.DataFrame,
                          wet_df: pd.DataFrame,
                          Z_m: float,
                          D_m: float):
    """
    输入：
        dry_df: 列 ['DeltaP_mmH2O', 'V_m3_h']
        wet_df: 列 ['DeltaP_mmH2O', 'V_m3_h', 'L_L_h', 'Phenomenon'(可选)]
    输出：
        加了 'u_m_s', 'DeltaP_per_Z_mmH2O_per_m' 的新 DataFrame
    """
    A = column_cross_section_area(D_m)

    def add_cols(df):
        df = df.copy()
        df['u_m_s'] = df['V_m3_h'] / A / 3600.0
        df['DeltaP_per_Z_mmH2O_per_m'] = df['DeltaP_mmH2O'] / Z_m
        return df

    dry_processed = add_cols(dry_df)
    wet_processed = add_cols(wet_df)
    return dry_processed, wet_processed


def plot_dp_u(dry_df: pd.DataFrame,
              wet_df: pd.DataFrame,
              save_path: str = 'dp_u_curve.png'):
    """画 (ΔP/Z)-u 曲线（双对数坐标）"""
    plt.figure(figsize=(6, 4))

    dry_plot = dry_df[dry_df['u_m_s'] > 0]
    wet_plot = wet_df[wet_df['u_m_s'] > 0]

    plt.loglog(dry_plot['u_m_s'], dry_plot['DeltaP_per_Z_mmH2O_per_m'],
               marker='o', linestyle='-', label='干填料')
    plt.loglog(wet_plot['u_m_s'], wet_plot['DeltaP_per_Z_mmH2O_per_m'],
               marker='s', linestyle='-', label='湿填料')

    plt.xlabel('空塔气速 u (m/s)')
    plt.ylabel('单位高度压降 ΔP/Z (mmH2O/m)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ===================== 4. 传质数据处理 =====================

def process_mass_transfer(df_raw: pd.DataFrame,
                          D_m: float,
                          Z_m: float,
                          P_kPa: float = ATM_PRESSURE_KPA) -> pd.DataFrame:
    """
    输入 DataFrame 列：
        V_CO2_L_min, V_air_L_min, L_L_h,
        T_liq_out_C, y_in, y_out, (可选 T_gas_C)

    输出增加：
        V_CO2_m3_h, V_air_m3_h, V_total_m3_h
        G_kmol_m2_h, L_kmol_m2_h
        E_kPa, m_eq, A, Y1, Y2, X1
        NOL, Kxa_kmol_m3_h, absorption_eta_percent
    """
    df = df_raw.copy()
    A_col = column_cross_section_area(D_m)

    # 体积流量换算
    df['V_CO2_m3_h'] = df['V_CO2_L_min'].apply(volumetric_flow_Lmin_to_m3h)
    df['V_air_m3_h'] = df['V_air_L_min'].apply(volumetric_flow_Lmin_to_m3h)
    df['V_total_m3_h'] = df['V_CO2_m3_h'] + df['V_air_m3_h']

    # 气体温度
    if 'T_gas_C' not in df.columns:
        df['T_gas_C'] = df['T_liq_out_C']

    # G, L
    df['G_kmol_m2_h'] = df.apply(
        lambda row: gas_molar_flux_from_vol_flow(
            V_m3_h=row['V_total_m3_h'],
            T_C=row['T_gas_C'],
            P_kPa=P_kPa,
            area_m2=A_col
        ),
        axis=1
    )

    df['L_kmol_m2_h'] = df.apply(
        lambda row: liquid_molar_flux_from_vol_flow(
            V_L_h=row['L_L_h'],
            area_m2=A_col
        ),
        axis=1
    )

    # 亨利常数 & 平衡线斜率
    df['E_kPa'] = df['T_liq_out_C'].apply(henry_constant_CO2_water)
    df['m_eq'] = df['E_kPa'] / P_kPa

    # y -> Y
    df['Y1'] = df['y_in'].apply(y_to_Y)
    df['Y2'] = df['y_out'].apply(y_to_Y)

    # 吸收因数
    df['A'] = df.apply(
        lambda row: compute_A(
            L=row['L_kmol_m2_h'],
            G=row['G_kmol_m2_h'],
            m=row['m_eq']
        ),
        axis=1
    )

    # X1 由物料衡算
    df['X1'] = df.apply(
        lambda row: row['G_kmol_m2_h'] * (row['Y1'] - row['Y2']) / row['L_kmol_m2_h'],
        axis=1
    )

    # NOL & Kxa
    NOL_list = []
    Kxa_list = []

    for _, row in df.iterrows():
        G = row['G_kmol_m2_h']
        L = row['L_kmol_m2_h']
        Y1 = row['Y1']
        Y2 = row['Y2']
        E_kPa = row['E_kPa']

        NOL = integral_NOL_numeric(
            G=G, L=L,
            Y1=Y1, Y2=Y2,
            E_kPa=E_kPa, P_kPa=P_kPa,
            n_steps=3000
        )

        Kxa = L * NOL / Z_m

        NOL_list.append(NOL)
        Kxa_list.append(Kxa)

    df['NOL'] = NOL_list
    df['Kxa_kmol_m3_h'] = Kxa_list

    # 吸收率
    df['absorption_eta_percent'] = (df['y_in'] - df['y_out']) / df['y_in'] * 100.0

    return df


def plot_Kxa_vs_L(df: pd.DataFrame,
                  save_path: str = 'Kxa_vs_L.png'):
    """画 Kxa-L 曲线"""
    plt.figure(figsize=(6, 4))
    plt.plot(df['L_kmol_m2_h'], df['Kxa_kmol_m3_h'],
             marker='o', linestyle='-')
    plt.xlabel('液体摩尔通量 L (kmol/m2·h)')
    plt.ylabel('Kxa (kmol/m3·h)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ===================== 5. 主程序：从 Excel 读表格 =====================

def main():
    excel_file = 'experiment_data.xlsx'

    # 读入三个工作表
    dry_df_raw = pd.read_excel(excel_file, sheet_name='dry')
    wet_df_raw = pd.read_excel(excel_file, sheet_name='wet')
    mt_df_raw = pd.read_excel(excel_file, sheet_name='mass_transfer')

    # 流体力学部分
    dry_df, wet_df = process_hydrodynamics(
        dry_df_raw, wet_df_raw,
        Z_m=PACKING_HEIGHT_M,
        D_m=COLUMN_DIAMETER_M
    )

    print("=== 干填料流体力学数据处理结果 ===")
    print(dry_df)
    print("\n=== 湿填料流体力学数据处理结果 ===")
    print(wet_df)

    plot_dp_u(dry_df, wet_df, save_path='dp_u_curve.png')
    print("\n已生成干 / 湿塔 (ΔP/Z)-u 曲线: dp_u_curve.png")

    # 传质部分
    mt_df = process_mass_transfer(
        mt_df_raw,
        D_m=COLUMN_DIAMETER_M,
        Z_m=PACKING_HEIGHT_M,
        P_kPa=ATM_PRESSURE_KPA
    )

    print("\n=== 传质实验数据处理结果 ===")
    pd.set_option('display.max_columns', None)
    print(mt_df)

    plot_Kxa_vs_L(mt_df, save_path='Kxa_vs_L.png')
    print("\n已生成 Kxa-L 曲线: Kxa_vs_L.png")

    # 如需要，把结果再导出成 Excel：
    # dry_df.to_excel('dry_processed.xlsx', index=False)
    # wet_df.to_excel('wet_processed.xlsx', index=False)
    # mt_df.to_excel('mass_transfer_processed.xlsx', index=False)


if __name__ == '__main__':
    main()
