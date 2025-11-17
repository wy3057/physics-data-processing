# -*- coding: utf-8 -*-
"""
乙醇-正丙醇 精馏实验数据自动处理脚本

功能：
1. 折光指数 -> 质量分数 -> 摩尔分数
2. 利用 t-x-y 表插值获得平衡曲线 y_eq(x)、x_eq(y)、泡点温度 t_bp(x)
3. 计算 q 值、q 线，绘制 McCabe-Thiele 图
   - 全回流（R = ∞）：操作线 = 对角线
   - 部分回流（有限 R，给定进料热状况）
4. 计算：
   - 理论板数 N_theoretical
   - 全塔效率 E_T = N_theoretical / N_actual
   - 单板效率（以液相浓度表示的 Murphree 效率）
5. 输出结果表为 pandas.DataFrame，并保存成 CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================
# 一、基本物性与插值数据
# ===========================

# 乙醇 / 正丙醇 的分子量
M_EtOH = 46.0  # 乙醇
M_nPrOH = 60.0  # 正丙醇

# 表2: 乙醇–正丙醇 t-x-y 数据 (以乙醇摩尔分率表示)
# t / ℃
T_data = np.array([97.60, 93.85, 92.66, 91.60, 88.32,
                   86.25, 84.98, 84.13, 83.06, 80.50, 78.38])
# x: 液相乙醇摩尔分数
x_data = np.array([0.0, 0.126, 0.188, 0.210, 0.358,
                   0.461, 0.546, 0.600, 0.663, 0.884, 1.0])
# y: 气相乙醇摩尔分数
y_data = np.array([0.0, 0.240, 0.318, 0.349, 0.550,
                   0.650, 0.711, 0.760, 0.799, 0.914, 1.0])


# ===========================
# 二、物性和基本计算函数
# ===========================

def nd_to_mass_fraction(nd: float) -> float:
    """
    折光指数 nD -> 乙醇质量分数 W
    说明书给出的回归式（30℃）:
        W = 58.844116 - 42.61325 * nD
    返回值 W 为 0~1 之间的质量分数。
    """
    W = 58.844116 - 42.61325 * nd
    return W


def mass_to_mole_fraction(W: float,
                          M_A: float = M_EtOH,
                          M_B: float = M_nPrOH) -> float:
    """
    质量分数 W_A -> 摩尔分数 x_A
    二元系公式:
        x_A = (W_A / M_A) / [W_A / M_A + (1 - W_A) / M_B]
    """
    n_A = W / M_A
    n_B = (1.0 - W) / M_B
    x_A = n_A / (n_A + n_B)
    return x_A


def y_eq(x: float) -> float:
    """
    根据 x（液相乙醇摩尔分数）插值得到平衡气相 y_eq(x)
    """
    return float(np.interp(x, x_data, y_data))


def x_eq_from_y(y: float) -> float:
    """
    根据 y（气相乙醇摩尔分数）插值得到平衡液相 x_eq(y)
    即“与该气相相平衡的液相浓度”
    """
    return float(np.interp(y, y_data, x_data))


def bubble_point_temp(x: float) -> float:
    """
    根据进料液相摩尔分数 x_F 插值获得泡点温度 t_BP(x_F)
    """
    return float(np.interp(x, x_data, T_data))


def mixture_cp_and_r(x: float,
                     Cp1: float, Cp2: float,
                     r1: float, r2: float) -> tuple[float, float]:
    """
    混合物的比热 Cpm 和汽化潜热 rpm（按说明书公式）
      Cpm = M1 * x1 * Cp1 + M2 * x2 * Cp2  [kJ/(kmol·℃)]
      rpm = M1 * x1 * r1 + M2 * x2 * r2   [kJ/kmol]
    其中：
        x1 = x (乙醇摩尔分数), x2 = 1 - x
        Cp1, Cp2 单位 kJ/(kg·℃)
        r1, r2  单位 kJ/kg
    """
    x1 = x
    x2 = 1.0 - x
    Cpm = M_EtOH * x1 * Cp1 + M_nPrOH * x2 * Cp2
    rpm = M_EtOH * x1 * r1 + M_nPrOH * x2 * r2
    return Cpm, rpm


def calc_q(t_F: float, x_F: float, Cp1: float, Cp2: float,
           r1: float, r2: float) -> tuple[float, float]:
    """
    计算进料热状况参数 q 及泡点温度 t_BP。
    采用常见公式：
        t_BP = bubble_point_temp(x_F)
        混合物物性: Cpm, rpm
        q = 1 + Cpm * (t_BP - t_F) / rpm
      (t_F < t_BP 时为过冷液体，q > 1)
    """
    t_BP = bubble_point_temp(x_F)
    Cpm, rpm = mixture_cp_and_r(x_F, Cp1, Cp2, r1, r2)
    q = 1.0 + Cpm * (t_BP - t_F) / rpm
    return q, t_BP


# ===========================
# 三、McCabe-Thiele 计算
# ===========================

def mccabe_thiele_total_reflux(x_D: float,
                               x_W: float,
                               max_stages: int = 50):
    """
    全回流 (R = ∞) 时的 McCabe-Thiele 计算
    操作线退化为 y = x（对角线）。
    返回:
        stages_N: 近似理论板数（可包含最后一块的分板）
        x_points, y_points: 阶梯线转折点坐标（用于绘图）
    算法：在平衡线和平分线之间“横-竖”阶梯
    """
    x_points = [x_D]
    y_points = [x_D]  # 初始在对角线上 (x_D, y_D=x_D)

    x_curr = x_D
    y_curr = x_D
    full_stages = 0

    while x_curr > x_W and full_stages < max_stages:
        # 1) 水平到平衡线：已知 y_curr，求 x_eq(y_curr)
        x_new = x_eq_from_y(y_curr)
        y_new = y_curr
        x_points.append(x_new)
        y_points.append(y_new)

        # 2) 垂直到对角线 y=x
        x_curr = x_new
        y_curr = x_new
        x_points.append(x_curr)
        y_points.append(y_curr)

        full_stages += 1

        if x_curr <= x_W:
            break

    # 简单处理分板：用线性比例估算 (可按需要改进)
    if x_curr > x_W and full_stages > 0:
        # 上一个板底液浓度
        x_prev_bottom = x_points[-3]  # 每块板增加两个点
        frac = (x_prev_bottom - x_W) / (x_prev_bottom - x_curr)
        stages_N = full_stages - (1 - frac)
    else:
        stages_N = full_stages

    return stages_N, np.array(x_points), np.array(y_points)


def mccabe_thiele_partial_reflux(x_D: float, x_W: float, x_F: float,
                                 R: float, q: float,
                                 max_stages: int = 50):
    """
    部分回流条件下的 McCabe-Thiele 计算（给定 x_D, x_W, x_F, R, q）

    操作线：
      Rectifying line (精馏段):
        y = (R/(R+1)) * x + x_D/(R+1)
      q-line:
        y = (q/(q-1)) * x - x_F/(q-1)
      q 线与精馏段操作线交点 (x_q, y_q)
      Stripping line (提馏段):
        过 (x_W, x_W) 和 (x_q, y_q)

    返回:
      stages_N: 理论板数（近似）
      x_points, y_points: 阶梯线坐标
      (m_R, b_R), (m_S, b_S), (m_q, b_q): 操作线和 q 线参数
    """
    # 精馏段操作线
    m_R = R / (R + 1.0)
    b_R = x_D / (R + 1.0)

    # q 线
    m_q = q / (q - 1.0)
    b_q = -x_F / (q - 1.0)

    # 交点 (x_q, y_q): m_R x + b_R = m_q x + b_q
    x_q = (b_q - b_R) / (m_R - m_q)
    y_q = m_R * x_q + b_R

    # 提馏段操作线
    m_S = (y_q - x_W) / (x_q - x_W)
    b_S = x_W - m_S * x_W

    x_points = [x_D]
    y_points = [x_D]  # 初始在塔顶馏出

    x_curr = x_D
    y_curr = x_D
    full_stages = 0

    while x_curr > x_W and full_stages < max_stages:
        # 1) 水平到平衡线：已知 y_curr -> x_eq(y_curr)
        x_new = x_eq_from_y(y_curr)
        y_new = y_curr
        x_points.append(x_new)
        y_points.append(y_new)

        # 2) 垂直到相应操作线
        x_curr = x_new
        if x_curr >= x_q:
            # 精馏段
            y_curr = m_R * x_curr + b_R
        else:
            # 提馏段
            y_curr = m_S * x_curr + b_S

        x_points.append(x_curr)
        y_points.append(y_curr)

        full_stages += 1

        if x_curr <= x_W:
            break

    # 分板估算：仍然可以用简单比例
    if x_curr > x_W and full_stages > 0:
        x_prev_bottom = x_points[-3]
        frac = (x_prev_bottom - x_W) / (x_prev_bottom - x_curr)
        stages_N = full_stages - (1 - frac)
    else:
        stages_N = full_stages

    return (stages_N,
            np.array(x_points),
            np.array(y_points),
            (m_R, b_R),
            (m_S, b_S),
            (m_q, b_q),
            (x_q, y_q))


# ===========================
# 四、单板效率（以液相浓度表示）
# ===========================

def murphree_liquid_efficiency(x_prev: float,
                               x_curr: float,
                               x_eq_curr: float) -> float:
    """
    单板 Murphree 效率（以液相浓度表示）:
        EmL = (x_{n-1} - x_n) / (x_{n-1} - x_n^*)
    其中:
        x_prev  = x_{n-1}
        x_curr  = x_n
        x_eq_curr = x_n^* (与离开该板气相相平衡的液相浓度)
    """
    numerator = x_prev - x_curr
    denominator = x_prev - x_eq_curr
    if abs(denominator) < 1e-8:
        return np.nan
    return numerator / denominator


# ===========================
# 五、绘图函数
# ===========================

def plot_mccabe_thiele_total(x_D, x_W, x_steps, y_steps,
                             save_path: str = "total_reflux_mccabe_thiele.png"):
    """
    绘制全回流 McCabe-Thiele 图
    """
    xs = np.linspace(0, 1, 200)
    ys_eq = np.array([y_eq(x) for x in xs])

    plt.figure()
    # 平衡线
    plt.plot(xs, ys_eq, label="平衡线 y_eq(x)")
    # 对角线
    plt.plot(xs, xs, linestyle="--", label="对角线 y=x")
    # 阶梯线
    plt.plot(x_steps, y_steps, drawstyle="steps-post", label="理论板阶梯")
    plt.xlabel("x (液相乙醇摩尔分数)")
    plt.ylabel("y (气相乙醇摩尔分数)")
    plt.title("全回流条件下 McCabe-Thiele 图")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mccabe_thiele_partial(x_D, x_W, x_F,
                               x_steps, y_steps,
                               lines_params,
                               x_qy_q,
                               save_path: str = "partial_reflux_mccabe_thiele.png"):
    """
    绘制部分回流（有限 R, 有 q 线） McCabe-Thiele 图
    lines_params: (m_R, b_R), (m_S, b_S), (m_q, b_q)
    x_qy_q: (x_q, y_q)
    """
    (m_R, b_R), (m_S, b_S), (m_q, b_q) = lines_params
    x_q, y_q = x_qy_q

    xs = np.linspace(0, 1, 200)
    ys_eq = np.array([y_eq(x) for x in xs])
    y_diag = xs
    y_rect = m_R * xs + b_R
    y_strip = m_S * xs + b_S
    y_qline = m_q * xs + b_q

    plt.figure()
    plt.plot(xs, ys_eq, label="平衡线 y_eq(x)")
    plt.plot(xs, y_diag, linestyle="--", label="对角线 y=x")
    plt.plot(xs, y_rect, label="精馏段操作线")
    plt.plot(xs, y_strip, label="提馏段操作线")
    plt.plot(xs, y_qline, linestyle=":", label="q 线")
    plt.scatter([x_q], [y_q], marker="o")  # 交点
    plt.scatter([x_F], [m_q * x_F + b_q], marker="x")  # 进料点理论位置

    plt.plot(x_steps, y_steps, drawstyle="steps-post", label="理论板阶梯")

    plt.xlabel("x (液相乙醇摩尔分数)")
    plt.ylabel("y (气相乙醇摩尔分数)")
    plt.title("部分回流条件下 McCabe-Thiele 图")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===========================
# 六、一个完整示例 (使用说明书中的示例数据)
# ===========================

def example_run():
    """
    用说明书中给出的示例数据跑一遍，
    你可以在此基础上替换为自己实验测得的数据。
    """

    # -------- 1. 全回流实验（R = ∞） --------
    # 说明书示例：塔顶 nD = 1.3611, 塔釜 nD = 1.3769 （30℃）
    nd_D_total = 1.3611
    nd_W_total = 1.3769

    W_D_total = nd_to_mass_fraction(nd_D_total)
    W_W_total = nd_to_mass_fraction(nd_W_total)

    x_D_total = mass_to_mole_fraction(W_D_total)
    x_W_total = mass_to_mole_fraction(W_W_total)

    # 实际塔板数（说明书中为 12 块）
    N_actual = 12

    N_theoretical_total, x_steps_total, y_steps_total = \
        mccabe_thiele_total_reflux(x_D_total, x_W_total)

    # 全塔效率（常用定义）
    overall_eff_total = N_theoretical_total / N_actual

    # 绘图
    plot_mccabe_thiele_total(
        x_D_total, x_W_total, x_steps_total, y_steps_total,
        save_path="total_reflux_mccabe_thiele.png"
    )

    # -------- 2. 单板效率示例（第 10 块板） --------
    # 表5示例：
    #  第9板: x9 = 0.654 (摩尔分数)
    #  第10板: x10 = 0.572
    #  平衡浓度 x10* = 0.474
    x9 = 0.654
    x10 = 0.572
    x10_star = 0.474

    EmL_10 = murphree_liquid_efficiency(x_prev=x9,
                                        x_curr=x10,
                                        x_eq_curr=x10_star)

    # -------- 3. 部分回流 (例如 R = 4, 进料量 2 L/h) --------
    # 示例中的折光指数：塔顶 1.3637, 塔釜 1.3782, 进料 1.3755
    nd_D_partial = 1.3637
    nd_W_partial = 1.3782
    nd_F_partial = 1.3755
    T_F_partial = 30.4  # 进料温度 (℃)

    W_D_partial = nd_to_mass_fraction(nd_D_partial)
    W_W_partial = nd_to_mass_fraction(nd_W_partial)
    W_F_partial = nd_to_mass_fraction(nd_F_partial)

    x_D_partial = mass_to_mole_fraction(W_D_partial)
    x_W_partial = mass_to_mole_fraction(W_W_partial)
    x_F_partial = mass_to_mole_fraction(W_F_partial)

    # 比热与汽化潜热：示例给的是在 60.3℃ 和 90.27℃ 下
    Cp1 = 3.08  # 乙醇, kJ/(kg·℃)
    Cp2 = 2.89  # 正丙醇, kJ/(kg·℃)
    r1 = 821.0  # 乙醇, kJ/kg
    r2 = 684.0  # 正丙醇, kJ/kg

    q_partial, tBP_partial = calc_q(T_F_partial, x_F_partial,
                                    Cp1, Cp2, r1, r2)

    R_partial = 4.0

    (N_theoretical_partial,
     x_steps_partial,
     y_steps_partial,
     (m_R, b_R),
     (m_S, b_S),
     (m_q, b_q),
     (x_q, y_q)) = mccabe_thiele_partial_reflux(
        x_D_partial, x_W_partial, x_F_partial,
        R=R_partial, q=q_partial
    )

    overall_eff_partial = N_theoretical_partial / N_actual

    # 绘图
    plot_mccabe_thiele_partial(
        x_D_partial, x_W_partial, x_F_partial,
        x_steps_partial, y_steps_partial,
        ((m_R, b_R), (m_S, b_S), (m_q, b_q)),
        (x_q, y_q),
        save_path="partial_reflux_mccabe_thiele.png"
    )

    # -------- 4. 汇总数据 -> DataFrame & 导出 --------
    results = {
        "工况": ["全回流", "部分回流"],
        "x_D(塔顶摩尔分数)": [x_D_total, x_D_partial],
        "x_W(塔釜摩尔分数)": [x_W_total, x_W_partial],
        "x_F(进料摩尔分数)": [np.nan, x_F_partial],
        "理论板数 N_theoretical": [N_theoretical_total, N_theoretical_partial],
        "实际塔板数 N_actual": [N_actual, N_actual],
        "全塔效率 E_T": [overall_eff_total, overall_eff_partial],
        "q 值": [np.nan, q_partial],
        "进料泡点温度 t_BP(℃)": [np.nan, tBP_partial],
    }

    df_results = pd.DataFrame(results)
    print("=== 精馏实验整体结果汇总 ===")
    print(df_results)

    df_results.to_csv("distillation_results_summary.csv", index=False,
                      encoding="utf-8-sig")

    # 单板效率单独输出
    df_plate10 = pd.DataFrame({
        "板号": [10],
        "x_(n-1)": [x9],
        "x_n": [x10],
        "x_n*": [x10_star],
        "EmL": [EmL_10]
    })

    print("\n=== 第 10 块板单板效率 ===")
    print(df_plate10)

    df_plate10.to_csv("tray10_efficiency.csv", index=False,
                      encoding="utf-8-sig")


# ===========================
# 主入口
# ===========================

if __name__ == "__main__":
    example_run()
