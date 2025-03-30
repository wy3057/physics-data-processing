import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import linregress
import pandas as pd
# 设置 Matplotlib 使用支持 CJK 的字体
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持 CJK 的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def calculate_rate_constant(t_values, l_values, l0, temp, verbose=True):
    """
    计算反应速率常数k，并输出详细的计算过程
    t_values: 时间列表
    l_values: 电导率列表
    l0: 起始电导率
    temp: 温度
    verbose: 是否输出详细计算过程
    """
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"温度 {temp}°C 下的速率常数计算过程：")
        print(f"{'=' * 50}")
        print("根据公式(2-5-9): 1/(L0-Lt) = m·(1/t) + B")
        print("其中 m = A/(C0²·k), B = A/C0")
        print("通过线性回归求出 m 和 B，再计算 k = B/m = C0·k")
        print(f"\n步骤1: 计算 L0-Lt 和 1/(L0-Lt) 值")
        print(f"起始电导率 L0 = {l0}")

    # 计算 L0-Lt 和 1/(L0-Lt)
    l0_minus_lt = [l0 - l for l in l_values]
    inv_l0_minus_lt = [1 / x if x != 0 else float('inf') for x in l0_minus_lt]

    # 计算 1/t
    inv_t = [1 / t if t != 0 else float('inf') for t in t_values]

    if verbose:
        print("\n时间 t(min)\t电导率 Lt\tL0-Lt\t\t1/(L0-Lt)\t1/t")
        for i in range(len(t_values)):
            print(
                f"{t_values[i]:.2f}\t\t{l_values[i]:.6f}\t{l0_minus_lt[i]:.6f}\t{inv_l0_minus_lt[i]:.6f}\t{inv_t[i]:.6f}")

    # 去掉无穷大的点
    valid_indices = [i for i, val in enumerate(inv_l0_minus_lt) if val != float('inf') and inv_t[i] != float('inf')]
    valid_inv_l0_minus_lt = [inv_l0_minus_lt[i] for i in valid_indices]
    valid_inv_t = [inv_t[i] for i in valid_indices]
    valid_t = [t_values[i] for i in valid_indices]

    if verbose:
        print("\n步骤2: 对 1/(L0-Lt) vs 1/t 进行线性回归")
        print("有效数据点:")
        print("时间 t(min)\t1/t\t\t1/(L0-Lt)")
        for i in range(len(valid_t)):
            print(f"{valid_t[i]:.2f}\t\t{valid_inv_t[i]:.6f}\t{valid_inv_l0_minus_lt[i]:.6f}")

    # 线性回归
    slope, intercept, r_value, p_value, std_err = linregress(valid_inv_t, valid_inv_l0_minus_lt)

    # 斜率m = A/(C0^2*k)，截距B = A/C0
    m = slope
    B = intercept

    # 计算反应速率常数k
    k = B / m  # A/C0 除以 A/(C0^2*k) = C0*k

    if verbose:
        print(f"\n线性回归结果:")
        print(f"斜率 m = {m:.6f}")
        print(f"截距 B = {B:.6f}")
        print(f"相关系数 R² = {r_value ** 2:.6f}")
        print("\n步骤3: 计算反应速率常数 k")
        print(f"k = B/m = {B:.6f}/{m:.6f} = {k:.6f} L·mol⁻¹·min⁻¹")

    return k, m, B, r_value


def calculate_activation_energy(k1, k2, T1, T2, verbose=True):
    """
    使用Arrhenius方程计算活化能，并输出详细的计算过程
    k1, k2: 不同温度下的反应速率常数
    T1, T2: 对应的温度(°C)
    verbose: 是否输出详细计算过程
    """
    if verbose:
        print(f"\n{'=' * 50}")
        print("活化能计算过程：")
        print(f"{'=' * 50}")
        print("根据Arrhenius方程(2-5-11):")
        print("ln[k(T₂)/k(T₁)] = E_a(T₂-T₁)/(R·T₁·T₂)")
        print("其中k(T₁)和k(T₂)是温度T₁和T₂下的反应速率常数")
        print("R是理想气体常数(8.314 J/(mol·K))，E_a是活化能")

    # 转换为开尔文温度
    T1_K = T1 + 273.15
    T2_K = T2 + 273.15

    # 理想气体常数 (R = 8.314 J/(mol·K))
    R = 8.314

    # 计算活化能 (单位: J/mol)
    ln_k_ratio = np.log(k2 / k1)
    Ea = R * T1_K * T2_K * ln_k_ratio / (T2_K - T1_K)

    # 转换为kcal/mol (1 kcal = 4184 J)
    Ea_kcal = Ea / 4184

    if verbose:
        print(f"\n步骤1: 温度转换为开尔文")
        print(f"T₁ = {T1}°C = {T1_K} K")
        print(f"T₂ = {T2}°C = {T2_K} K")
        print(f"\n步骤2: 计算ln[k(T₂)/k(T₁)]")
        print(f"k(T₁) = {k1:.6f} L·mol⁻¹·min⁻¹")
        print(f"k(T₂) = {k2:.6f} L·mol⁻¹·min⁻¹")
        print(f"k(T₂)/k(T₁) = {k2 / k1:.6f}")
        print(f"ln[k(T₂)/k(T₁)] = {ln_k_ratio:.6f}")
        print(f"\n步骤3: 计算活化能")
        print(f"E_a = R·T₁·T₂·ln[k(T₂)/k(T₁)]/(T₂-T₁)")
        print(f"    = {R}·{T1_K}·{T2_K}·{ln_k_ratio:.6f}/({T2_K}-{T1_K})")
        print(f"    = {Ea:.2f} J/mol")
        print(f"    = {Ea_kcal:.2f} kcal/mol (1 kcal = 4184 J)")

    return Ea_kcal


def main():
    print("=== 乙酸乙酯皂化反应速率常数的测定 ===")
    print("\n本实验根据乙酸乙酯皂化反应的动力学原理，通过测量不同时间点的电导率来计算反应速率常数和活化能。")
    print("实验原理：乙酸乙酯与NaOH的皂化反应 CH₃COOC₂H₅ + NaOH → CH₃COONa + C₂H₅OH")
    print("反应速率方程：dx/dt = k(C₀-x)²")
    print("线性关系式：1/(L₀-Lₜ) = m·(1/t) + B，其中m = A/(C₀²·k)，B = A/C₀\n")

    # 输入实验基本参数
    T1 = float(input("请输入第一个温度T1 (°C): "))
    T2 = float(input("请输入第二个温度T2 (°C): "))
    C0 = float(input("请输入反应物起始浓度C0 (mol/L): "))

    print("\n=== 数据输入 ===")
    print("根据公式(2-5-4): x = A(L₀ - Lₜ)")
    print("其中，L₀是起始电导率，Lₜ是t时刻的电导率")

    # 输入第一个温度的L0
    L0_T1 = float(input(f"\n请输入温度T1 ({T1}°C)下的起始电导率L0: "))

    # 创建表格用于存储数据
    t_values = [2, 4, 6, 8, 10, 13, 16, 19, 21, 25, 30]

    # 初始化数据结构
    data = {
        'time_min': t_values,
        '1/t': [1 / t for t in t_values],
        'Lt_T1': [],
        'L0-Lt_T1': [],
        '1/(L0-Lt)_T1': [],
        'Lt_T2': [],
        'L0-Lt_T2': [],
        '1/(L0-Lt)_T2': []
    }

    # 输入第一个温度下的电导率数据
    print(f"\n请输入温度T1 ({T1}°C)下不同时间的电导率Lt值:")
    for t in t_values:
        lt = float(input(f"t = {t} min时的电导率Lt: "))
        data['Lt_T1'].append(lt)
        diff = L0_T1 - lt
        data['L0-Lt_T1'].append(diff)
        data['1/(L0-Lt)_T1'].append(1 / diff if diff != 0 else float('inf'))

    # 输入第二个温度的L0
    L0_T2 = float(input(f"\n请输入温度T2 ({T2}°C)下的起始电导率L0: "))

    # 输入第二个温度下的电导率数据
    print(f"\n请输入温度T2 ({T2}°C)下不同时间的电导率Lt值:")
    for t in t_values:
        lt = float(input(f"t = {t} min时的电导率Lt: "))
        data['Lt_T2'].append(lt)
        diff = L0_T2 - lt
        data['L0-Lt_T2'].append(diff)
        data['1/(L0-Lt)_T2'].append(1 / diff if diff != 0 else float('inf'))

    # 计算速率常数，并输出详细计算过程
    k1, m1, B1, r1 = calculate_rate_constant(t_values, data['Lt_T1'], L0_T1, T1)
    k2, m2, B2, r2 = calculate_rate_constant(t_values, data['Lt_T2'], L0_T2, T2)

    # 计算35°C时的k值
    # 使用Arrhenius方程进行插值
    T_target = 35
    T_values = [T1, T2]
    k_values = [k1, k2]

    print(f"\n{'=' * 50}")
    print(f"计算35°C时的反应速率常数:")
    print(f"{'=' * 50}")
    print("使用Arrhenius方程: ln(k) = ln(A) - E_a/(R·T)")
    print("对ln(k)和1/T做线性回归，得到ln(k) = a·(1/T) + b")

    # 计算ln(k)与1/T的线性关系
    inv_T = [1 / (t + 273.15) for t in T_values]
    ln_k = [np.log(k) for k in k_values]

    # 线性回归
    slope_arrh, intercept_arrh, r_arrh, p_arrh, std_err_arrh = linregress(inv_T, ln_k)

    print(f"\n步骤1: 计算ln(k)和1/T的值")
    print(f"温度T₁ = {T1}°C = {T1 + 273.15} K, 1/T₁ = {1 / (T1 + 273.15):.8f} K⁻¹")
    print(f"速率常数k₁ = {k1:.6f} L·mol⁻¹·min⁻¹, ln(k₁) = {ln_k[0]:.6f}")
    print(f"温度T₂ = {T2}°C = {T2 + 273.15} K, 1/T₂ = {1 / (T2 + 273.15):.8f} K⁻¹")
    print(f"速率常数k₂ = {k2:.6f} L·mol⁻¹·min⁻¹, ln(k₂) = {ln_k[1]:.6f}")

    print(f"\n步骤2: 线性回归得到ln(k) = a·(1/T) + b")
    print(f"a = {slope_arrh:.6f}, b = {intercept_arrh:.6f}, R² = {r_arrh ** 2:.6f}")

    # 计算35°C时的k值
    inv_T_35 = 1 / (35 + 273.15)
    ln_k_35 = slope_arrh * inv_T_35 + intercept_arrh
    k_35 = np.exp(ln_k_35)

    print(f"\n步骤3: 计算35°C时的k值")
    print(f"35°C = {35 + 273.15} K, 1/T = {inv_T_35:.8f} K⁻¹")
    print(f"ln(k₃₅) = {slope_arrh:.6f}·{inv_T_35:.8f} + {intercept_arrh:.6f} = {ln_k_35:.6f}")
    print(f"k₃₅ = e^{ln_k_35:.6f} = {k_35:.6f} L·mol⁻¹·min⁻¹")

    # 计算活化能
    Ea = calculate_activation_energy(k1, k2, T1, T2)

    # 创建完整的数据表
    df = pd.DataFrame({
        't/min': data['time_min'],
        '1/t': data['1/t'],
        f'Lt (T1={T1}°C)': data['Lt_T1'],
        f'L0-Lt (T1={T1}°C)': data['L0-Lt_T1'],
        f'1/(L0-Lt) (T1={T1}°C)': data['1/(L0-Lt)_T1'],
        f'Lt (T2={T2}°C)': data['Lt_T2'],
        f'L0-Lt (T2={T2}°C)': data['L0-Lt_T2'],
        f'1/(L0-Lt) (T2={T2}°C)': data['1/(L0-Lt)_T2']
    })

    # 绘图：1/(L0-Lt) vs 1/t
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.scatter(data['1/t'], data['1/(L0-Lt)_T1'], label=f'T1 = {T1}°C 数据点')

    # 绘制拟合直线
    x_fit = np.linspace(min(data['1/t']), max(data['1/t']), 100)
    y_fit = m1 * x_fit + B1
    plt.plot(x_fit, y_fit, 'r-', label=f'拟合直线: y = {m1:.4f}x + {B1:.4f}')

    plt.xlabel('1/t (min⁻¹)')
    plt.ylabel('1/(L₀-Lₜ)')
    plt.title(f'T1 = {T1}°C 时的 1/(L₀-Lₜ) vs 1/t 图')
    plt.grid(True)
    plt.legend()

    plt.subplot(122)
    plt.scatter(data['1/t'], data['1/(L0-Lt)_T2'], label=f'T2 = {T2}°C 数据点')

    # 绘制拟合直线
    x_fit = np.linspace(min(data['1/t']), max(data['1/t']), 100)
    y_fit = m2 * x_fit + B2
    plt.plot(x_fit, y_fit, 'r-', label=f'拟合直线: y = {m2:.4f}x + {B2:.4f}')

    plt.xlabel('1/t (min⁻¹)')
    plt.ylabel('1/(L₀-Lₜ)')
    plt.title(f'T2 = {T2}°C 时的 1/(L₀-Lₜ) vs 1/t 图')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('rate_constant_plots.png', dpi=300)

    # 绘制Arrhenius图
    plt.figure(figsize=(8, 6))

    # 计算更多温度点进行插值
    temperatures = np.linspace(min(T_values) - 5, max(T_values) + 5, 100)
    inv_temperatures = [1 / (t + 273.15) for t in temperatures]
    ln_k_values = [slope_arrh * (1 / (t + 273.15)) + intercept_arrh for t in temperatures]

    plt.scatter(inv_T, ln_k, color='red', s=50, label='实验数据点')
    plt.plot(inv_temperatures, ln_k_values, 'b-', label='Arrhenius拟合线')

    # 标记35°C点
    plt.scatter([1 / (35 + 273.15)], [np.log(k_35)], color='green', s=80, label='35°C预测点')

    plt.xlabel('1/T (K⁻¹)')
    plt.ylabel('ln(k)')
    plt.title('Arrhenius图: ln(k) vs 1/T')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('arrhenius_plot.png', dpi=300)

    # 打印结果
    print("\n=== 数据表 ===")
    print(df.to_string(index=False))

    print("\n{'='*50}")
    print("实验结果总结：")
    print(f"{'=' * 50}")
    print(f"1. 温度T1 ({T1}°C)下的反应速率常数 k1 = {k1:.6f} L·mol⁻¹·min⁻¹")
    print(f"2. 温度T2 ({T2}°C)下的反应速率常数 k2 = {k2:.6f} L·mol⁻¹·min⁻¹")
    print(f"3. 预测35°C下的反应速率常数 k_35 = {k_35:.6f} L·mol⁻¹·min⁻¹")
    print(f"4. 活化能 Ea = {Ea:.2f} kcal·mol⁻¹")

    # 保存完整的计算过程到文件
    with open('detailed_results.txt', 'w', encoding='utf-8') as f:
        f.write("==============================================================\n")
        f.write("            乙酸乙酯皂化反应速率常数的测定 - 详细计算过程\n")
        f.write("==============================================================\n\n")

        f.write("一、实验原理\n")
        f.write("============\n")
        f.write("乙酸乙酯皂化反应是一个典型的二级反应，其反应速率与乙酸乙酯和碱的浓度相乘成正比。\n")
        f.write("化学反应方程式：CH₃COOC₂H₅ + NaOH → CH₃COONa + C₂H₅OH\n\n")

        f.write("设起始的乙酸乙酯浓度和碱的起始浓度相同，并以C₀表示。当反应进行到某一时间t时，两者浓度减少均为x，即：\n")
        f.write("t = 0时的浓度\tC₀\t\tC₀\t\t0\t\t0\n")
        f.write("t = t时的浓度\tC₀-x\t\tC₀-x\t\tx\t\tx\n\n")

        f.write("则此二级反应速率方程为：\n")
        f.write("dx/dt = k(C₀-x)²    (2-5-1)\n\n")

        f.write("积分得：\n")
        f.write("∫₀ᵗ dx/(C₀-x)² = ∫₀ᵗ kdt    (2-5-2)\n\n")

        f.write("得到动力学方程：\n")
        f.write("x/(C₀(C₀-x)) = kt    (2-5-3)\n\n")

        f.write("通过电导率测量，可以得到：\n")
        f.write("x = A(L₀-Lₜ)    (2-5-4)\n")
        f.write("其中，A为比例常数，L₀为起始溶液的电导率，Lₜ为t时溶液的电导率。\n\n")

        f.write("将式(2-5-4)代入式(2-5-3)得：\n")
        f.write("kt = A(L₀-Lₜ)/(C₀[C₀-A(L₀-Lₜ)])    (2-5-5)\n\n")

        f.write("整理得：\n")
        f.write("1/(L₀-Lₜ) = m·(1/t) + B    (2-5-9)\n")
        f.write("其中m = A/(C₀²·k)，B = A/C₀\n\n")

        f.write("通过对1/(L₀-Lₜ)对1/t作图，得到一条直线，由直线的斜率m和截距B，可求出反应速率常数k = B/m。\n\n")

    return df, k1, k2, k_35, Ea


if __name__ == "__main__":
    main()
