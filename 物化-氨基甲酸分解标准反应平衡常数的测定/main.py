import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
R = 8.314  # Gas constant in J/(mol·K)

# Thermodynamic data from Table 2-4-2
delta_H_m = {
    'NH4COONH4(s)': -645.51,
    'NH3(g)': -46.141,
    'CO2(g)': -393.792
}

delta_G_m = {
    'NH4COONH4(s)': -448.386,
    'NH3(g)': -16.497,
    'CO2(g)': -394.642
}

S_m = {
    'NH4COONH4(s)': 133.565,
    'NH3(g)': 192.476,
    'CO2(g)': 213.788
}


def get_user_input():
    """Get experimental data from user input"""
    print("\n===== 输入实验数据 =====")
    print("请输入实验室温度(°C):")
    room_temp = float(input())

    print("请输入大气压(kPa):")
    atm_pressure = float(input())

    print("\n请输入实验中测量的各温度点数据")
    print("请输入要记录的温度点数量:")
    num_points = int(input())

    temperatures_c = []
    p_eq_kPa = []

    print("\n对于每个温度点，请输入以下数据：")
    for i in range(num_points):
        print(f"\n温度点 #{i + 1}:")
        print("温度(°C):")
        temp = float(input())
        temperatures_c.append(temp)

        print("测量的平衡压力(kPa):")
        pressure = float(input())
        p_eq_kPa.append(pressure)

    return room_temp, atm_pressure, np.array(temperatures_c), np.array(p_eq_kPa)


def use_reference_data():
    """Use reference data from the PDF tables"""
    # Reference data from Table 2-4-3 (Temperature vs. Decomposition Pressure)
    room_temp = 25  # Assumed room temperature
    atm_pressure = 101.325  # Standard atmospheric pressure
    temperatures_c = np.array([25, 30, 35, 40, 45, 50])  # Temperature in °C
    p_eq_kPa = np.array([11.73, 17.07, 23.80, 32.93, 45.33, 62.93])  # Equilibrium pressure in kPa

    return room_temp, atm_pressure, temperatures_c, p_eq_kPa


def calculate_equilibrium_constant(p, T):
    """
    Calculate equilibrium constant K° for NH4COONH4(s) ⇌ 2NH3(g) + CO2(g)
    using equation (2-4-3):
    K° = (4/27) * (p/p°)³

    where p is the equilibrium pressure and p° is the standard pressure (100 kPa)
    """
    p_standard = 100  # Standard pressure in kPa

    # Detailed calculation
    p_ratio = p / p_standard
    p_ratio_cubed = p_ratio ** 3
    K = (4 / 27) * p_ratio_cubed
    ln_K = np.log(K)

    return K, ln_K, {
        'p_ratio': p_ratio,
        'p_ratio_cubed': p_ratio_cubed,
        '4/27': 4 / 27
    }


def convert_to_kelvin(temp_c):
    """Convert temperature from Celsius to Kelvin"""
    return temp_c + 273.15


def calculate_thermodynamic_parameters(delta_H_m, delta_G_m, S_m, T):
    """
    Calculate thermodynamic parameters for the reaction:
    NH4COONH4(s) ⇌ 2NH3(g) + CO2(g)
    """
    # Detailed calculation of reaction enthalpy change
    print("\n反应焓变计算 (ΔH°reaction):")
    print(f"ΔH°reaction = 2 × ΔH°f[NH3(g)] + ΔH°f[CO2(g)] - ΔH°f[NH4COONH4(s)]")
    print(f"ΔH°reaction = 2 × ({delta_H_m['NH3(g)']}) + ({delta_H_m['CO2(g)']}) - ({delta_H_m['NH4COONH4(s)']})")

    delta_H_reaction = 2 * delta_H_m['NH3(g)'] + delta_H_m['CO2(g)'] - delta_H_m['NH4COONH4(s)']

    print(
        f"ΔH°reaction = {2 * delta_H_m['NH3(g)']:.3f} + {delta_H_m['CO2(g)']:.3f} - ({delta_H_m['NH4COONH4(s)']:.3f})")
    print(f"ΔH°reaction = {delta_H_reaction:.3f} J/mol = {delta_H_reaction / 1000:.3f} kJ/mol")

    # Detailed calculation of reaction Gibbs free energy change
    print("\n反应吉布斯自由能变计算 (ΔG°reaction):")
    print(f"ΔG°reaction = 2 × ΔG°f[NH3(g)] + ΔG°f[CO2(g)] - ΔG°f[NH4COONH4(s)]")
    print(f"ΔG°reaction = 2 × ({delta_G_m['NH3(g)']}) + ({delta_G_m['CO2(g)']}) - ({delta_G_m['NH4COONH4(s)']})")

    delta_G_reaction = 2 * delta_G_m['NH3(g)'] + delta_G_m['CO2(g)'] - delta_G_m['NH4COONH4(s)']

    print(
        f"ΔG°reaction = {2 * delta_G_m['NH3(g)']:.3f} + {delta_G_m['CO2(g)']:.3f} - ({delta_G_m['NH4COONH4(s)']:.3f})")
    print(f"ΔG°reaction = {delta_G_reaction:.3f} J/mol = {delta_G_reaction / 1000:.3f} kJ/mol")

    # Calculate theoretical equilibrium constant from ΔG°
    K_theoretical = np.exp(-delta_G_reaction / (R * T))
    print(f"\n从ΔG°计算理论平衡常数:")
    print(f"K° = exp(-ΔG°/(R×T)) = exp(-{delta_G_reaction:.3f}/({R:.3f}×{T:.2f}))")
    print(f"K° = exp(-{delta_G_reaction / (R * T):.4f}) = {K_theoretical:.6e}")

    # Detailed calculation of reaction entropy change
    print("\n反应熵变计算 (ΔS°reaction):")
    print(f"ΔS°reaction = 2 × S°[NH3(g)] + S°[CO2(g)] - S°[NH4COONH4(s)]")
    print(f"ΔS°reaction = 2 × ({S_m['NH3(g)']}) + ({S_m['CO2(g)']}) - ({S_m['NH4COONH4(s)']})")

    delta_S_reaction_from_S = 2 * S_m['NH3(g)'] + S_m['CO2(g)'] - S_m['NH4COONH4(s)']

    print(f"ΔS°reaction = {2 * S_m['NH3(g)']:.3f} + {S_m['CO2(g)']:.3f} - {S_m['NH4COONH4(s)']:.3f}")
    print(f"ΔS°reaction (从标准熵) = {delta_S_reaction_from_S:.3f} J/(K·mol)")

    # Alternative calculation of ΔS° from ΔH° and ΔG°
    delta_S_reaction = (delta_H_reaction - delta_G_reaction) / T
    print("\n从ΔH°和ΔG°计算反应熵变:")
    print(f"ΔS°reaction = (ΔH°reaction - ΔG°reaction) / T")
    print(f"ΔS°reaction = ({delta_H_reaction:.3f} - {delta_G_reaction:.3f}) / {T:.2f}")
    print(f"ΔS°reaction (从ΔH°和ΔG°) = {delta_S_reaction:.3f} J/(K·mol)")

    return delta_H_reaction, delta_G_reaction, delta_S_reaction, delta_S_reaction_from_S, K_theoretical


def linear_fit(x, a, b):
    """Linear function for curve fitting: y = ax + b"""
    return a * x + b


def save_results_to_file(results, filename="ammonium_carbamate_results.txt"):
    """Save calculation results to a text file"""
    with open(filename, "w") as f:
        f.write(results)
    print(f"\n结果已保存到文件: {filename}")


def main():
    print("======= 氨基甲酸铵分解反应平衡常数的测定 =======")
    print("反应方程式: NH4COONH4(s) ⇌ 2NH3(g) + CO2(g)")
    print("=" * 50)

    # Ask user for data input method
    print("\n请选择数据输入方式:")
    print("1. 手动输入实验数据")
    print("2. 使用参考数据(来自PDF表格)")

    choice = input("请输入选择 (1 或 2): ")

    if choice == "1":
        room_temp, atm_pressure, temperatures_c, p_eq_kPa = get_user_input()
    else:
        room_temp, atm_pressure, temperatures_c, p_eq_kPa = use_reference_data()

    # Results storage for file output
    results_text = ["======= 氨基甲酸铵分解反应平衡常数的测定 =======",
                    f"实验室温度: {room_temp}°C",
                    f"大气压: {atm_pressure} kPa",
                    "=" * 50]

    # Convert temperatures to Kelvin
    temperatures_k = np.array([convert_to_kelvin(t) for t in temperatures_c])
    temp_conversion = "\n温度转换 (摄氏度 → 开尔文):"
    print(temp_conversion)
    results_text.append(temp_conversion)

    for i in range(len(temperatures_c)):
        temp_line = f"{temperatures_c[i]}°C = {temperatures_k[i]:.2f} K"
        print(temp_line)
        results_text.append(temp_line)

    # Calculate equilibrium constants and their natural logarithms with detailed steps
    detailed_calc = "\n平衡常数计算详细过程:"
    print(detailed_calc)
    results_text.append(detailed_calc)

    K_values = []
    ln_K_values = []
    calc_details = []

    for i in range(len(temperatures_c)):
        t_c = temperatures_c[i]
        T_K = temperatures_k[i]
        p = p_eq_kPa[i]

        temp_calc = f"\n温度 {t_c}°C ({T_K:.2f} K) 下的计算:"
        print(temp_calc)
        results_text.append(temp_calc)

        p_line = f"平衡压力 p = {p} kPa"
        print(p_line)
        results_text.append(p_line)

        K, ln_K, details = calculate_equilibrium_constant(p, T_K)

        p_ratio_line = f"p/p° = {p}/100 = {details['p_ratio']:.6f}"
        print(p_ratio_line)
        results_text.append(p_ratio_line)

        p_ratio_cubed_line = f"(p/p°)³ = ({details['p_ratio']:.6f})³ = {details['p_ratio_cubed']:.6e}"
        print(p_ratio_cubed_line)
        results_text.append(p_ratio_cubed_line)

        k_calc_line = f"K° = (4/27) × (p/p°)³ = {details['4/27']:.6f} × {details['p_ratio_cubed']:.6e} = {K:.6e}"
        print(k_calc_line)
        results_text.append(k_calc_line)

        ln_k_line = f"ln(K°) = ln({K:.6e}) = {ln_K:.6f}"
        print(ln_k_line)
        results_text.append(ln_k_line)

        K_values.append(K)
        ln_K_values.append(ln_K)
        calc_details.append(details)

    # Create data table
    table_header = "\n表 2-4-1 分解压及平衡常数实验数据"
    print(table_header)
    results_text.append(table_header)

    table_separator = "-" * 100
    print(table_separator)
    results_text.append(table_separator)

    table_columns = f"{'温度 t/°C':<10} {'T/K':<10} {'1/T×10³ (1/K)':<15} {'p/kPa':<10} {'K°':<15} {'lnK°':<10}"
    print(table_columns)
    results_text.append(table_columns)

    print(table_separator)
    results_text.append(table_separator)

    # Fill in the table with calculated values
    inv_T_values = []
    for i in range(len(temperatures_c)):
        t = temperatures_c[i]
        T = temperatures_k[i]
        inv_T = (1 / T) * 1000  # 1/T × 10³
        inv_T_values.append(inv_T)
        p = p_eq_kPa[i]
        K = K_values[i]
        ln_K = ln_K_values[i]

        table_row = f"{t:<10.1f} {T:<10.1f} {inv_T:<15.4f} {p:<10.2f} {K:<15.6e} {ln_K:<10.4f}"
        print(table_row)
        results_text.append(table_row)

    # Fit ln(K) vs 1/T to determine ΔH°
    # According to equation (2-4-4): ln(K°) = -ΔH°m/RT + C
    fit_header = "\nln(K°)对1/T的线性拟合:"
    print(fit_header)
    results_text.append(fit_header)

    fit_equation = "根据公式 (2-4-4): ln(K°) = -ΔH°m/(R×T) + C"
    print(fit_equation)
    results_text.append(fit_equation)

    r_constant = f"其中 R = {R} J/(mol·K)"
    print(r_constant)
    results_text.append(r_constant)

    # Perform linear regression
    params, covariance = curve_fit(linear_fit, inv_T_values, ln_K_values)
    a, b = params

    fit_result = f"\n线性拟合结果: ln(K°) = {a:.4f}×(1/T×10³) + {b:.4f}"
    print(fit_result)
    results_text.append(fit_result)

    slope_line = f"斜率 a = {a:.4f}"
    print(slope_line)
    results_text.append(slope_line)

    intercept_line = f"截距 b = {b:.4f}"
    print(intercept_line)
    results_text.append(intercept_line)

    # Calculate ΔH°m from the slope
    delta_H_m_experimental = -a * R / 1000  # Convert from slope (which uses 1/T×10³)

    enthalpy_calc_header = "\n从斜率计算反应焓变:"
    print(enthalpy_calc_header)
    results_text.append(enthalpy_calc_header)

    enthalpy_formula = f"ΔH°m = -a × R × 1000"
    print(enthalpy_formula)
    results_text.append(enthalpy_formula)

    enthalpy_calc = f"ΔH°m = -{a:.4f} × {R:.3f} × 1000"
    print(enthalpy_calc)
    results_text.append(enthalpy_calc)

    enthalpy_result = f"ΔH°m = {delta_H_m_experimental:.2f} kJ/mol"
    print(enthalpy_result)
    results_text.append(enthalpy_result)

    # Calculate thermodynamic parameters at 25°C (298.15 K)
    T_ref = 298.15  # Reference temperature (25°C) in Kelvin
    thermo_header = f"\n在标准温度 (25°C = {T_ref} K) 下计算热力学参数:"
    print(thermo_header)
    results_text.append(thermo_header)

    # Capture thermodynamic calculations
    original_stdout = sys.stdout
    from io import StringIO
    thermo_output = StringIO()
    sys.stdout = thermo_output

    delta_H_reaction, delta_G_reaction, delta_S_reaction, delta_S_from_S, K_theoretical = calculate_thermodynamic_parameters(
        delta_H_m, delta_G_m, S_m, T_ref)

    # Restore stdout and get the captured output
    sys.stdout = original_stdout
    thermo_results = thermo_output.getvalue()
    print(thermo_results)
    results_text.append(thermo_results)

    # Compare experimental and theoretical values
    comparison_header = "\n实验值与理论值比较:"
    print(comparison_header)
    results_text.append(comparison_header)

    # Get experimental K at 25°C if available, otherwise use the closest temperature
    try:
        idx_25C = np.where(temperatures_c == 25)[0][0]
        K_exp_25C = K_values[idx_25C]
        comparison_available = True
    except:
        # Find the closest temperature to 25°C
        closest_idx = np.abs(temperatures_c - 25).argmin()
        closest_temp = temperatures_c[closest_idx]
        K_exp_closest = K_values[closest_idx]
        comparison_available = False

    if comparison_available:
        exp_k_line = f"25°C时实验测定的平衡常数: K°(exp) = {K_exp_25C:.6e}"
        print(exp_k_line)
        results_text.append(exp_k_line)

        theor_k_line = f"25°C时理论计算的平衡常数: K°(theor) = {K_theoretical:.6e}"
        print(theor_k_line)
        results_text.append(theor_k_line)

        error_line = f"相对误差: {(K_exp_25C - K_theoretical) / K_theoretical * 100:.2f}%"
        print(error_line)
        results_text.append(error_line)
    else:
        closest_k_line = f"{closest_temp}°C时实验测定的平衡常数: K°(exp) = {K_exp_closest:.6e}"
        print(closest_k_line)
        results_text.append(closest_k_line)

        theor_k_line = f"25°C时理论计算的平衡常数: K°(theor) = {K_theoretical:.6e}"
        print(theor_k_line)
        results_text.append(theor_k_line)

        note_line = "注: 由于没有25°C的实验数据点，无法直接比较"
        print(note_line)
        results_text.append(note_line)

    # Print summary of results
    summary_header = "\n结果总结:"
    print(summary_header)
    results_text.append(summary_header)

    exp_enthalpy = f"实验测定的标准反应焓变 ΔH°m = {delta_H_m_experimental:.2f} kJ/mol"
    print(exp_enthalpy)
    results_text.append(exp_enthalpy)

    theor_enthalpy = f"理论计算的标准反应焓变 ΔH°reaction = {delta_H_reaction / 1000:.2f} kJ/mol"
    print(theor_enthalpy)
    results_text.append(theor_enthalpy)

    gibbs_energy = f"标准吉布斯自由能变 ΔG°reaction(25°C) = {delta_G_reaction / 1000:.2f} kJ/mol"
    print(gibbs_energy)
    results_text.append(gibbs_energy)

    entropy_change = f"标准熵变 ΔS°reaction(25°C) = {delta_S_reaction:.2f} J/(K·mol)"
    print(entropy_change)
    results_text.append(entropy_change)

    # Plot ln(K) vs 1/T with detailed annotations
    plt.figure(figsize=(10, 7))
    plt.scatter(inv_T_values, ln_K_values, color='blue', marker='o', label='实验数据')

    # Plot the fitted line
    x_line = np.linspace(min(inv_T_values), max(inv_T_values), 100)
    y_line = linear_fit(x_line, a, b)
    plt.plot(x_line, y_line, 'r-', label=f'线性拟合: ln(K) = {a:.2f}×(1/T×10³) + {b:.2f}')

    # Add formula annotation
    plt.annotate(
        f"ln(K°) = -ΔH°m/(R×T) + C\n斜率 = -{a:.4f}\nΔH°m = {delta_H_m_experimental:.2f} kJ/mol",
        xy=(3.2, min(ln_K_values) + (max(ln_K_values) - min(ln_K_values)) * 0.7),  # Position of text
        xycoords='data',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )

    plt.xlabel('1/T × 10³ (K⁻¹)')
    plt.ylabel('ln(K°)')
    plt.title('氨基甲酸铵分解反应平衡常数的测定 (van\'t Hoff图)')
    plt.grid(True)
    plt.legend()

    # Plot equilibrium pressure vs temperature
    plt.figure(figsize=(10, 7))
    plt.scatter(temperatures_c, p_eq_kPa, color='green', marker='s', label='实验数据')

    # Fit and plot an exponential curve
    def exp_func(x, a, b):
        return a * np.exp(b * x)

    # Convert to arrays for curve fitting
    t_array = np.array(temperatures_c)
    p_array = np.array(p_eq_kPa)

    params_p, _ = curve_fit(exp_func, t_array, p_array)
    x_smooth = np.linspace(min(t_array), max(t_array), 100)
    y_smooth = exp_func(x_smooth, *params_p)

    plt.plot(x_smooth, y_smooth, 'r-', label=f'拟合曲线: p = {params_p[0]:.2f}·e^({params_p[1]:.4f}·T)')

    # Annotate with Clausius-Clapeyron relation
    plt.annotate(
        "根据Clausius-Clapeyron方程:\nln(p) = -ΔH°/(R·T) + C\n温度升高，平衡压力增大",
        xy=(min(temperatures_c) + (max(temperatures_c) - min(temperatures_c)) * 0.5,
            min(p_eq_kPa) + (max(p_eq_kPa) - min(p_eq_kPa)) * 0.3),
        xycoords='data',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )

    # Plot ln(p) vs T for linear analysis
    plt.figure(figsize=(10, 7))
    ln_p_array = np.log(p_array)
    plt.scatter(temperatures_c, ln_p_array, color='blue', marker='o', label='实验数据 (ln(p))')

    # Perform linear regression on ln(p) vs T
    params_ln_p, _ = curve_fit(linear_fit, t_array, ln_p_array)
    a_ln_p, b_ln_p = params_ln_p

    # Plot the fitted line
    y_smooth_ln_p = linear_fit(x_smooth, a_ln_p, b_ln_p)
    plt.plot(x_smooth, y_smooth_ln_p, 'r-', label=f'线性拟合: ln(p) = {a_ln_p:.4f}·T + {b_ln_p:.4f}')

    # Annotate with the linear fit equation
    plt.annotate(
        f"ln(p) = {a_ln_p:.4f}·T + {b_ln_p:.4f}",
        xy=(min(temperatures_c) + (max(temperatures_c) - min(temperatures_c)) * 0.5,
            min(ln_p_array) + (max(ln_p_array) - min(ln_p_array)) * 0.7),
        xycoords='data',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )

    plt.xlabel('温度 (°C)')
    plt.ylabel('ln(平衡压力) (kPa)')
    plt.title('ln(平衡压力) 随温度的变化 (线性分析)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Ask if user wants to save results to a file
    save_option = input("\n是否保存结果到文件? (y/n): ")
    if save_option.lower() == 'y':
        filename = input("请输入文件名 (默认: ammonium_carbamate_results.txt): ") or "ammonium_carbamate_results.txt"
        save_results_to_file("\n".join(results_text), filename)



if __name__ == "__main__":
    import sys

    main()