import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tabulate import tabulate
import math
import pandas as pd
from datetime import datetime


def main():
    print("\n" + "=" * 80)
    print(" " * 30 + "蔗糖水解反应速率常数的测定")
    print("=" * 80)

    # 实验环境参数收集
    print("\n【实验基本信息】")
    experimenter = input("实验者姓名: ")
    experiment_date = input("实验日期 (YYYY-MM-DD): ")
    room_temp = input("室温 (°C): ")
    atm_pressure = input("大气压 (mmHg): ")
    experiment_temp = input("实验温度 (°C): ")
    polarimeter_tube_length = input("旋光管长度 l (dm): ")

    # 初始设置
    print("\n【实验初始参数】")
    alpha_0 = float(input("α₀ (反应开始时的旋光度): "))
    alpha_infinity = float(input("α∞ (完全水解后的旋光度): "))
    polarimeter_zero = float(input("旋光仪零点校正值: "))

    # 确定催化剂浓度选择
    print("\n【催化剂选择】")
    print("请选择催化剂浓度:")
    print("1. c(HCl) = 2 mol·L⁻¹")
    print("2. c(HCl) = 1 mol·L⁻¹")
    print("3. 同时输入两种浓度的数据")
    catalyst_choice = input("请输入选项 (1/2/3): ")

    # 创建表格数据结构
    if catalyst_choice == "3":
        data_2mol = []
        data_1mol = []

        # 收集2mol·L⁻¹数据
        print("\n现在输入 c(HCl) = 2 mol·L⁻¹ 的实验数据:")
        print("按照格式输入时间和对应的旋光度α_t，完成输入后请输入'done'")
        print("例如: 5,2.35")

        while True:
            entry = input("时间(min),旋光度α_t: ")
            if entry.lower() == 'done':
                break

            try:
                time, alpha_t = map(float, entry.split(','))
                data_2mol.append([time, alpha_t])
            except ValueError:
                print("输入格式错误，请按照'时间,旋光度'的格式输入")

        # 收集1mol·L⁻¹数据
        print("\n现在输入 c(HCl) = 1 mol·L⁻¹ 的实验数据:")
        print("按照格式输入时间和对应的旋光度α_t，完成输入后请输入'done'")
        print("例如: 5,2.35")

        while True:
            entry = input("时间(min),旋光度α_t: ")
            if entry.lower() == 'done':
                break

            try:
                time, alpha_t = map(float, entry.split(','))
                data_1mol.append([time, alpha_t])
            except ValueError:
                print("输入格式错误，请按照'时间,旋光度'的格式输入")

        # 确保有数据
        if not data_2mol or not data_1mol:
            print("错误：两种浓度的数据都需要输入。")
            return

        # 处理两种数据
        process_data(np.array(data_2mol), np.array(data_1mol), alpha_0, alpha_infinity,
                     "2 mol·L⁻¹", "1 mol·L⁻¹", room_temp, atm_pressure, experiment_temp,
                     polarimeter_tube_length, polarimeter_zero, experimenter, experiment_date)
    else:
        data = []
        hcl_conc = "2 mol·L⁻¹" if catalyst_choice == "1" else "1 mol·L⁻¹"

        # 收集数据点
        print(f"\n现在输入 c(HCl) = {hcl_conc} 的实验数据:")
        print("按照格式输入时间和对应的旋光度α_t，完成输入后请输入'done'")
        print("例如: 5,2.35")

        while True:
            entry = input("时间(min),旋光度α_t: ")
            if entry.lower() == 'done':
                break

            try:
                time, alpha_t = map(float, entry.split(','))
                data.append([time, alpha_t])
            except ValueError:
                print("输入格式错误，请按照'时间,旋光度'的格式输入")

        # 确保有数据
        if not data:
            print("错误：没有输入数据。")
            return

        # 处理单种数据
        process_single_data(np.array(data), alpha_0, alpha_infinity, hcl_conc,
                            room_temp, atm_pressure, experiment_temp,
                            polarimeter_tube_length, polarimeter_zero,
                            experimenter, experiment_date)


def process_single_data(data, alpha_0, alpha_infinity, hcl_conc,
                        room_temp, atm_pressure, experiment_temp,
                        polarimeter_tube_length, polarimeter_zero,
                        experimenter, experiment_date):
    """处理单个浓度的数据"""
    times = data[:, 0]
    alpha_values = data[:, 1]

    # 计算各项数据
    alpha_diff = alpha_values - alpha_infinity
    ln_alpha_diff = np.log(alpha_diff)
    lg_alpha_diff = np.log10(alpha_diff)

    # 计算相对旋光度变化百分比
    percent_rotation_change = ((alpha_values - alpha_0) / (alpha_infinity - alpha_0)) * 100
    percent_reaction = ((alpha_0 - alpha_values) / (alpha_0 - alpha_infinity)) * 100

    # 线性回归计算k值
    slope, intercept, r_value, p_value, std_err = linregress(times, ln_alpha_diff)

    # k值为斜率的负值，因为原始方程为ln(α_t - α_∞) = -k·t/2.303 + ln(α_0 - α_∞)
    k = -slope
    k_min = k  # min^-1
    k_sec = k / 60  # sec^-1

    # 计算半衰期
    half_life = 0.693 / k

    # 准备详细的结果表格
    table_data = []
    for i in range(len(times)):
        table_data.append([
            times[i],
            alpha_values[i],
            alpha_diff[i],
            ln_alpha_diff[i],
            lg_alpha_diff[i],
            percent_rotation_change[i],
            percent_reaction[i]
        ])

    # 创建DataFrame用于保存和展示
    df = pd.DataFrame(table_data, columns=[
        "反应时间/min",
        f"α_t (c_HCl = {hcl_conc})",
        f"α_t - α_∞",
        f"ln(α_t - α_∞)",
        f"lg(α_t - α_∞)",
        "旋光度变化百分比/%",
        "反应进度/%"
    ])

    # 输出基本实验信息表
    print("\n" + "=" * 80)
    print(" " * 30 + "实验基本信息")
    print("=" * 80)

    info_table = [
        ["实验者", experimenter],
        ["实验日期", experiment_date],
        ["室温", f"{room_temp} °C"],
        ["大气压", f"{atm_pressure} mmHg"],
        ["实验温度", f"{experiment_temp} °C"],
        ["旋光管长度", f"{polarimeter_tube_length} dm"],
        ["催化剂浓度", hcl_conc],
        ["初始旋光度 α₀", f"{alpha_0}°"],
        ["完全水解后旋光度 α∞", f"{alpha_infinity}°"],
        ["旋光仪零点校正值", f"{polarimeter_zero}°"]
    ]

    print(tabulate(info_table, tablefmt="grid"))

    # 输出结果表格
    print("\n" + "=" * 80)
    print(" " * 20 + "表 2-6-1 蔗糖水解反应实验数据")
    print("=" * 80 + "\n")

    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".4f"))

    # 生成CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sucrose_hydrolysis_data_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n数据已保存至 {csv_filename}")

    # 计算结果
    results_table = [
        ["反应速率常数 k", f"{k:.6f} min⁻¹"],
        ["反应速率常数 k", f"{k_sec:.8f} sec⁻¹"],
        ["反应半衰期 t₁/₂", f"{half_life:.4f} min"],
        ["线性回归相关系数 R", f"{r_value:.6f}"],
        ["线性回归相关系数 R²", f"{r_value ** 2:.6f}"],
        ["线性回归截距", f"{intercept:.6f}"],
        ["线性回归标准误差", f"{std_err:.6f}"]
    ]

    print("\n" + "=" * 80)
    print(" " * 30 + "计算结果")
    print("=" * 80 + "\n")

    print(tabulate(results_table, tablefmt="grid"))

    # 使用matplotlib绘制ln(α_t - α_∞)与t的关系图
    plt.figure(figsize=(10, 6))
    plt.scatter(times, ln_alpha_diff, color='blue', marker='o')

    # 添加回归线
    regression_line = intercept + slope * times
    plt.plot(times, regression_line, color='red', linewidth=2)

    # 添加图表标签
    plt.xlabel('反应时间 t (min)')
    plt.ylabel('ln(α_t - α_∞)')
    plt.title(f'蔗糖水解反应 ln(α_t - α_∞) vs t 图 (c_HCl = {hcl_conc})')
    plt.grid(True)

    # 在图表上显示k值和半衰期
    equation_text = f'ln(α_t - α_∞) = {slope:.4f}·t + {intercept:.4f}\n'
    equation_text += f'k = {k:.4f} min⁻¹\n'
    equation_text += f't₁/₂ = {half_life:.4f} min\n'
    equation_text += f'R² = {r_value ** 2:.4f}'

    plt.annotate(equation_text,
                 xy=(0.65, 0.2), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    # 保存图片
    plot_filename = f"sucrose_hydrolysis_plot_{hcl_conc.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存为 '{plot_filename}'")

    # 显示图表
    plt.show()

    # 反应进度图
    plt.figure(figsize=(10, 6))
    plt.scatter(times, percent_reaction, color='green', marker='s')
    plt.plot(times, percent_reaction, color='green', linestyle='-')
    plt.xlabel('反应时间 t (min)')
    plt.ylabel('反应进度 (%)')
    plt.title(f'蔗糖水解反应进度 (c_HCl = {hcl_conc})')
    plt.grid(True)

    progress_filename = f"sucrose_hydrolysis_progress_{hcl_conc.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(progress_filename, dpi=300, bbox_inches='tight')
    print(f"反应进度图已保存为 '{progress_filename}'")

    plt.show()


def process_data(data_2mol, data_1mol, alpha_0, alpha_infinity,
                 hcl_conc_2mol, hcl_conc_1mol, room_temp, atm_pressure,
                 experiment_temp, polarimeter_tube_length, polarimeter_zero,
                 experimenter, experiment_date):
    """处理两个浓度的数据并比较"""
    print("\n首先处理 c(HCl) = 2 mol·L⁻¹ 的数据:")
    process_single_data(data_2mol, alpha_0, alpha_infinity, hcl_conc_2mol,
                        room_temp, atm_pressure, experiment_temp,
                        polarimeter_tube_length, polarimeter_zero,
                        experimenter, experiment_date)

    print("\n接下来处理 c(HCl) = 1 mol·L⁻¹ 的数据:")
    process_single_data(data_1mol, alpha_0, alpha_infinity, hcl_conc_1mol,
                        room_temp, atm_pressure, experiment_temp,
                        polarimeter_tube_length, polarimeter_zero,
                        experimenter, experiment_date)

    # 计算两种浓度的k值用于比较
    times_2mol = data_2mol[:, 0]
    alpha_values_2mol = data_2mol[:, 1]
    alpha_diff_2mol = alpha_values_2mol - alpha_infinity
    ln_alpha_diff_2mol = np.log(alpha_diff_2mol)

    times_1mol = data_1mol[:, 0]
    alpha_values_1mol = data_1mol[:, 1]
    alpha_diff_1mol = alpha_values_1mol - alpha_infinity
    ln_alpha_diff_1mol = np.log(alpha_diff_1mol)

    slope_2mol, intercept_2mol, r_value_2mol, _, _ = linregress(times_2mol, ln_alpha_diff_2mol)
    slope_1mol, intercept_1mol, r_value_1mol, _, _ = linregress(times_1mol, ln_alpha_diff_1mol)

    k_2mol = -slope_2mol
    k_1mol = -slope_1mol

    half_life_2mol = 0.693 / k_2mol
    half_life_1mol = 0.693 / k_1mol

    # 比较两个浓度的结果
    comparison_table = [
        ["参数", f"c(HCl) = 2 mol·L⁻¹", f"c(HCl) = 1 mol·L⁻¹", "比值 (2mol/1mol)"],
        ["反应速率常数 k (min⁻¹)", f"{k_2mol:.6f}", f"{k_1mol:.6f}", f"{k_2mol / k_1mol:.2f}"],
        ["反应半衰期 t₁/₂ (min)", f"{half_life_2mol:.4f}", f"{half_life_1mol:.4f}",
         f"{half_life_2mol / half_life_1mol:.2f}"],
        ["线性回归相关系数 R²", f"{r_value_2mol ** 2:.6f}", f"{r_value_1mol ** 2:.6f}", "-"]
    ]

    print("\n" + "=" * 80)
    print(" " * 20 + "不同催化剂浓度的反应速率常数比较")
    print("=" * 80 + "\n")

    print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))

    # 两种浓度的ln(α_t - α_∞)对比图
    plt.figure(figsize=(12, 8))

    # 2 mol/L数据点和回归线
    plt.scatter(times_2mol, ln_alpha_diff_2mol, color='blue', marker='o', label=f'2 mol·L⁻¹ 数据点')
    regression_line_2mol = intercept_2mol + slope_2mol * times_2mol
    plt.plot(times_2mol, regression_line_2mol, color='blue', linewidth=2, linestyle='-',
             label=f'2 mol·L⁻¹ 拟合线 (k = {k_2mol:.4f} min⁻¹)')

    # 1 mol/L数据点和回归线
    plt.scatter(times_1mol, ln_alpha_diff_1mol, color='red', marker='s', label=f'1 mol·L⁻¹ 数据点')
    regression_line_1mol = intercept_1mol + slope_1mol * times_1mol
    plt.plot(times_1mol, regression_line_1mol, color='red', linewidth=2, linestyle='--',
             label=f'1 mol·L⁻¹ 拟合线 (k = {k_1mol:.4f} min⁻¹)')

    plt.xlabel('反应时间 t (min)')
    plt.ylabel('ln(α_t - α_∞)')
    plt.title('不同催化剂浓度的蔗糖水解反应动力学对比')
    plt.grid(True)
    plt.legend()

    # 添加结果文本
    comparison_text = f'2 mol·L⁻¹: k = {k_2mol:.4f} min⁻¹, t₁/₂ = {half_life_2mol:.2f} min\n'
    comparison_text += f'1 mol·L⁻¹: k = {k_1mol:.4f} min⁻¹, t₁/₂ = {half_life_1mol:.2f} min\n'
    comparison_text += f'k比值 (2mol/1mol) = {k_2mol / k_1mol:.2f}'

    plt.annotate(comparison_text,
                 xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    # 保存对比图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = f"sucrose_hydrolysis_comparison_{timestamp}.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存为 '{comparison_filename}'")

    plt.show()

    # 结论
    print("\n" + "=" * 80)
    print(" " * 30 + "实验结论")
    print("=" * 80 + "\n")

    print(f"1. 从实验结果可见，催化剂浓度为2 mol·L⁻¹时的反应速率常数k为{k_2mol:.6f} min⁻¹，")
    print(f"   而催化剂浓度为1 mol·L⁻¹时的反应速率常数k为{k_1mol:.6f} min⁻¹。")
    print(f"2. 催化剂浓度增加一倍，反应速率常数增加了{k_2mol / k_1mol:.2f}倍。")
    if k_2mol / k_1mol > 1.8 and k_2mol / k_1mol < 2.2:
        print("   这一比例接近于2，表明反应速率与催化剂浓度（H⁺浓度）成正比。")
    elif k_2mol / k_1mol > 2.2:
        print("   这一比例大于2，表明催化剂浓度增加可能有协同效应。")
    else:
        print("   这一比例小于2，表明催化剂浓度增加时效率略有下降。")
    print("3. 蔗糖水解反应符合一级反应动力学特征，ln(α_t - α_∞)与反应时间t呈良好的线性关系。")
    print(f"4. 在实验温度{experiment_temp}°C条件下，反应的半衰期分别为：")
    print(f"   2 mol·L⁻¹: t₁/₂ = {half_life_2mol:.2f} min")
    print(f"   1 mol·L⁻¹: t₁/₂ = {half_life_1mol:.2f} min")


if __name__ == "__main__":
    main()