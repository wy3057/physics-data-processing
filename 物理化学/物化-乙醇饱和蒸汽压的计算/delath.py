import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
import time

# 设置中文显示
try:
    # 尝试设置中文字体
    font = FontProperties(fname=r'C:\Windows\Fonts\SimHei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    print("无法设置中文字体，图表将使用默认字体")


# 用于在计算过程中添加暂停，以便用户阅读
def pause_for_reading(seconds=1):
    """暂停执行指定的秒数，让用户有时间阅读输出"""
    time.sleep(seconds)


# 数据输入
def input_data():
    """手动输入实验数据"""
    print("请输入实验数据:")

    # 实验条件
    sample_name = input("被测液体: ")
    room_temp = float(input("室温(°C): "))
    atm_pressure = float(input("大气压(kPa): "))

    # 输入数据点数
    n = int(input("输入数据点数量: "))

    # 创建空列表存储数据
    temps = []  # t/°C
    pressures = []  # Δp/kPa

    # 输入每个数据点
    for i in range(n):
        print(f"\n数据点 {i + 1}:")
        t = float(input("温度 t(°C): "))
        p = float(input("压力计读数 Δp(kPa): "))
        temps.append(t)
        pressures.append(p)

    return sample_name, room_temp, atm_pressure, temps, pressures


# 计算数据
def calculate_data(temps, pressures, atm_pressure):
    """计算表格中的所有数据"""
    # 创建DataFrame存储数据
    df = pd.DataFrame()

    # 显示详细计算过程标题
    print("\n\033[1;36m【详细计算过程】\033[0m")
    print("=" * 100)
    print("\033[1;33m第一步：数据预处理与基本计算\033[0m")
    pause_for_reading(0.5)

    print("\n首先，我们需要进行以下计算：")
    print("1. 将摄氏温度(°C)转换为开尔文温度(K)")
    print("2. 计算温度倒数并乘以1000(1/T×10³)")
    print("3. 计算饱和蒸气压(p = Patm - Δp)")
    print("4. 计算饱和蒸气压的常用对数(lgp = log10(p))")
    pause_for_reading(1)

    # 温度数据
    print("\n\033[1;33m处理温度数据：\033[0m")
    df['t/°C'] = temps
    print(f"输入的温度数据(°C): {temps}")

    df['T/K'] = df['t/°C'] + 273.15
    print(f"转换为开尔文温度(K): {list(df['T/K'])}")
    print(f"转换公式: T(K) = t(°C) + 273.15")
    pause_for_reading(0.5)

    df['1/T×10³/(1/K)'] = (1 / df['T/K']) * 1000
    print(f"计算1/T×10³(1/K): {list(df['1/T×10³/(1/K)'])}")
    print(f"计算公式: 1/T×10³ = (1/T(K)) × 1000")
    pause_for_reading(0.5)

    # 压力数据
    print("\n\033[1;33m处理压力数据：\033[0m")
    df['Δp/kPa'] = pressures
    print(f"输入的压力计读数(kPa): {pressures}")

    df['饱和蒸气压 p=Patm-Δp'] = atm_pressure - df['Δp/kPa']
    print(f"计算饱和蒸气压(kPa): {list(df['饱和蒸气压 p=Patm-Δp'])}")
    print(f"计算公式: p = Patm - Δp = {atm_pressure} - Δp")
    pause_for_reading(0.5)

    df['lgp'] = np.log10(df['饱和蒸气压 p=Patm-Δp'])
    print(f"计算lgp: {list(df['lgp'])}")
    print(f"计算公式: lgp = log10(p)")
    pause_for_reading(0.5)

    # 显示每个数据点的详细计算过程
    print("\n\033[1;33m每个数据点的详细计算：\033[0m")
    for i in range(len(temps)):
        t = temps[i]
        T = t + 273.15
        inv_T = (1 / T) * 1000
        p = atm_pressure - pressures[i]
        lg_p = np.log10(p)

        print(f"\n\033[1m数据点 {i + 1} 计算过程:\033[0m")
        print(f"  温度转换:")
        print(f"    t = {t}°C")
        print(f"    T = t + 273.15 = {t} + 273.15 = {T} K")

        print(f"  温度倒数:")
        print(f"    1/T = 1/{T} = {1 / T:.8f} 1/K")
        print(f"    1/T×10³ = {1 / T:.8f} × 1000 = {inv_T:.6f} 1/K")

        print(f"  饱和蒸气压:")
        print(f"    Patm = {atm_pressure} kPa")
        print(f"    Δp = {pressures[i]} kPa")
        print(f"    p = Patm - Δp = {atm_pressure} - {pressures[i]} = {p:.6f} kPa")

        print(f"  对数计算:")
        print(f"    lgp = log10(p) = log10({p:.6f}) = {lg_p:.6f}")

        pause_for_reading(0.5)

    print("=" * 100)

    return df


# 拟合直线
def fit_line(x, y):
    """拟合lgp与1/T的线性关系"""

    # 线性函数 y = mx + b
    def linear(x, m, b):
        return m * x + b

    print("\n\033[1;33m第二步：线性回归分析\033[0m")
    print("\n根据克劳修斯-克拉珀龙方程，lgp与1/T之间存在线性关系：")
    print("lgp = -ΔvapHm/(2.303×R) × (1/T) + C")
    print("这可以简化为线性方程：lgp = m × (1/T×10³) + b")
    print("其中，m = -ΔvapHm/(2.303×R×10³)，b = C")
    pause_for_reading(1)

    print("\n现在使用最小二乘法进行线性回归拟合...")

    # 使用最小二乘法拟合
    params, covariance = curve_fit(linear, x, y)
    m, b = params

    print(f"\n拟合完成！得到参数：")
    print(f"  斜率 m = {m:.6f}")
    print(f"  截距 b = {b:.6f}")
    pause_for_reading(0.5)

    # 计算相关系数R²
    y_pred = linear(x, m, b)
    residuals = y - y_pred
    mean_y = np.mean(y)
    ss_total = np.sum((y - mean_y) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    print("\n\033[1;33m计算拟合优度：决定系数R²\033[0m")
    print("决定系数R²衡量模型解释因变量变异性的程度，R²越接近1，拟合效果越好")
    pause_for_reading(0.5)

    print("\n预测值与实际值的比较:")
    print("-" * 70)
    print(f"{'数据点':^8}{'1/T×10³':^15}{'实际lgp':^15}{'预测lgp':^15}{'残差':^15}")
    print("-" * 70)

    for i in range(len(x)):
        pred = m * x[i] + b
        print(f"{i + 1:^8}{x[i]:^15.6f}{y[i]:^15.6f}{pred:^15.6f}{y[i] - pred:^15.6f}")

    print("-" * 70)
    pause_for_reading(1)

    print("\n决定系数R²的计算:")
    print(f"  1. 计算因变量lgp的平均值: mean_y = {mean_y:.6f}")
    print(f"  2. 计算总离差平方和: SS_total = Σ(y - mean_y)² = {ss_total:.6f}")
    print(f"  3. 计算残差平方和: SS_residual = Σ(y - y_pred)² = {ss_residual:.6f}")
    print(
        f"  4. 计算决定系数: R² = 1 - (SS_residual/SS_total) = 1 - ({ss_residual:.6f}/{ss_total:.6f}) = {r_squared:.6f}")

    print(f"\n最终结果: R² = {r_squared:.6f}")
    if r_squared > 0.95:
        print("  这是一个很好的拟合结果！(R² > 0.95)")
    elif r_squared > 0.9:
        print("  这是一个不错的拟合结果。(R² > 0.9)")
    else:
        print("  拟合效果不理想，建议检查数据或实验条件。(R² < 0.9)")

    print("\n拟合方程: lgp = ({:.4f}) × (1/T×10³) + ({:.4f})".format(m, b))
    pause_for_reading(1)

    return m, b, r_squared


# 计算摩尔蒸发热
def calculate_heat_vaporization(slope):
    """根据直线斜率计算摩尔蒸发热"""
    # 斜率 m = -ΔvapHm/(2.303*R*10³)
    # 其中R是气体常数 8.314 J/(mol·K)
    R = 8.314  # J/(mol·K)

    print("\n\033[1;33m第三步：计算摩尔蒸发热\033[0m")
    print("\n根据克劳修斯-克拉珀龙方程，我们可以通过斜率计算摩尔蒸发热:")
    print("  原方程: lgp = -ΔvapHm/(2.303×R) × (1/T) + C")
    print("  因为我们用的是1/T×10³，所以需要考虑这个因素")
    print("  斜率 m = -ΔvapHm/(2.303×R×10³)")
    print("  所以 ΔvapHm = -2.303 × R × m × 10³")
    pause_for_reading(1)

    vap_heat = -2.303 * R * slope * 1000

    print("\n计算步骤:")
    print(f"  1. 气体常数 R = {R} J/(mol·K)")
    print(f"  2. 拟合得到的斜率 m = {slope:.6f}")
    print(f"  3. ΔvapHm = -2.303 × {R} × ({slope:.6f}) × 1000")
    print(f"  4. ΔvapHm = -2.303 × {R} × {slope:.6f} × 1000 = {vap_heat:.2f} J/mol")

    print(f"\n最终结果:")
    print(f"  摩尔蒸发热 ΔvapHm = {vap_heat:.2f} J/mol = {vap_heat / 1000:.2f} kJ/mol")

    # 判断结果合理性
    if 20000 < vap_heat < 60000:
        print("  这个值在大多数液体的合理范围内 (20-60 kJ/mol)")
    elif vap_heat <= 0:
        print("  警告：计算得到的摩尔蒸发热为负值或零，这在物理上是不合理的。请检查数据或计算过程。")
    elif vap_heat > 100000:
        print("  警告：计算得到的摩尔蒸发热异常大，请检查数据或计算过程。")

    pause_for_reading(1)

    return vap_heat


# 绘制图表
def plot_data(df, m, b, r_squared, sample_name):
    """绘制lgp vs 1/T图表"""
    print("\n\033[1;33m第四步：绘制图表分析\033[0m")
    print("\n根据克劳修斯-克拉珀龙方程，lgp与1/T的关系应该是线性的")
    print("现在将创建图表，展示实验数据点和拟合直线")
    pause_for_reading(1)

    print("\n绘图过程:")
    print("  1. 创建散点图显示实验数据点")
    print("  2. 根据拟合参数绘制拟合直线")
    print("  3. 添加图表标题、坐标轴标签和图例")
    print("  4. 保存图表到文件")
    pause_for_reading(0.5)

    # 创建绘图
    print("\n创建图表...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 散点图
    ax.scatter(df['1/T×10³/(1/K)'], df['lgp'], color='blue', marker='o', s=60, label='实验数据')

    # 拟合直线
    x_range = np.linspace(min(df['1/T×10³/(1/K)']) - 0.1, max(df['1/T×10³/(1/K)']) + 0.1, 100)
    y_range = m * x_range + b
    ax.plot(x_range, y_range, 'r-', label=f'拟合直线: y = {m:.4f}x + {b:.4f}, R² = {r_squared:.4f}')

    # 设置标签和标题
    ax.set_xlabel('1/T×10³ (1/K)', fontsize=12)
    ax.set_ylabel('lgp', fontsize=12)
    ax.set_title(f'{sample_name}的饱和蒸气压与温度关系', fontsize=14)

    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=10)

    # 添加方程到图表
    equation_text = f'lgp = {m:.4f} × (1/T×10³) + {b:.4f}'
    r2_text = f'R² = {r_squared:.4f}'
    ax.annotate(equation_text + '\n' + r2_text,
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))

    # 显示图表
    print(f"\n图表已创建，将保存为: {sample_name}_vapor_pressure.png")
    plt.tight_layout()
    plt.savefig(f'{sample_name}_vapor_pressure.png', dpi=300)
    print("显示图表...")
    plt.show()

    print("\n图表分析:")
    print("  1. 如果数据点基本落在拟合直线上，则实验结果符合克劳修斯-克拉珀龙方程")
    print("  2. 直线斜率反映了液体的摩尔蒸发热")
    print("  3. R²值接近1表示拟合效果好")
    pause_for_reading(1)


# 主函数
def main():
    """主程序"""
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    print("\033[1;32m" + "               饱和蒸气压测定数据处理程序" + "\033[0m")
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    print("\n本程序根据克劳修斯-克拉珀龙方程(Clausius-Clapeyron equation):")
    print("             lgp = -ΔvapHm/(2.303×R) × (1/T) + C")
    print("计算液体的摩尔蒸发热，并绘制lgp vs 1/T的关系图。")
    print("\n\033[1;33m程序将显示详细的计算过程，帮助您理解每一步操作。\033[0m")
    pause_for_reading(2)

    # 输入数据
    print("\n\033[1;33m实验数据输入\033[0m")
    sample_name, room_temp, atm_pressure, temps, pressures = input_data()

    # 计算数据
    df = calculate_data(temps, pressures, atm_pressure)

    # 拟合直线
    m, b, r_squared = fit_line(df['1/T×10³/(1/K)'], df['lgp'])

    # 计算摩尔蒸发热
    vap_heat = calculate_heat_vaporization(m)

    # 绘制图表
    plot_data(df, m, b, r_squared, sample_name)

    # 输出结果汇总


# 运行程序
if __name__ == "__main__":
    main()