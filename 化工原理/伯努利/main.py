import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 基本参数设置
g = 9.81  # 重力加速度 (m/s^2)
rho = 1000  # 水密度 (kg/m^3)
alpha = 1  # 动能修正系数

# 截面参数 (单位：mm 转换为 m)
d_A = 0.014  # A截面直径 (m)
d_B = 0.028  # B截面直径 (m)
d_C = 0.014  # C截面直径 (m)
d_D = 0.014  # D截面直径 (m)

# 截面积 (m^2)
A_A = np.pi * (d_A / 2) ** 2
A_B = np.pi * (d_B / 2) ** 2
A_C = np.pi * (d_C / 2) ** 2
A_D = np.pi * (d_D / 2) ** 2

# 位能高度 (单位：mm 转换为 m)
z_A = 0.1  # A截面高度 (m)
z_B = 0.1  # B截面高度 (m)
z_C = 0.1  # C截面高度 (m)
z_D = 0.0  # D截面高度 (m)


# 计算流速 (m/s)
def calculate_velocity(Q, A):
    return Q / A


# 计算动能头 (m)
def calculate_kinetic_head(v):
    return alpha * (v ** 2) / (2 * g)


# 计算总水头 (m)
def calculate_total_head(p_head, z, v_head):
    return p_head + z + v_head


# 计算各截面数据
def process_section_data(Q, p_data, section, A, z):
    v = calculate_velocity(Q, A)
    v_head = calculate_kinetic_head(v)
    p_head = p_data / 1000  # mmH2O to mH2O
    total_head = calculate_total_head(p_head, z, v_head)
    return v, v_head, p_head, total_head


# 读取CSV文件数据
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return None

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        if df.shape[0] != 3 or df.shape[1] != 17:  # 假设3组数据，16列数据+1列组别
            print("错误：数据格式不符合预期（应有3行数据和17列）！")
            return None

        # 提取数据
        data = []
        for i in range(3):
            group_label = df.iloc[i, 0]
            Q = df.iloc[i, 1] / 3600 / 1000  # 流量 l/h 转换为 m^3/s
            p_data = df.iloc[i, 2:].values.tolist()  # 压强数据
            data.append((group_label, Q, p_data))
        return data
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None


# 处理数据并输出结果
def analyze_data(Q, p_data, label):
    print(f"\n=== {label} 数据分析 (流量: {Q * 3600 * 1000:.0f} l/h) ===")

    # 假设测量点 1-15 中某些点对应 A, B, C, D 截面 (根据实际实验调整)
    # 示例：点1为A截面静压，点3为B截面静压，点5为C截面静压，点7为D截面静压
    v_A, v_head_A, p_head_A, total_head_A = process_section_data(Q, p_data[0], 'A', A_A, z_A)
    v_B, v_head_B, p_head_B, total_head_B = process_section_data(Q, p_data[2], 'B', A_B, z_B)
    v_C, v_head_C, p_head_C, total_head_C = process_section_data(Q, p_data[4], 'C', A_C, z_C)
    v_D, v_head_D, p_head_D, total_head_D = process_section_data(Q, p_data[6], 'D', A_D, z_D)

    # 输出计算结果
    print(
        f"A截面 - 流速: {v_A:.3f} m/s, 静压头: {p_head_A:.3f} m, 动能头: {v_head_A:.3f} m, 总水头: {total_head_A:.3f} m")
    print(
        f"B截面 - 流速: {v_B:.3f} m/s, 静压头: {p_head_B:.3f} m, 动能头: {v_head_B:.3f} m, 总水头: {total_head_B:.3f} m")
    print(
        f"C截面 - 流速: {v_C:.3f} m/s, 静压头: {p_head_C:.3f} m, 动能头: {v_head_C:.3f} m, 总水头: {total_head_C:.3f} m")
    print(
        f"D截面 - 流速: {v_D:.3f} m/s, 静压头: {p_head_D:.3f} m, 动能头: {v_head_D:.3f} m, 总水头: {total_head_D:.3f} m")

    # 压头损失 (以A到D为例)
    head_loss = total_head_A - total_head_D
    print(f"压头损失 (A到D): {head_loss:.3f} m")

    return [p_head_A, p_head_B, p_head_C, p_head_D], head_loss, p_data


# 绘制图形
def plot_pressure_data(data_Q1, data_Q2, data_Q3, Q1, Q2, Q3):
    points = np.arange(1, 16)
    plt.figure(figsize=(10, 6))
    plt.plot(points, data_Q1, label=f"{Q1 * 3600 * 1000:.0f} l/h", marker='o')
    plt.plot(points, data_Q2, label=f"{Q2 * 3600 * 1000:.0f} l/h", marker='s')
    plt.plot(points, data_Q3, label=f"{Q3 * 3600 * 1000:.0f} l/h", marker='^')
    plt.xlabel("测试点标号")
    plt.ylabel("压强 (mmH2O)")
    plt.title("能量转换位置--压强图")
    plt.legend()
    plt.grid(True)
    plt.show()


# 生成实验报告
def generate_report(Q1, Q2, Q3, loss_Q1, loss_Q2, loss_Q3):
    report = """
=== 能量转换（伯努利方程）实验报告 ===

一、实验目的：
1. 演示流体在管内流动时静压能、动能、位能的相互转换关系，加深对伯努利方程的理解。
2. 通过能量变化了解流体在管内流动时的阻力表现形式。
3. 观测流体经过扩大、收缩管段时，各截面静压头的变化过程。

二、实验内容：
1. 测量不同流量下的压头，并进行分析比较。
2. 测定管中水的平均流速及点C、D处的点流速，并比较。

三、实验原理：
伯努利方程：p/γ + z + α*v^2/(2g) = 常数
通过测量静压头、计算流速及动能头，得到各截面总水头。

四、实验数据处理：
1. 流速、静压头、动能头及总水头的计算结果见上文输出。
2. 压头损失分析：
   - 流量 {:.0f} l/h: 压头损失 = {:.3f} m
   - 流量 {:.0f} l/h: 压头损失 = {:.3f} m
   - 流量 {:.0f} l/h: 压头损失 = {:.3f} m
3. 文丘里测量段分析：
   流体经过扩大管段（A到B）时，流速减小，静压头增加；
   流体经过收缩管段（B到C）时，流速增加，静压头减小；
   符合伯努利方程能量守恒原理。

五、结论：
1. 实验验证了伯努利方程的正确性，流体在流动过程中能量形式发生转换，但总能量基本守恒。
2. 压头损失随着流量的增加而增大，反映了流体阻力与流速的关系。
3. 通过压强图可以直观观察到静压头在各截面的变化趋势。
""".format(Q1 * 3600 * 1000, loss_Q1, Q2 * 3600 * 1000, loss_Q2, Q3 * 3600 * 1000, loss_Q3)
    print(report)


# 主程序：读取文件并处理数据
def main():
    file_path = input("请输入数据文件路径 (例如: experiment_data.csv): ")
    data = load_data(file_path)

    if data is None:
        print("无法加载数据，程序退出。")
        return

    # 提取三组数据
    label1, Q1, p_data_Q1 = data[0]
    label2, Q2, p_data_Q2 = data[1]
    label3, Q3, p_data_Q3 = data[2]

    # 分析数据
    p_heads_Q1, loss_Q1, raw_data_Q1 = analyze_data(Q1, p_data_Q1, label1)
    p_heads_Q2, loss_Q2, raw_data_Q2 = analyze_data(Q2, p_data_Q2, label2)
    p_heads_Q3, loss_Q3, raw_data_Q3 = analyze_data(Q3, p_data_Q3, label3)

    # 绘制压强图
    plot_pressure_data(raw_data_Q1, raw_data_Q2, raw_data_Q3, Q1, Q2, Q3)

    # 生成实验报告
    generate_report(Q1, Q2, Q3, loss_Q1, loss_Q2, loss_Q3)


# 运行程序
if __name__ == "__main__":
    main()
