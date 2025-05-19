import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 全局参数设置
rho = 1000  # 水密度，kg/m^3
g = 9.81  # 重力加速度，m/s^2
eta_motor = 0.85  # 假设电动机效率为85%


# 从文本文件读取数据的函数
def read_data_from_txt(filename):
    """从文本文件中读取实验数据"""
    data_pump = []
    data_pipeline = []
    data_friction = []
    parameters = {}
    current_table = None

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if 'Table 1: Centrifugal Pump Data' in line:
                current_table = 'pump'
                print("Found Table 1: Centrifugal Pump Data at line", i + 1)
                continue
            elif 'Table 2: Pipeline Characteristic Data' in line:
                current_table = 'pipeline'
                print("Found Table 2: Pipeline Characteristic Data at line", i + 1)
                continue
            elif 'Table 3: Friction Loss Data' in line:
                current_table = 'friction'
                print("Found Table 3: Friction Loss Data at line", i + 1)
                continue
            elif 'Parameters:' in line:
                current_table = 'parameters'
                print("Found Parameters at line", i + 1)
                continue

            if current_table == 'pump':
                if 'No' in line or 'Freq_Hz' in line:
                    print("Skipping header for Table 1 at line", i + 1, ":", line)
                    continue
                values = line.split()
                if len(values) == 6:
                    try:
                        data_pump.append([float(v) for v in values])
                    except ValueError as e:
                        print(f"Error converting data in Table 1 at line {i + 1}: {line}, Error: {e}")
            elif current_table == 'pipeline':
                if 'No' in line or 'Freq_VFD' in line:
                    print("Skipping header for Table 2 at line", i + 1, ":", line)
                    continue
                values = line.split()
                if len(values) == 6:
                    try:
                        data_pipeline.append([float(v) for v in values])
                    except ValueError as e:
                        print(f"Error converting data in Table 2 at line {i + 1}: {line}, Error: {e}")
            elif current_table == 'friction':
                if 'No' in line or 'Q_m3s' in line:
                    print("Skipping header for Table 3 at line", i + 1, ":", line)
                    continue
                values = line.split()
                if len(values) == 3:
                    try:
                        data_friction.append([float(v) for v in values])
                    except ValueError as e:
                        print(f"Error converting data in Table 3 at line {i + 1}: {line}, Error: {e}")
            elif current_table == 'parameters':
                if ':' in line:
                    key, value = line.split(':')
                    try:
                        parameters[key.strip()] = float(value.strip())
                    except ValueError as e:
                        print(f"Error converting parameter at line {i + 1}: {line}, Error: {e}")

    # 将数据转换为字典格式，适应提供的列顺序
    data_pump_dict = {
        'Freq_Hz': [row[0] for row in data_pump],
        'Q_m3h': [row[1] for row in data_pump],  # 假设单位是m³/h，稍后转换为m³/s
        'P_vacuum_kPa': [row[2] for row in data_pump],
        'P_pressure_kPa': [row[3] for row in data_pump],
        'Power_kW': [row[4] for row in data_pump],
        'No': [row[5] for row in data_pump]
    }
    data_pipeline_dict = {
        'Freq_VFD': [row[0] for row in data_pipeline],
        'Freq_Hz': [row[1] for row in data_pipeline],
        'Q_m3s': [row[2] for row in data_pipeline],  # 单位已经是m³/s
        'P_vacuum_kPa': [row[3] for row in data_pipeline],
        'P_pressure_kPa': [row[4] for row in data_pipeline],
        'No': [row[5] for row in data_pipeline]
    }
    data_friction_dict = {
        'Q_m3h': [row[0] for row in data_friction],  # 假设单位是m³/h，稍后转换为m³/s
        'P_diff_kPa': [row[1] for row in data_friction],
        'No': [row[2] for row in data_friction]
    }

    print("Data loaded successfully:")
    print("Table 1 (Pump) rows:", len(data_pump))
    print("Table 2 (Pipeline) rows:", len(data_pipeline))
    print("Table 3 (Friction) rows:", len(data_friction))
    print("Parameters:", parameters)

    return data_pump_dict, data_pipeline_dict, data_friction_dict, parameters


# 计算函数
def calculate_pump_head(P_vacuum, P_pressure, Z_out=0, Z_in=0, u_out=0, u_in=0):
    """计算离心泵扬程H，单位：m"""
    P_diff = (P_pressure - P_vacuum) * 1000  # 压差，Pa
    H = (P_diff / (rho * g)) + (Z_out - Z_in) + ((u_out ** 2 - u_in ** 2) / (2 * g))
    return H


def calculate_pump_power(Power_read, eta_motor):
    """计算轴功率N，单位：kW"""
    N = Power_read * eta_motor
    return N


def calculate_pump_efficiency(H, Q, N):
    """计算泵效率eta"""
    Ne = (rho * g * H * Q) / 1000  # 有效功率，kW
    eta = Ne / N if N > 0 else 0
    return eta


def calculate_friction_lambda(delta_P, u, d, l):
    """计算直管摩擦系数lambda"""
    lambda_ = (2 * delta_P * d) / (rho * u ** 2 * l) if u > 0 else 0
    return lambda_


def calculate_reynolds(u, d, rho, mu):
    """计算雷诺数Re"""
    Re = (rho * u * d) / mu if u > 0 else 0
    return Re


# 数据处理函数
def process_pump_data(data):
    """处理离心泵特性曲线数据"""
    df = pd.DataFrame(data)
    # 将Q从m³/h转换为m³/s
    df['Q_m3s'] = df['Q_m3h'] / 3600.0
    df['H_m'] = [calculate_pump_head(p_vac, p_press) for p_vac, p_press in
                 zip(df['P_vacuum_kPa'], df['P_pressure_kPa'])]
    df['N_kW'] = [calculate_pump_power(p, eta_motor) for p in df['Power_kW']]
    df['eta'] = [calculate_pump_efficiency(h, q, n) for h, q, n in zip(df['H_m'], df['Q_m3s'], df['N_kW'])]
    return df


def process_pipeline_data(data):
    """处理管路特性曲线数据"""
    df = pd.DataFrame(data)
    df['H_m'] = [calculate_pump_head(p_vac, p_press) for p_vac, p_press in
                 zip(df['P_vacuum_kPa'], df['P_pressure_kPa'])]
    return df


def process_friction_data(data, d_m, l_m, mu):
    """处理直管摩擦数据（仅使用P_diff_kPa）"""
    df = pd.DataFrame(data)
    # 将Q从m³/h转换为m³/s
    df['Q_m3s'] = df['Q_m3h'] / 3600.0
    u = df['Q_m3s'] / (np.pi * (d_m ** 2) / 4)  # 流速，m/s
    delta_P = df['P_diff_kPa'] * 1000  # 压差，Pa
    df['delta_P_Pa'] = delta_P
    df['Re'] = [calculate_reynolds(u_val, d_m, rho, mu) for u_val in u]
    df['lambda'] = [calculate_friction_lambda(dp, u_val, d_m, l_m) for dp, u_val in zip(delta_P, u)]
    return df


# 生成报告和绘图
def generate_report_and_plots(pump_df, pipeline_df, friction_df):
    """生成实验报告和绘图"""
    with open('experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write("流动过程综合实验报告\n")
        f.write("======================\n\n")
        f.write("1. 离心泵特性曲线计算结果\n")
        f.write(pump_df.to_string(index=False) + "\n\n")
        f.write("2. 管路特性曲线计算结果\n")
        f.write(pipeline_df.to_string(index=False) + "\n\n")
        f.write("3. 直管摩擦系数与雷诺数关系\n")
        f.write(friction_df.to_string(index=False) + "\n\n")

    # 绘图
    # 1. 离心泵特性曲线
    plt.figure(figsize=(10, 6))
    plt.plot(pump_df['Q_m3s'], pump_df['H_m'], label='H-Q Curve', marker='o')
    plt.plot(pump_df['Q_m3s'], pump_df['N_kW'], label='N-Q Curve', marker='s')
    plt.plot(pump_df['Q_m3s'], pump_df['eta'], label='η-Q Curve', marker='^')
    plt.xlabel('Flow Rate Q (m³/s)')
    plt.ylabel('Head H (m), Power N (kW), Efficiency η')
    plt.title('Centrifugal Pump Characteristic Curves')
    plt.legend()
    plt.grid()
    plt.savefig('pump_curves.png')
    plt.close()

    # 2. 管路特性曲线
    plt.figure(figsize=(10, 6))
    plt.plot(pipeline_df['Q_m3s'], pipeline_df['H_m'], label='Pipeline H-Q Curve', marker='o')
    plt.xlabel('Flow Rate Q (m³/s)')
    plt.ylabel('Head H (m)')
    plt.title('Pipeline Characteristic Curve')
    plt.legend()
    plt.grid()
    plt.savefig('pipeline_curve.png')
    plt.close()

    # 3. 直管摩擦系数与雷诺数关系（对数坐标）
    plt.figure(figsize=(10, 6))
    plt.loglog(friction_df['Re'], friction_df['lambda'], label='λ-Re Curve', marker='o')
    plt.xlabel('Reynolds Number Re')
    plt.ylabel('Friction Factor λ')
    plt.title('Friction Factor vs Reynolds Number')
    plt.legend()
    plt.grid()
    plt.savefig('friction_re_curve.png')
    plt.close()


# 主程序
if __name__ == "__main__":
    print("欢迎使用流动过程综合实验数据处理程序！")
    filename = input("请输入包含实验数据的文本文件名（如 experiment_data.txt）：")

    try:
        data_pump, data_pipeline, data_friction, params = read_data_from_txt(filename)
        d_m = params.get('Pipe Diameter (m)', 0.019)
        l_m = params.get('Pipe Length (m)', 1.0)
        mu = params.get('Fluid Viscosity (N·s/m^2)', 0.74772)

        # 处理数据
        pump_df = process_pump_data(data_pump)
        pipeline_df = process_pipeline_data(data_pipeline)
        friction_df = process_friction_data(data_friction, d_m, l_m, mu)

        # 生成报告和绘图
        generate_report_and_plots(pump_df, pipeline_df, friction_df)
        print("实验数据处理完成，结果已保存至'experiment_report.txt'，绘图已保存为PNG文件。")
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到，请检查文件名和路径。")
    except Exception as e:
        print(f"错误：读取或处理数据时发生问题，详细信息：{str(e)}")
