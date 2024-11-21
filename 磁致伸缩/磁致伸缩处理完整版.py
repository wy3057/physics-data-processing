import matplotlib.pyplot as plt
def nanometers_to_meters(nanometers):
    """
    将纳米值转换为米值

    参数:
    nanometers: float，输入的纳米值

    返回:
    float，转换后的米值
    """
    meters = nanometers * 1e-9
    return meters
def calculate_magnetic_field(N, L, I):
    """
    计算螺线管内部的磁场强度

    参数:
    N: int，总匝数
    L: float，线圈长度（米）
    I: float，电流强度（安培）

    返回:
    float，磁场强度（特斯拉）
    """
    # 真空磁导率


    # 单位长度的匝数
    n = N / L  # 单位 匝/m

    # 磁场计算
    B = [n * t for t in I ] # 单位 T

    return B

def user_input():
    """
    从用户输入中获取 I 和 n 的值
    """
    print("请输入场强 I 和 n 数据，以逗号分隔，每组数据一行，输入 'done' 结束：")
    field_strengths = []
    n_values = []

    while True:
        input_data = input("输入一组数据 (I, n)：")
        if input_data.lower() == 'done':
            break
        try:
            # 用户输入拆分
            i_value, n_value = map(float, input_data.split(','))
            field_strengths.append(i_value)
            n_values.append(n_value)

            # 进一步验证输入数据
            if i_value <= 0:
                raise ValueError("场强 I 和 n 必须大于 0")
        except ValueError as e:
            print(f"输入格式错误或数据不合理：{e}，请重新输入，例如：1, -1")
        except KeyboardInterrupt:
            print("\n输入中断，程序已终止。")
            break

    return field_strengths, n_values


def plot_graph(H_values, delta_L1_L1_values):
    """
    绘制以 ΔL1/L1 × 10⁻⁶ 为纵坐标，H × 10³ (A/m) 为横坐标的图

    参数:
    H_values: list，场强 H 的值 (单位：A/m)
    delta_L1_L1_values: list，对应的 ΔL1/L1 的值 (无量纲)

    返回:
    None
    """
    # 转换 H 为 H × 10³
    H_scaled = [H * 1e-3 for H in H_values]

    # 转换 ΔL1/L1 为 ΔL1/L1 × 10⁻⁶
    delta_L1_L1_scaled = [value * 1e6 for value in delta_L1_L1_values]

    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.plot(H_scaled, delta_L1_L1_scaled, marker='o', label='ΔL1/L1 vs H')

    # 设置坐标轴标签和标题
    plt.xlabel("场强 H × 10³ (A/m)", fontsize=12)
    plt.ylabel("ΔL1/L1 × 10⁻⁶", fontsize=12)
    plt.title("磁致伸缩特性曲线", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()


# 用户输入

# 计算磁场


# 输出结果
#print(f"线圈内部的磁场强度为: {H_values:.6f} 特斯拉")
#delta_L1=n*meters
#delta_L1_L1_values=delta_L1/l1
#print("倒数第二列和最后一列的结果分别为")
#print(delta_L1,delta_L1_L1_values)
if __name__ == "__main__":
    # 用户输入
    print("磁致伸缩特性计算程序")
    N = 1200  # 匝数
    L = 0.07  # 线圈长度 (米)
    l1 = 0.11  # 原始长度 L1 (米)
    wavelength_nanometers = 632.8  # 激光波长 (单位：纳米)

    # 用户输入电流强度列表
    I,n = user_input()

    if len(I) >= 0:
        # 计算磁场强度
        H_values = calculate_magnetic_field(N, L, I)

        # 计算 ΔL1 和 ΔL1/L1
        meters = nanometers_to_meters(wavelength_nanometers) / 2
        delta_L1 = [h * meters for h in n]  # ΔL1 (单位：米)
        delta_L1_L1_values = [delta / l1 for delta in delta_L1]  # ΔL1/L1

        # 输出结果
        print(f"线圈的磁场强度 H 列表为 (单位：T): {H_values}")
        print(f"倒数第二列 (ΔL1) 的结果为 (单位：米): {delta_L1}")
        print(f"最后一列 (ΔL1/L1) 的结果为: {delta_L1_L1_values}")
        plot_graph(H_values, delta_L1_L1_values)
    else:
        print("未输入任何有效数据，程序结束。")

