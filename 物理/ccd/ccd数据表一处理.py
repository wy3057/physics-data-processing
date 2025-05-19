# Python程序实现公式计算

def calculate_a(D, wavelength, x_minus_1, x_plus_1, delta):
    """
    计算单缝宽度 a.
    参数:
    D: 距离
    wavelength: 光的波长
    x_minus_1, x_plus_1: -1级和+1级光强点的横坐标
    delta: 像素间距

    返回:
    a: 单缝宽度
    """
    return (2 * D * wavelength) / (abs(x_plus_1 - x_minus_1) * delta)


def calculate_error(a_measured, a_theoretical):
    """
    计算百分误差 ε.
    参数:
    a_measured: 测量值
    a_theoretical: 理论值

    返回:
    ε: 百分误差
    """
    return abs(a_measured - a_theoretical) / a_theoretical * 100

def mm_to_m(mm):
    """
    将毫米 (mm) 转换为米 (m)。

    参数:
        mm (float): 毫米值

    返回:
        float: 转换后的米值
    """
    if not isinstance(mm, (int, float)):
        raise ValueError("输入值必须是数字")
    return mm / 1000

def um_to_m(um):
    """
    将微米 (μm) 转换为米 (m)。

    参数:
        um (float): 微米值

    返回:
        float: 转换后的米值
    """
    if not isinstance(um, (int, float)):
        raise ValueError("输入值必须是数字")
    return um / 1_000_000  # 1 米 = 1,000,000 微米

# 示例输入
D = mm_to_m(float(input("请输入距离 D (单位: m): ")) - 61.5) # 距离 (单位: m)
wavelength = 650e-9  # 波长 (单位: m, 500 nm)
x_minus_1 = int(input("请输入-1级光强点的横坐标: "))  # -1级光强点位置 (单位: m)
x_plus_1 = int(input("请输入+1级光强点的横坐标: "))  # +1级光强点位置 (单位: m)
delta = um_to_m(int(11))  # 像素间距 (单位: m)


a_theoretical = mm_to_m(float(0.07))# 理论单缝宽度 (单位: m)

# 计算结果
a = calculate_a(D, wavelength, x_minus_1, x_plus_1, delta)
a_measured = a  # 测量单缝宽度 (单位: m)
error = calculate_error(a_measured, a_theoretical)
print(f"计算百分误差 ε: {error:.9f} %")
print(f"计算单缝宽度 a: {a:.9f} m")


