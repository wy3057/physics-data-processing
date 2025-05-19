def calculate_v0(v0min, v_minus_1, v_plus_1):
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
    qw = 0.5*(v_minus_1-v_plus_1)
    return float(v0min-qw)

v0min = int(input("请输入v0明："))
v_minus_1= int(input("请输入v-1："))
v_plus_1= int(input("请输入v+1："))
print(calculate_v0(v0min, v_minus_1, v_plus_1))