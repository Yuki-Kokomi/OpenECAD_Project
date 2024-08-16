import math
import numpy as np


def rads_to_degs(rads):
    """Convert an angle from radians to degrees"""
    return 180 * rads / math.pi

def distance_of_point_and_plane(origin, x_axis, y_axis, point):
    # 确保输入是 numpy 数组
    origin = np.array(origin)
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    point = np.array(point)
    normal_vector = np.cross(x_axis, y_axis)
    distance = np.dot(point - origin, normal_vector)
    return np.abs(distance)

def is_point_on_plane(origin, x_axis, y_axis, point, tolerance=1e-10):
    """
    判断一个3D点是否在给定的平面上
    
    参数:
    origin - 平面的原点坐标，形状为 (3,)
    x_axis - 平面X轴的方向向量，形状为 (3,)
    y_axis - 平面Y轴的方向向量，形状为 (3,)
    point - 要检查的3D点，形状为 (3,)
    tolerance - 容忍度，默认值为 1e-10
    
    返回:
    如果点在平面上返回 True，否则返回 False
    """
    # 确保输入是 numpy 数组
    origin = np.array(origin)
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    point = np.array(point)
    
    # 计算法向量
    normal_vector = np.cross(x_axis, y_axis)
    
    # 计算点到平面的距离
    distance = np.dot(point - origin, normal_vector)
    
    # 判断距离是否在容忍度范围内
    return np.abs(distance) < tolerance


def number_to_pi_string(number):
    """
    将数字转换为带有 np.pi 的字符串表示形式
    
    参数:
    number - 输入数字，可以是 np.pi 的倍数
    
    返回:
    带有 np.pi 的字符串表示形式
    """
    # 定义一个容忍度来比较浮点数
    tolerance = 1e-10

    # 预定义一些常见的 π 的倍数及其对应的字符串表示
    pi_factors = {
        np.pi: 'np.pi',
        np.pi / 2: 'np.pi/2',
        np.pi / 3: 'np.pi/3',
        np.pi / 4: 'np.pi/4',
        np.pi / 6: 'np.pi/6',
        2 * np.pi: '2*np.pi',
        3 * np.pi / 2: '3*np.pi/2',
        3 * np.pi / 4: '3*np.pi/4',
        5 * np.pi / 6: '5*np.pi/6',
        5 * np.pi / 3: '5*np.pi/3',
        7 * np.pi / 6: '7*np.pi/6',
        4 * np.pi / 3: '4*np.pi/3',
    }
    
    # 检查输入数字是否接近这些常见的 π 倍数
    for key, value in pi_factors.items():
        if np.abs(np.abs(number) - key) < tolerance:
            return value if number > 0 else '-' + value
    
    # 如果数字不在预定义的 π 倍数中，返回原数字
    return str(number.round(6))

def are_parallel(v1, v2, tol=1e-10):
    # 计算叉积
    cross_product = np.cross(v1, v2)
    # 判断叉积是否接近于零向量
    return np.all(np.abs(cross_product) < tol)

def find_circle_center_and_radius(start_point, end_point, mid_point):    
    # Calculate midpoints of the chords
    mid_point_start_end = (start_point + end_point) / 2
    mid_point_start_mid = (start_point + mid_point) / 2
    
    # Calculate direction vectors of the chords
    direction_start_end = end_point - start_point
    direction_start_mid = mid_point - start_point
    
    # Calculate perpendicular direction vectors
    perp_start_end = np.array([-direction_start_end[1], direction_start_end[0]])
    perp_start_mid = np.array([-direction_start_mid[1], direction_start_mid[0]])
    
    # Solve for the intersection of the perpendicular bisectors
    A = np.array([perp_start_end, -perp_start_mid]).T
    b = mid_point_start_mid - mid_point_start_end
    
    # Solve the linear system
    t, s = np.linalg.solve(A, b)
    
    # Calculate the center
    center = mid_point_start_end + t * perp_start_end
    
    # Calculate the radius
    radius = np.linalg.norm(center - start_point)
    
    return center, radius

def rotate_vector(vector, axis, angle):
    """
    将一个3D向量绕指定轴逆时针旋转给定角度
    
    参数:
    vector - 要旋转的3D向量，形状为 (3,)
    axis - 旋转轴，形状为 (3,)
    angle - 旋转角度（弧度）

    返回:
    旋转后的3D向量，形状为 (3,)
    """
    # 确保输入是 numpy 数组
    vector = np.array(vector)
    axis = np.array(axis)
    
    # 计算单位轴向量
    axis = axis / np.linalg.norm(axis)
    
    # 计算旋转矩阵的各个分量
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_product = np.cross(axis, vector)
    dot_product = np.dot(axis, vector)
    
    # 计算旋转后的向量
    rotated_vector = (vector * cos_theta + 
                      cross_product * sin_theta + 
                      axis * dot_product * (1 - cos_theta))
    
    return rotated_vector

def calculate_rotation_angle(v1, v2, axis):
    """
    计算从向量 v1 到向量 v2 绕给定轴的逆时针旋转角度

    参数:
    v1 - 初始向量，形状为 (3,)
    v2 - 旋转后的向量，形状为 (3,)
    axis - 旋转轴，形状为 (3,)

    返回:
    旋转角度（弧度），范围 (-pi, pi]
    """
    # 确保输入是 numpy 数组
    v1 = np.array(v1)
    v2 = np.array(v2)
    axis = np.array(axis)
    
    # 计算单位轴向量
    axis = axis / np.linalg.norm(axis)
    
    # 计算点积
    dot_product = np.dot(v1, v2)
    
    # 计算叉积
    cross_product = np.cross(v1, v2)
    
    # 计算叉积在旋转轴上的投影长度
    projection_length = np.dot(cross_product, axis)
    
    # 计算向量的范数
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 计算角度的余弦值和正弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    sin_theta = projection_length / (norm_v1 * norm_v2)
    
    # 使用 arctan2 计算角度
    angle = np.arctan2(sin_theta, cos_theta)
    
    return angle

def map_2d_to_3d(origin, x_axis, y_axis, point):
    u, v = point
    return origin + u * x_axis + v * y_axis

def map_3d_to_2d(origin, x_axis, y_axis, point_3d):
    """
    将三维空间中的点转换为二维平面上的点
    
    参数:
    origin - 原点坐标 (Ox, Oy, Oz)，形状为 (3,)
    x_axis - X轴向量 (Xx, Xy, Xz)，形状为 (3,)
    y_axis - Y轴向量 (Yx, Yy, Yz)，形状为 (3,)
    point_3d - 三维空间中的点 (Px, Py, Pz)，形状为 (3,)

    返回:
    二维平面上的点 (u, v)，形状为 (2,)
    """
    # 确保输入是 numpy 数组
    origin = np.array(origin)
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    point_3d = np.array(point_3d)
    
    # 构建矩阵 A 和向量 b
    A = np.vstack([x_axis, y_axis]).T
    b = point_3d - origin
    
    # 求解线性方程组 Ax = b
    uv = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return uv


def unit_vector(vector):
    """
    计算给定向量的单位向量
    
    参数:
    vector - 输入向量，形状为 (n,)

    返回:
    单位向量，形状为 (n,)
    """
    # 计算向量的范数
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("零向量没有单位向量")
    # 计算单位向量
    unit_vector = vector / norm
    return unit_vector


def find_n_from_x_and_y(x, y):
    """
    Given vectors x and y, find a vector n such that y = n × x.
    Assumes that n is orthogonal to x.
    
    Parameters:
    x (numpy array): The vector x.
    y (numpy array): The vector y.
    
    Returns:
    numpy array: The vector n.
    """
    # Step 1: Compute the cross product of x and y to get n'
    n_prime = np.cross(x, y)
    
    # Step 2: Normalize n' to get the unit vector
    n_prime_unit = n_prime / np.linalg.norm(n_prime)
    
    # Step 3: Determine the correct sign of n_prime_unit
    # To ensure y = n × x, we should check if the direction is correct
    if np.allclose(np.cross(n_prime_unit, x), y):
        n = n_prime_unit
    else:
        n = -n_prime_unit
    
    return n

def angle_from_vector_to_x(vec):
    """computer the angle (0~2pi) between a unit vector and positive x axis"""
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle


def cartesian2polar(vec, with_radius=False):
    """convert a vector in cartesian coordinates to polar(spherical) coordinates"""
    vec = vec.round(6)
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2] / norm) # (0, pi)
    phi = np.arctan(vec[1] / (vec[0] + 1e-15)) # (-pi, pi) # FIXME: -0.0 cannot be identified here
    if not with_radius:
        return np.array([theta, phi])
    else:
        return np.array([theta, phi, norm])


def polar2cartesian(vec):
    """convert a vector in polar(spherical) coordinates to cartesian coordinates"""
    r = 1 if len(vec) == 2 else vec[2]
    theta, phi = vec[0], vec[1]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rotate_by_x(vec, theta):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_y(vec, theta):
    mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_z(vec, phi):
    mat = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    return np.dot(mat, vec)


def polar_parameterization(normal_3d, x_axis_3d):
    """represent a coordinate system by its rotation from the standard 3D coordinate system

    Args:
        normal_3d (np.array): unit vector for normal direction (z-axis)
        x_axis_3d (np.array): unit vector for x-axis

    Returns:
        theta, phi, gamma: axis-angle rotation 
    """
    normal_polar = cartesian2polar(normal_3d)
    theta = normal_polar[0]
    phi = normal_polar[1]

    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)

    gamma = np.arccos(np.dot(x_axis_3d, ref_x).round(6))
    if np.dot(np.cross(ref_x, x_axis_3d), normal_3d) < 0:
        gamma = -gamma
    return theta, phi, gamma


def polar_parameterization_inverse(theta, phi, gamma):
    """build a coordinate system by the given rotation from the standard 3D coordinate system"""
    normal_3d = polar2cartesian([theta, phi])
    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)
    ref_y = np.cross(normal_3d, ref_x)
    x_axis_3d = ref_x * np.cos(gamma) + ref_y * np.sin(gamma)
    return normal_3d, x_axis_3d
