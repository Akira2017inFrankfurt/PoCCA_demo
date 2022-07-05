import numpy as np
from subsample_fps import fps


def get_random_center():
    """随机生成一个cube中心点"""
    u = np.random.uniform(-1.0, 1.0)
    theta = 2 * np.pi * np.random.uniform(0.0, 2)
    
    x = np.power((1 - u * u), 1/2) * np.cos(theta)
    x = np.abs(x)
    x = np.random.uniform(-x, x)
    y = np.power((1 - u * u), 1/2) * np.sin(theta)
    y = np.abs(y)
    y = np.random.uniform(-y, y)
    z = u
    return (x, y, z)


def point_in_cube(point_xyz, center_xyz, side_length):
    """判断一个点是否在cube内部"""
    flag = True
    for i in range(0, len(point_xyz)):
        if abs(point_xyz[i] - center_xyz[i]) >= (side_length / 2):
            flag = False
            break
    return flag


def get_1_cube(point_set, center_xyz, side_length, npoints=1024):
    """从单个点云中，根据1个中心点找到1个cube"""
    output_samples = []
    for i in range(0, len(point_set)):
        if point_in_cube(point_set[i], center_xyz, side_length):
            output_samples.append(i)
    samples = point_set[output_samples]
    
    if len(output_samples) > npoints:
        result = fps(samples, npoints)
        return result
    else:
        return get_1_cube(point_set, center_xyz, side_length + 0.2, npoints)
    
    
def get_cubes(point_set, num_cube=8, side_length=0.5, npoints=1024):
    """从单个点云中，根据8个中心点找到8个cube"""
    result = np.ones((num_cube, npoints, 3))
    centers = []  # record 8 cube centers
    for i in range(0, num_cube):
        center = get_random_center()
        centers.append(center)
        result[i] = get_1_cube(point_set, center, side_length, npoints)
    return result, centers
