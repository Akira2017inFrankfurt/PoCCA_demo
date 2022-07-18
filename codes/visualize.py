import numpy as np
import open3d as o3d
import os
import time
from network.augmentation import PointWOLF
from utils.crops import fps
import random
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# crop method 1: slice
def get_index(index_slice, ration_per_slice, overlap_ration, num_all_points):
    # index_slice = 0, 1, 2
    start_index = index_slice * (ration_per_slice - overlap_ration) * num_all_points
    end_index = start_index + ration_per_slice * num_all_points
    return int(start_index), int(end_index)


def get_slice(point_set, npoints, xyz_dim, index_slice):
    # xyz_dim: 0, 1, 2 for x, y, z
    axis = 'x'
    if xyz_dim == 1:
        axis = 'y'
    elif xyz_dim == 2:
        axis = 'z'
    print(f"这个是沿着{axis}轴来切割的，第{index_slice}个slice.")

    start_index, end_index = get_index(index_slice, 0.4, 0.1, len(point_set))
    big_patch_index = np.argsort(point_set, axis=0)[start_index: end_index, xyz_dim]
    big_patch = point_set[big_patch_index]
    random.shuffle(big_patch)
    # 返回fps后的值，fps前的值
    return (fps(big_patch, npoints), big_patch)


# crop method 2: cube
def get_random_center():
    """随机生成一个cube中心点"""
    u = np.random.uniform(-1.0, 1.0)
    theta = 2 * np.pi * np.random.uniform(0.0, 2)

    x = np.power((1 - u * u), 1 / 2) * np.cos(theta)
    x = np.abs(x)
    x = np.random.uniform(-x, x)
    y = np.power((1 - u * u), 1 / 2) * np.sin(theta)
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


# crop method 3: knn
import torch


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def b_FPS(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids, index_points(xyz, centroids)


def k_points(a, b, k):
    # a: small, b: big one
    inner = -2 * torch.matmul(a, b.transpose(2, 1))
    aa = torch.sum(a ** 2, dim=2, keepdim=True)
    bb = torch.sum(b ** 2, dim=2, keepdim=True)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def new_k_patch(x, k=2048, n_patch=8, n_points=1024):
    """k = 2048, with FPS subsample method and FPS centroids"""
    patch_centers_index, _ = b_FPS(x, n_patch)  # torch.Size([B, n_patch])
    center_point_xyz = index_points(x, patch_centers_index)  # [B, n_patch]

    idx = k_points(center_point_xyz, x, k)  # B, k, 2048
    idx = idx.permute(0, 2, 1)  # B, k, n_patch

    new_patch = torch.zeros([n_patch, x.shape[0], n_points, x.shape[-1]]).to(device)
    for i in range(n_patch):
        patch_idx = idx[:, :, i].reshape(x.shape[0], -1)
        _, patch_points = b_FPS(index_points(x, patch_idx), n_points)
        new_patch[i] = patch_points
    new_patch = new_patch.permute(1, 0, 2, 3)  # torch.Size([B, 8, 1024, 3])
    return new_patch


def new_k_patch_1024(x, k=1024, n_patch=8, n_points=1024):
    """k=1024 without subsample method but FPS centroids"""
    patch_centers_index, _ = b_FPS(x, n_patch)  # torch.Size([B, n_patch])
    center_point_xyz = index_points(x, patch_centers_index)  # [B, n_patch]

    idx = k_points(center_point_xyz, x, k)  # B, n_patch, 1024
    idx = idx.permute(0, 2, 1)  # B, k, n_patch

    new_patch = torch.zeros([n_patch, x.shape[0], n_points, x.shape[-1]]).to(device)
    for i in range(n_patch):
        patch_idx = idx[:, :, i].reshape(x.shape[0], -1)
        patch_points = index_points(x, patch_idx)
        new_patch[i] = patch_points
    new_patch = new_patch.permute(1, 0, 2, 3)  # torch.Size([B, 8, 1024, 3])
    return new_patch


def random_k_patch_1024(x, k=1024, n_patch=8, n_points=1024):
    """k=1024 with random centroids"""

    def get_random_index(point_set, num):
        result = []
        for i in range(point_set.shape[0]):
            result.append(random.sample(range(0, point_set.shape[1]), num))
        return torch.tensor(result, dtype=torch.int64).to(device)

    patch_centers_index = get_random_index(x, n_patch)  # torch.Size([B, n_patch])
    center_point_xyz = index_points(x, patch_centers_index)  # [B, n_patch]

    idx = k_points(center_point_xyz, x, k)  # B, n_patch, 1024
    idx = idx.permute(0, 2, 1)  # B, k, n_patch

    new_patch = torch.zeros([n_patch, x.shape[0], n_points, x.shape[-1]]).to(device)
    for i in range(n_patch):
        patch_idx = idx[:, :, i].reshape(x.shape[0], -1)
        patch_points = index_points(x, patch_idx)
        new_patch[i] = patch_points
    new_patch = new_patch.permute(1, 0, 2, 3)  # torch.Size([B, 8, 1024, 3])
    return new_patch

def get_screen_filename():
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    now_png = now + '.png'
    current_file_path = "/home/haruki/桌面/demo/knn_1024"
    png_file_name = os.path.join(current_file_path, now_png)
    return png_file_name


def custom_draw_geometry_with_key_callback(pcd):
    def close_window(vis):
        vis.close()
        return False

    def rotate_view(vis):
        ctr = vis.get_view_control()
        # every step how much to rotate
        # x -800
        # y 100
        ctr.rotate(-800.0, 100.0)
        return False

    def save_capture_screen_image(vis):
        vis.capture_screen_image(get_screen_filename())
        return False

    key_to_callback = {}
    key_to_callback[ord("C")] = close_window
    key_to_callback[ord("R")] = rotate_view
    key_to_callback[ord("S")] = save_capture_screen_image

    o3d.visualization.draw_geometries_with_key_callbacks([pcd],
                                                         key_to_callback,
                                                         width=800,
                                                         height=600,
                                                         left=10,
                                                         top=10)


def visualization(point_set):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_set)
    color = [102.0 / 255.0, 111.0 / 255.0, 142.0 / 255.0]
    # source.paint_uniform_color(color)
    custom_draw_geometry_with_key_callback(source)


# 展示原始点云
def vis(number):
    root = r'/home/haruki/下载/modelnet40_normal_resampled/airplane/airplane_000'
    new_root = root + str(number) + '.txt'
    # 163, for chair best
    point_set = np.loadtxt(new_root, delimiter=',').astype(np.float32)[:, 0:3]
    visualization(point_set)


def run_10_times(num, point_set):
    w_sigma = 0.9
    num_anchor = 4
    sample_type = 'fps'
    for i in range(num):
        morph_func = PointWOLF(w_sigma, num_anchor, sample_type)
        _, morph_point_set = morph_func(point_set)
        visualization(morph_point_set)


if __name__ == "__main__":
    # 0.8 nice for point wolf
    root = r'/home/haruki/下载/modelnet40_normal_resampled/chair/chair_0163.txt'
    point_set = np.loadtxt(root, delimiter=',').astype(np.float32)[:, 0:3]
    # point wolf augmentation
    # run_10_times(10, point_set)

    # slice
    # after_fps_slice, direct_slice = get_slice(point_set, 1024, 0, 1)
    # visualization(after_fps_slice)

    # cube
    # cubes, centers = get_cubes(point_set)
    # for i in range(8):
    #     chose_1_cube = cubes[i, :, :]
    #     visualization(chose_1_cube)

    # knn
    # tensor_point_set = torch.tensor(point_set[np.newaxis, :, :]).to(device)
    #
    # knn_2048_fps_centroids = new_k_patch(tensor_point_set)
    # knn_1024_fps_centroids = new_k_patch_1024(tensor_point_set)
    # knn_1024_random_centroids = new_k_patch_1024(tensor_point_set)
    #
    # knn_2048_fps_centroids = knn_2048_fps_centroids.reshape(8, 1024, 3)
    # knn_1024_fps_centroids = knn_1024_fps_centroids.reshape(8, 1024, 3)
    # k1 = knn_2048_fps_centroids.cpu().numpy()
    # k2 = knn_1024_fps_centroids.cpu().numpy()
    #
    # for i in range(8):
    #     demo = k2[i]
    #     visualization(demo)