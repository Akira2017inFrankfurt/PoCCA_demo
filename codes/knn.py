import random
import torch
import numpy as np


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
    distance = torch.ones(B, N).to(device)* 1e10
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
    aa = torch.sum(a**2, dim=2, keepdim=True)
    bb = torch.sum(b**2, dim=2, keepdim=True)
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
    new_patch = new_patch.permute(1, 0, 2, 3)   # torch.Size([B, 8, 1024, 3])
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
    new_patch = new_patch.permute(1, 0, 2, 3)   # torch.Size([B, 8, 1024, 3])
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

    new_patch = torch.zeros([n_patch, x.shape[0], n_points, x.shape[-1]]).cuda()
    for i in range(n_patch):
        patch_idx = idx[:, :, i].reshape(x.shape[0], -1)
        patch_points = index_points(x, patch_idx)
        new_patch[i] = patch_points
    new_patch = new_patch.permute(1, 0, 2, 3)   # torch.Size([B, 8, 1024, 3])
    return new_patch
