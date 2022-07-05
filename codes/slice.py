import numpy as np


def fps(original, npoints):
    center_xyz = np.sum(original, 0)
    center_xyz = center_xyz / len(original)
    dist = np.sum((original - center_xyz) ** 2, 1)
    farthest = np.argmax(dist)
    distance = np.ones(len(original)) * 1e10
    target_index = np.zeros(npoints, dtype=np.int32)

    for i in range(npoints):
        target_index[i] = farthest
        target_point_xyz = original[target_index[i], :]

        dist = np.sum((original - target_point_xyz) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return original[target_index]

# 3 slices per axis， 40% points of one point cloud
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
