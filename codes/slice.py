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


# patch（3 slices， 40% per slice）
def get_slice_index(index_slice, ratio_per_slice, overlap_ratio, num_all_points):
    # index_slice = 0, 1, 2
    start_index = index_slice * (ratio_per_slice - overlap_ratio) * num_all_points
    end_index = start_index + ratio_per_slice * num_all_points
    return int(start_index), int(end_index)


def get_slice(point_set, xyz_dim, index_slice, npoints):
    # xyz_dim: 0, 1, 2 for x, y, z
    start_index, end_index = get_slice_index(index_slice, 0.4, 0.1, len(point_set))
    patch_index = np.argsort(point_set, axis=0)[start_index: end_index, xyz_dim]
    patch = point_set[patch_index]
    random.shuffle(patch)
    if len(patch_index) > npoints:
        patch = fps(patch, npoints)
    return patch
