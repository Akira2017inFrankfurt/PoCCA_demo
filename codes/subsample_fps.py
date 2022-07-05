import numpy as np

def fps(original, npoints):
    """
    original: numpy array [10000, 3]
    npoints: output point number
    return: downsampled point set [npoints, 3]
    """
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
