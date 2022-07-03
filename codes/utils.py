import open3d as o3d


# 可视化函数
def visualize_samples(d_sample):
    print("Points in downsample set: ", len(d_sample))
    # d_sample = d_sample.numpy()
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(d_sample)
    color = [102.0 / 255.0 ,111.0 / 255.0, 142.0 / 255.0]
    source.paint_uniform_color(color)
    o3d.visualization.draw_geometries([source])
    
