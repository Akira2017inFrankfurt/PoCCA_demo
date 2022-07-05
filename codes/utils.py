import os
import time
import open3d as o3d


# 可视化函数
# 最简单直接展示结果，不包含旋转 保存等功能
def visualize_samples(d_sample):
    """
    d_sample: numpy array [10000, 3]
    """
    print("Points in downsample set: ", len(d_sample))
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(d_sample)
    color = [102.0 / 255.0 ,111.0 / 255.0, 142.0 / 255.0]
    source.paint_uniform_color(color)
    o3d.visualization.draw_geometries([source])

    
def save_txt(d_sample):
    """
    d_sample: sampled point set
    可以根据运行的本地时间直接在当前文件夹下面创建并保存一个新txt文件，稍后根据自己需要重新命名
    """
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    now_txt = now + '.txt'
    current_file_path = os.getcwd()
    txt_file_name = os.path.join(current_file_path, now_txt)
    np.savetxt(txt_file_name, d_sample, fmt='%.6e', delimiter=',')
    print("Text Saved!")
    
    
def get_screen_filename():
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    now_png = now + '.png'
    current_file_path = os.getcwd()
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

    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    

# 功能复杂一些的可视化函数
# 显示出点云窗口之后，按r键旋转，s键保存，c键关闭窗口
def visualization(point_set):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_set)
    color = [102.0 / 255.0, 111.0 / 255.0, 142.0 / 255.0]
    source.paint_uniform_color(color)
    custom_draw_geometry_with_key_callback(source)
