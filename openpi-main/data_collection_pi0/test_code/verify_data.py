import h5py

import cv2

import numpy as np

import argparse

import sys



def view_h5_file(file_path):

    """

    读取并显示H5文件中的三个相机RGB视频流。



    Args:

        file_path (str): H5文件的路径。

    """

    try:

        f = h5py.File(file_path, 'r')

    except IOError:

        print(f"❌ 错误: 无法打开文件 '{file_path}'。请检查路径是否正确。")

        return



    print("✅ H5文件加载成功!")

    print("窗口操作: 按 'q' 键退出, 按 'p' 键暂停/继续。")



    # 定义相机ID和对应的窗口名称

    camera_ids = {

        'camera_high': 'High Camera',

        'camera_left_wrist': 'Left Wrist Camera',

        'camera_right_wrist': 'Right Wrist Camera'

    }



    camera_data = {}

    frame_counts = {}



    # --- 1. 加载所有相机数据 ---

    print("正在加载相机数据...")

    try:

        for cam_id in camera_ids.keys():

            # 根据 hdf5_storage.py 中的结构读取数据

            path_to_color = f'observations/cameras/{cam_id}/color'

            if path_to_color in f:

                camera_data[cam_id] = f[path_to_color]

                frame_counts[cam_id] = len(camera_data[cam_id])

                print(f"  - 找到 '{cam_id}': {frame_counts[cam_id]} 帧")

            else:

                print(f"  - ⚠️  警告: 在文件中未找到 '{cam_id}' 的数据。")

    except Exception as e:

        print(f"❌ 错误: 读取H5文件中的数据集时出错: {e}")

        f.close()

        return



    if not camera_data:

        print("❌ 错误: 文件中没有找到任何有效的相机数据。")

        f.close()

        return

        

    # --- 2. 准备播放 ---

    # 获取最长的视频帧数作为播放基准

    max_frames = max(frame_counts.values()) if frame_counts else 0

    current_frame_index = 0

    is_paused = False



    # --- 3. 循环播放帧 ---

    while current_frame_index < max_frames:

        if not is_paused:

            # 显示每个相机的当前帧

            for cam_id, window_name in camera_ids.items():

                if cam_id in camera_data:

                    # 确保索引不超过当前视频的长度

                    if current_frame_index < frame_counts[cam_id]:

                        # HDF5数据集返回的是BGR格式，OpenCV可以直接显示

                        frame = camera_data[cam_id][current_frame_index]

                        cv2.imshow(window_name, frame)

            

            current_frame_index += 1



        # --- 4. 用户交互 ---

        key = cv2.waitKey(33) & 0xFF  # 大约30fps的播放速度 (1000/30 ≈ 33ms)



        if key == ord('q'):  # 按 'q' 退出

            break

        elif key == ord('p'):  # 按 'p' 暂停/继续

            is_paused = not is_paused

            print("▶️ 暂停" if is_paused else "✅ 继续")



    # --- 5. 清理资源 ---

    print("清理资源并关闭。")

    f.close()

    cv2.destroyAllWindows()





if __name__ == "__main__":

    # 使用 argparse 来方便地从命令行接收文件路径

    parser = argparse.ArgumentParser(description="查看H5文件中记录的三个相机RGB视野。")

    parser.add_argument("h5_file", type=str, default="20251022_185401_episode.h5", help="要查看的HDF5文件路径。")

    

    # 检查是否提供了参数

    if len(sys.argv) < 2:

        parser.print_help()

        sys.exit(1)

        

    args = parser.parse_args()

    

    view_h5_file(args.h5_file)