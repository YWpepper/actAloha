import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed




JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


'''
output: 返回读取的 qpos(位置)、qvel(速度)、action(动作)和image_dict(图像)
'''

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

'''
funt: 将图像数据保存为视频文件
content: 一个是包含多个相机图像的字典，另一个是一个包含相机图像字典的列表。
         函数将图像帧按照时间顺序合并，并将它们保存为视频文件。
'''
def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        cam_names = sorted(cam_names)
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        cam_names = sorted(cam_names)
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])

        # 使用np.stack函数来将列表中的数组沿着新的轴连接起来
        # 创建一个三维数组，其中n_frames表示帧数，n和m表示每帧的形状 
        # all_cam_videos = np.stack(all_cam_videos, axis=0)  # shape: (n_frames, n, m)
        # ## check the shape     
        # print([video.shape for video in all_cam_videos])

        all_cam_videos = np.concatenate(all_cam_videos, axis=0)  # width dimension
        all_cam_videos = np.stack(all_cam_videos, axis=0)
        print(all_cam_videos.shape)
        n_frames, h, w = all_cam_videos.shape
        fps = int(1 / dt)
        output_size = (w // 2, h // 2)  # 缩小视频尺寸为原来的四分之一
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, output_size)
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = cv2.resize(image, output_size)  # 调整帧的尺寸
            # 交换图像的蓝色通道（B）和红色通道（R）
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


'''
func:     可视化关节状态和命令 
cnotent:  接受关节位置列表qpos_list和动作列表command_list作为输入,
          并绘制每个关节的状态和命令曲线。
'''
def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()



'''
func:     可视化时间戳信息
content:  接受时间戳列表t_list作为输入,并绘制相机帧的时间戳以及时间间隔.
'''
def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()



'''
func: 解析命令行参数,并根据指定的数据集目录和索引加载HDF5数据集
'''
def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    ismirror = args['ismirror']
    if ismirror:
        dataset_name = f'mirror_episode_{episode_idx}'
    else:
        dataset_name = f'episode_{episode_idx}'

    # 获取 hdf5 数据集中的关于 位置、速度、动作、图像 ， 用于后期的可视化
    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    # 调用save_videos函数将图像数据保存为视频文件
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    # 可视化关节状态和命令
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back




if __name__ == '__main__':
    default_dataset_dir = './dataset'
    default_episode_idx = '8'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', default=default_dataset_dir)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=default_episode_idx)
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))
