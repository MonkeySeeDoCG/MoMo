# Base on https://github.com/EricGuo5513/text-to-motion

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import os
from moviepy.editor import clips_array


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if dataset == 'bvh_general':
        orange = "#DD5A37"
        colors = [orange] * len(kinematic_tree) 
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    n_frames = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        # sometimes index is equal to n_frames/fps due to floating point issues. in such case, we duplicate the last frame
        assert int(index*fps) <= n_frames
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_axis_off()
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    
    plt.close()
    return ani


def plot_3d_motion_features(kinematic_tree, joints, title, dataset, figsize=(5, 3), fps=120, radius=3,
                            vis_mode='default', gt_frames=[], feat_frames_pca=None, feat_joints_pca=None):
    matplotlib.use('Agg')
    assert(feat_joints_pca is None or feat_joints_pca.shape[-1] == joints.shape[-1])  # ensure same number of joints for features and locations
    assert(feat_frames_pca is None or feat_frames_pca.shape[0] == joints.shape[0])  # ensure same number of frames for features and locations

    # stretch pca values in the range [0,1]
    if feat_joints_pca is not None:
        feat_per_joint_colors = (feat_joints_pca - feat_joints_pca.min()) / (feat_joints_pca.max() - feat_joints_pca.min())
    if  feat_frames_pca is not None:
        feat_frames_colors = (feat_frames_pca - feat_frames_pca.min()) / (feat_frames_pca.max() - feat_frames_pca.min())

    title = '\n'.join(wrap(title, 40))
    n_frames = joints.shape[0]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

        # plot a bar of features per frame
        if feat_frames_pca is not None:
            ax2.bar(range(n_frames), height=1, color=feat_frames_colors, width=1, align='edge', edgecolor="none")                   
            ax2.set_xticks(range(0, n_frames, n_frames//5))
            # ax2.set_xlabel('time (frames)')
            ax2.yaxis.set_visible(False)
            ax2.axis('off')

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(20)
    plt.tight_layout()
    ax = fig.add_subplot(gs[3:-1], projection='3d')  # animation
    if feat_frames_pca is not None:
        ax2 = fig.add_subplot(gs[1])  # color bar    
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    gray = "#888888"
    colors_gray = [gray] * len(kinematic_tree)  # basic gray to plot features above
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue
    elif feat_joints_pca is not None:
        colors = colors_gray

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        index = int(index*fps)
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index in gt_frames:
            used_colors = colors_blue
        elif feat_frames_pca is not None:
            used_colors = np.tile(feat_frames_colors[index], (len(kinematic_tree), 1))
        else:
            used_colors = colors
        # used_colors = colors_blue if index in gt_frames else feat_frames_colors[index] if feat_frames_pca is not None else colors
        
        #  plot all edges according to kinematic chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #  plot all vertices according to deep features
        if feat_joints_pca is not None:
            joint_colors = feat_per_joint_colors[index] if feat_per_joint_colors.ndim == 3 else feat_per_joint_colors
            ax.scatter(xs=data[index,:,0], ys=data[index,:,1], zs=data[index,:,2], c=joint_colors, marker='o', s=81)
        
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # indicate time marker on colorbar
        if feat_frames_pca is not None:
            if index > 0:
               ax2.axvline(x=index-1, color=feat_frames_colors[index],  ymax=1)  # delete previous time marker
            ax2.axvline(x=index, color='black', ymax=1)  # draw time marker
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    
    plt.close()
    return ani


def plot_3d_motion_attention(attention_dict, kinematic_tree, joints, title, dataset, figsize=(5, 3), fps=120, radius=3, do_stretch=True):
    matplotlib.use('Agg')

    attn = attention_dict['map_per_frame']
    if do_stretch:
        attn = (attn - attn.min()) / (attn.max() - attn.min())  # stretch in the range [0, 1]
    else:
        assert attn.min() >= 0 and attn.max() <= 1  # attn has to be already in the range [0, 1]
    no_color = np.array([0.9, 0.9, 0.9])  # light gray
    full_color_name = list(mcolors.BASE_COLORS.keys())[attention_dict['color_idx']]
    full_color = np.array(mcolors.to_rgba(full_color_name)[:3])

    color_per_frame = np.concatenate([(((1.-e) * no_color) + (e * full_color))[None] for e in attn], axis=0)

    title = '\n'.join(wrap(title, 40))
    n_frames = joints.shape[0]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)  # , y=1+0.025*title.count('\n'))
        ax.grid(b=False)

        ax2.bar(range(n_frames), height=1, color=color_per_frame, width=1, align='edge', edgecolor="none")
        # ax2.set_xticks(range(0, n_frames, n_frames // 5))
        ax2.set_xticks([])
        # ax2.set_xlabel('time (frames)')
        ax2.set_xlabel(attention_dict['keyword'], fontweight='bold')
        ax2.yaxis.set_visible(False)
        ax2.xaxis.label.set_color(full_color_name)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        # ax2.axis('off')
        # ax2.tick_params(bottom=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5  # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(20)
    plt.tight_layout()
    # ax = fig.add_subplot(2, 1, 2, projection='3d')
    # ax2 = fig.add_subplot(2, 1, 1)
    ax = fig.add_subplot(gs[3:-1], projection='3d')  # animation
    ax2 = fig.add_subplot(gs[1])  # color bar
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        index = int(index*fps)
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        used_colors = np.tile(color_per_frame[index], (len(kinematic_tree), 1))
        # used_colors = colors_blue if index in gt_frames else feat_frames_colors[index] if feat_all_joints_pca is not None else colors

        #  plot all edges according to kinematic chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #  plot all vertices according to deep features

        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # indicate time marker on colorbar
        if index > 0:
            ax2.axvline(x=index - 1, color=color_per_frame[index], ymax=1)  # delete previous time marker
        ax2.axvline(x=index, color='black', ymax=1)  # draw time marker
        return mplfig_to_npimage(fig)


    ani = VideoClip(update)
   
    plt.close()
    return ani


def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, n_rows_in_out_file=3, *args, **kwargs):
    
    n_rows_in_data = len(animations)
    prefix = kwargs.get('prefix', '')
    os.makedirs(out_path, exist_ok=True)
    
    for sample_i in range(0, n_rows_in_data, n_rows_in_out_file):
        last_sample_i = min(sample_i+n_rows_in_out_file, n_rows_in_data)
        if file_templates is not None:
            all_sample_save_file = prefix + file_templates['all'].format(sample_i, last_sample_i-1, *args)
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}', flush=True)
        else: 
            all_sample_save_path = out_path
            print(f'saving {all_sample_save_path}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important

