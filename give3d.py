import os
import io
from plyfile import PlyData
import numpy as np
import yaml
import math

from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points

def sample_fps_points(data_root):
    #ply_path = os.path.join(data_root, 'model.ply')
    #ply_points = read_ply_points(ply_path)
    #fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    #np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)

    model_path = os.path.join(data_root, 'model.ply')
    renderer = OpenGLRenderer(model_path)

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    # xyz corners
    print("corner_3d:")
    print(corner_3d)
    # xyz center
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    print("center_3d:")
    print(center_3d)
    # xyz distances
    distances = np.max(corner_3d, 0) - np.min(corner_3d, 0)
    print("distances:")
    print(distances)
    # diameter is not correct. Its not used in EffieicentPose
    diameter = np.linalg.norm(np.max(corner_3d, 0) - np.min(corner_3d, 0))
    print("diameter:")
    print(diameter)

    # saving the data
    savedata = os.path.join(data_root, 'models_info.yml')
    export_model_para_yml(savedata,16,corner_3d,distances,diameter)

    print(corner_3d[0][0])


def export_model_para_yml(file_path, model_number, corner_3d, distances,diameter):
    model_paras = {
        model_number: {
            'diameter': diameter.tolist(),
            'min_x': corner_3d[0][0].tolist(),
            'min_y': corner_3d[0][1].tolist(),
            'min_z': corner_3d[0][2].tolist(),
            'size_x': distances[0].tolist(),
            'size_y': distances[1].tolist(),
            'size_z': distances[2].tolist(),
        }
    }

    with io.open(file_path, 'w', encoding='utf8') as outfile:
        yaml.dump(model_paras, outfile, default_flow_style=None)



sample_fps_points('data')

