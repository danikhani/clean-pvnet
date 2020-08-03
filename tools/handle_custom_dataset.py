import os
from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from lib.utils import base_utils
import json


def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


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


def record_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

# hier sind Änderungen der Pfade notwendig, um die richtigen Bilder rauszusuchen
   # pose_dir = os.path.join(data_root, 'pose')
   # rgb_dir = os.path.join(data_root, 'rgb')
   # mask_dir = os.path.join(data_root, 'mask')

    #inds = range(len(os.listdir(data_root)))
    inds = range(99)


    for ind in tqdm.tqdm(inds):
        # rgb_path = os.path.join(rgb_dir, '{}.jpg'.format(ind))
        klasse = 'BC283R_CPA_000' # Name der Objektklasse, für welche trainiert werden soll

        number = str(ind) 
        number = number.zfill(6)
        datei = number + '.png'
        rgb_path = os.path.join(data_root, datei)

        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        # pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        # hier muss die Pose aus dem .json abgegriffen werden
        datei = number + '.json'
        pose_path = os.path.join(data_root, datei)

        # hier muss die Pose (R,t) aus den Annotationen richtig ausgelesen werden
        # pose = np.load(pose_path)

        #annotation = json.load(pose_path) 

        with open(pose_path,'r') as file:
            annotation = json.loads(file.read())

        objekt = annotation['objects']
        objekt_klasse = objekt[0]["class"]

        if klasse in objekt_klasse:
            koordinaten = objekt[0]["pose_transform"]
            pose = np.array(koordinaten)
            pose = pose[:,0:3]
            pose = pose.transpose()
        else:
            pass 

        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]
        fps_2d = base_utils.project(fps_3d, K, pose)

        # mask_path = os.path.join(mask_dir, '{}.png'.format(ind))
        # hier muss die Segmentierungs-Maske (Instanz) aus dem erzeugten png abgegriffen werden
        mask_path = os.path.join(data_root, '{}.is.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})

        # rgb_dir existiert nicht mehr, hier wird mit data_root gearbeitet
        anno.update({'data_root': data_root})

        # anstatt der Klasse "cat" wird hier die Klasse "Instrument" erkannt
        anno.update({'type': 'real', 'cls': 'BC283R_CPA_000'})
        annotations.append(anno)

    return img_id, ann_id


def custom_to_coco(data_root):
    model_path = os.path.join(data_root, 'model.ply')

    renderer = OpenGLRenderer(model_path)
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))

    model_meta = {
        'K': K,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations)
    # auch hier wird "cat" zu "BC283R_CPA_000"
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'BC283R_CPA_000'}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)
