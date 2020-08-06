import os
from plyfile import PlyData
import numpy as np
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import tqdm
from PIL import Image
from lib.utils import base_utils
import json
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv




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


def record_ann(model_meta, img_id, ann_id, images, annotations, cls_type):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    #inds = range(len(os.listdir(data_root)))
    length = int(input("Bildanzahl? "))
    #length = 10
    inds = range(length)

    for ind in tqdm.tqdm(inds):
        klasse = cls_type # Name der Objektklasse, für welche trainiert werden soll

        number = str(ind) 
        number = number.zfill(6)
        datei = number + '.png'
        rgb_path = os.path.join(data_root, datei)

        rgb = Image.open(rgb_path).convert('RGB')
        img_size = rgb.size
        img_id += 1
        #info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        info = {'file_name': rgb_path, 'height': 512, 'width': 512, 'id': img_id}
        images.append(info)

        # hier muss die Pose aus dem .json abgegriffen werden
        datei = number + '.json'
        pose_path = os.path.join(data_root, datei)


        # hier muss die Pose (R,t) aus den Annotationen richtig ausgelesen werden

        with open(pose_path,'r') as file:
            annotation = json.loads(file.read())
        
        objekt = annotation['objects']
        objekt_klasse = objekt[0]["class"]
        #quaternion_cam2world = R.from_quat(np.array(annotation['camera_data']['quaternion_xyzw_worldframe']))
       
        if klasse in objekt_klasse:

            translation = np.array(objekt[0]['location']) * 10
            #translation[0] = translation[0]*(-1)
            #translation[1] = translation[1]*(-1)

            #quaternion_obj2cam = R.from_quat(np.array(objekt[0]['quaternion_xyzw']))
            #quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
            #mirrored_y_axis = np.dot(quaternion_obj2cam.as_matrix(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
            
            pose = np.array(objekt[0]['pose_transform'])
            rotation = pose[0:3,0:3]
            #rotation = rotation.transpose()

            # Drehung um 270° um die Y-Achse
            #rotation = np.dot(rotation, np.array([[0,0,-1],[0,1,0],[1,0,0]]))

            # Drehung um 180° um die Z-Achse --> von links auf rechtshand-koordiantensystem
            rotation = np.dot(rotation, np.array([[-1,0,0],[0,-1,0],[0,0,1]]))

            #mirrored_y_axis = np.dot(rotation, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
          
            pose = np.column_stack((rotation, translation))

            #corner_2d = np.array(objekt[0]['projected_cuboid'])
            #center_2d = np.array(objekt[0]['projected_cuboid_centroid'])
            print(pose)
        else:
            print("Klasse in Annotation nicht enthalten!")
            pass 

        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]
        print("corner_2d")
        print(corner_2d)
        print("center_2d")
        print(center_2d)
        fps_2d = base_utils.project(fps_3d, K, pose)

        # mask_path = os.path.join(mask_dir, '{}.png'.format(ind))
        # hier muss die Segmentierungs-Maske (Instanz) aus dem erzeugten png abgegriffen werden
        datei = number + '.cs.png'
        mask_path = os.path.join(data_root, datei)

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})

        # rgb_dir existiert nicht mehr, hier wird mit data_root gearbeitet
        anno.update({'data_root': data_root})

        # anstatt der Klasse "cat" wird hier die Klasse durch die Eingabe übergeben
        anno.update({'type': 'real', 'cls': cls_type})
        annotations.append(anno)

    return img_id, ann_id


def custom_to_coco(data_root):
    model_path = os.path.join(data_root, 'model.ply')

    #cls_type = input("Klassenname? ")
    cls_type = 'BC283R_CPA_000'
    renderer = OpenGLRenderer(model_path)
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model) 
    print("corner_3d:")
    print(corner_3d)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    print("center_3d:")
    print(center_3d)
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

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations, cls_type)
    # auch hier wird "cat" zu cls_type
    categories = [{'supercategory': 'none', 'id': 1, 'name': cls_type}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)
