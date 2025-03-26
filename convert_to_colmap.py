import os
from pathlib import Path
import numpy as np
import trimesh
import argparse
import pycolmap
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage
from typing import Tuple, List


def read_pose_file(pose_file) -> Tuple[List[pycolmap.Image], List[pycolmap.Camera]]:
    # assume all images are the same
    images = []
    cameras = []
    with open(pose_file, 'r') as f:
        for i, line in enumerate(f):
            tokens = line.split()
            img_path = tokens[0].split('/')[-6:]
            img_path = '../' + '/'.join(img_path)
            with PILImage.open(img_path) as pil_img:
                width, height = pil_img.size
            img_basename = os.path.basename(img_path)
            qw, qx, qy, qz = [float(t) for t in tokens[1:5]]
            tx, ty, tz = [float(t) for t in tokens[5:8]]
            focal_length = float(tokens[8])
            confidence = float(tokens[9])

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            T = np.eye(3, 4)
            T[:, :3] = R
            T[:, 3] = t

            camera = pycolmap.Camera(
                camera_id=i,
                model='SIMPLE_PINHOLE',
                width=width,
                height=height,
                params=[focal_length, width/2, height/2]
            )
            image = pycolmap.Image(
                image_id=i,
                camera_id=i,
                name=img_basename,
                cam_from_world=pycolmap.Rigid3d(T),
                registered=True if confidence >= 1000 else False
            )
            cameras.append(camera)
            images.append(image)

    return images, cameras

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir',
        type=str,
        default='../data/results/acezero/TanksAndTemples/Ignatius/acezero_format/',
        help='source directory'
    )
    parser.add_argument(
        '--dst_dir',
        default='../data/results/acezero/TanksAndTemples/Ignatius/colmap/sparse/0',
        type=str,
        help='destination directory'
    )

    src_dir=parser.parse_args().src_dir
    dst_dir=parser.parse_args().dst_dir

    pose_file=os.path.join(src_dir, 'poses_final.txt')
    pt_file=os.path.join(src_dir, 'pc_final.ply')

    images, cameras = read_pose_file(pose_file)

    reconstruction = pycolmap.Reconstruction()
    for image, camera in zip(images, cameras):
        reconstruction.add_camera(camera=camera)
        reconstruction.add_image(image=image)
    reconstruction.import_PLY(pt_file)

    # Save the reconstruction
    os.makedirs(dst_dir, exist_ok=True)
    reconstruction.write(dst_dir)