import shutil
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import quaternion
from plyfile import PlyData, PlyElement
from tqdm.auto import tqdm

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

try:
    import mast3r.utils.path_to_dust3r  # noqa
except ImportError:
    pass

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


def compute_frustum(K, z_far=0.5):
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    width, height = px * 2, py * 2

    corners = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
    frustum = [[0.0, 0.0, 0.0, 1.0]]
    for x, y in corners:
        xc = (x - px) * z_far / fx
        yc = (y - py) * z_far / fy
        frustum.append([xc, yc, z_far, 1.0])

    return np.array(frustum, dtype=np.float32)


def export_gltf(intrinsics, extrinsics, points, colors, glb_file: Path):
    """
    Convert a PLY file to GLB format.
    """
    scene = trimesh.Scene()

    # Point cloud
    pcd = trimesh.PointCloud(points, colors=colors)
    scene.add_geometry([pcd], geom_name='Point cloud')

    scene_scale = np.max(np.std(points, axis=0))

    # Camera frustums
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 3, 2],
            [1, 4, 3],
        ],
        dtype=np.int32,
    )

    for idx, (K, P) in enumerate(zip(intrinsics, extrinsics)):
        v_view = compute_frustum(K, z_far=0.05 * scene_scale)
        v_world = (P @ v_view.T).T[:, :3]
        vertex_colors = np.array([255, 0, 0] * len(v_world), dtype=np.uint8).reshape((-1, 3))
        frustum = trimesh.Trimesh(vertices=v_world, faces=faces, vertex_colors=vertex_colors, process=False)
        scene.add_geometry([frustum], geom_name=f'Camera #{idx + 1:d}')

    # Save glTF (binary)
    scene.export(str(glb_file), file_type='glb')


def main(args: argparse.Namespace):
    dataset_path = Path(args.input)
    image_dir = dataset_path / 'images'
    out_dir = dataset_path / 'mast3r'
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_dir / 'sparse' / '0'
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logging.info(f'Using CUDA device: {torch.cuda.get_device_name(args.gpu)}')
    else:
        device = torch.device('cpu')
        logging.warning('CUDA is not available. Continue with CPU')

    model_name = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # load_images can take a list of images or a directory
    file_glob = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
    image_files = []
    for f in image_dir.iterdir():
        for e in file_glob:
            if str(f).endswith(e):
                image_files.append(f)
                break
    print(f'Loading images from {image_dir} ({len(image_files)} files)')

    filelist = [str(f) for f in image_files]
    images = load_images(filelist, size=args.resize)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    # sparse GA
    scene = sparse_global_alignment(filelist, pairs, str(cache_dir), model)

    # get optimized values from scene
    imgs = scene.imgs
    poses = scene.get_im_poses()
    focals = scene.get_focals()
    principal_points = scene.get_principal_points()
    poses = [t.detach().cpu().numpy() for t in poses]
    focals = [t.detach().cpu().numpy() for t in focals]
    principal_points = [t.detach().cpu().numpy() for t in principal_points]
    intrinsics = [
        np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]]) for focal, pp in zip(focals, principal_points)
    ]

    if args.tsdf_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=args.tsdf_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=args.clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=args.clean_depth))

    masks = to_numpy([c > args.min_conf_thr for c in confs])

    # Calculate image scaling factor
    org_image = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
    org_height, org_width = org_image.shape[:2]

    # Save COLMAP cameras
    cam_txt = sparse_dir / 'cameras.txt'
    with open(cam_txt, mode='w') as f:
        if args.single_camera:
            focal = np.mean(focals, axis=0)
            pp = np.mean(principal_points, axis=0)

            px = org_width * 0.5
            py = org_height * 0.5
            fx = px / pp[0] * focal
            fy = py / pp[1] * focal
            f.write(f'1 PINHOLE {org_width:d} {org_height:d} {fx:f} {fy:f} {px:f} {py:f}\n')
        else:
            for i in tqdm(range(len(focals)), desc='Saving COLMAP cameras'):
                focal = focals[i]
                pp = principal_points[i]

                px = org_width * 0.5
                py = org_height * 0.5
                fx = px / pp[0] * focal
                fy = py / pp[1] * focal

                # CAMERA_ID, TYPE, WIDTH, HEIGHT, FX, FY, PX, PY (for PINHOLE)
                f.write(f'{i + 1:d} PINHOLE {org_width:d} {org_height:d} {fx:f} {fy:f} {px:f} {py:f}\n')

    # Save COLMAP images
    img_txt = sparse_dir / 'images.txt'
    with open(img_txt, mode='w') as f:
        for i in tqdm(range(len(imgs)), desc='Saving COLMAP images'):
            pose = poses[i]
            tv = pose[:3, 3]
            qv = quaternion.from_rotation_matrix(pose[:3, :3])
            name = image_files[i].name
            cam_id = 1 if args.single_camera else i + 1
            f.write(
                f'{i + 1:d} {qv.w:f} {qv.x:f} {qv.y:f} {qv.z:f} {tv[0]:f} {tv[1]:f} {tv[2]:f} {cam_id:d} {name:s}\n'
            )
            f.write('\n')  # every 2nd line is for the point data, not needed for GSplats.

    # Save COLMAP points
    pts_txt = sparse_dir / 'points3D.txt'
    point_id = 1
    points = []
    colors = []
    with open(pts_txt, mode='w') as f:
        for i in tqdm(range(len(pts3d)), desc='Saving COLMAP points'):
            conf_i = masks[i]
            pts = pts3d[i][conf_i.ravel()]
            rgb = imgs[i][conf_i]
            rgb = (rgb * 255.0).astype(np.uint8)
            err = 0.0
            for j, p in enumerate(pts):
                # POINT_ID, X, Y, Z, R, G, B, ERR
                f.write(f'{point_id:d} {p[0]} {p[1]} {p[2]} {rgb[j, 0]:d} {rgb[j, 1]:d} {rgb[j, 2]:d} {err:f}\n')
                point_id += 1

            points.append(pts)
            colors.append(rgb)

    points = np.concatenate(points, axis=0).astype(np.float32)
    normals = np.zeros_like(points)
    colors = np.concatenate(colors, axis=0).astype(np.uint8)
    print(f'Total {len(points)} are detected!')

    # Save PLY
    point_data = [(x, y, z, nx, ny, nz, r, g, b) for (x, y, z), (nx, ny, nz), (r, g, b) in zip(points, normals, colors)]
    point_data = np.array(
        point_data,
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
        ],
    )
    elements = PlyElement.describe(point_data, 'vertex')
    ply_data = PlyData([elements], text=False, byte_order='<')
    ply_data.write(sparse_dir / 'points3D.ply')

    # Copy images
    out_image_dir = out_dir / 'images'
    out_image_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(image_files):
        shutil.copy(f, out_image_dir / f.name)

    # Save glTF file for sanity check
    export_gltf(intrinsics, poses, points, colors, sparse_dir / 'points3D.glb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-r', '--resize', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--min_conf_thr', type=float, default=2.0)
    parser.add_argument('--tsdf_thresh', type=float, default=0.0)
    parser.add_argument('--clean_depth', action='store_true', default=False)
    parser.add_argument('--single_camera', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
