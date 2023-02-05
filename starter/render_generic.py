"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import sys
import imageio


from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image, gif_from_pcloud


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_plant(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
    output_path="images/plant_render.gif",
):
    """
    Renders a point cloud from 2 rgbd imgs.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    rgbd_data = load_rgbd_data()
    print(f"rgbd_data.keys(): {rgbd_data.keys()}")
    print(f" shape of rgb1 {rgbd_data['rgb1'].shape}, depth1 {rgbd_data['depth1'].shape}, rgb2 {rgbd_data['rgb2'].shape}, depth2 {rgbd_data['depth2'].shape}")

    
    vert1, rgb1 = unproject_depth_image (torch.Tensor(rgbd_data['rgb1']), 
        torch.Tensor(rgbd_data['mask1']), torch.Tensor(rgbd_data['depth1']), rgbd_data['cameras1'])

    vert2, rgb2 = unproject_depth_image (torch.Tensor(rgbd_data['rgb2']), 
        torch.Tensor(rgbd_data['mask2']), torch.Tensor(rgbd_data['depth2']), rgbd_data['cameras2'])


    vert1 = vert1.to(device).unsqueeze(0)
    rgb1 = rgb1.to(device).unsqueeze(0)
    vert2 = vert2.to(device).unsqueeze(0)
    rgb2 = rgb2.to(device).unsqueeze(0)

    vert_union = torch.cat((vert1, vert2), dim=1)
    rgb_union = torch.cat((rgb1, rgb2), dim=1)

    # (vert, faces, renderer, dist, NUM_VIEW, device):
    dist = 6.
    NUM_VIEW = 36
    rendered_imgs = gif_from_pcloud(vert_union, rgb_union, renderer, dist ,NUM_VIEW, device)
    # print(f"shape of rendered_imgs {rendered_imgs.shape}")

    imageio.mimsave(output_path, rendered_imgs, fps=15)



    
    point_cloud1 = pytorch3d.structures.Pointclouds(points=vert1, features=rgb1)

    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud1, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    print(f"shape of pcloud {point_cloud1.points_packed().shape}, rgb {point_cloud1.features_packed().shape}")

    

    return rend


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)

    # print(f" shape of verts {verts.shape}, rgb {rgb.shape}")
    
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    

    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        # image = render_bridge(image_size=args.image_size)
        image = render_plant(image_size=args.image_size, output_path=args.output_path)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    # plt.imsave(args.output_path, image)
    plt.imshow(image)
    plt.show()

