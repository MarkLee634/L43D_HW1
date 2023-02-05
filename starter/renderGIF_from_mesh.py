"""
Sample code to render an img GIF from object

Usage:
    python -m starter.renderGIF_from_mesh --image_size 256 --output_path images/cow_render.gif
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
import numpy as np

from starter.utils import get_device, get_mesh_renderer, load_obj_mesh

def assign_color_from_vertices(vertices, color1, color2):
    textures = torch.ones_like(vertices)  # (N_v, 3)
    textures = textures * torch.tensor(color1)  # (N_v, 3)

    print(f"1. shape of color {torch.tensor(color1).shape }")

    z = vertices[:,2]
    z_min = torch.min(z)
    z_max = torch.max(z)

    alpha = (z-z_min)/(z_max-z_min)

    color1 = np.array(color1)
    color2 = np.array(color2)
    
    color_list = []
    for a in alpha:
        # print(f" from a {a}, color1 {color1}")
        color = a * color2 + (1-a) * color1
        color_list.append(color)
    

    print(f"2. vert shape {torch.ones_like(vertices).shape} ") 


    for texture_index in range(len(textures)):
        textures[texture_index] = textures[texture_index]  * torch.tensor(color_list[texture_index])
    
    print(f"4. shape of textures {textures.shape}")
    textures = textures.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    return textures


def render_GIF_from_obj(
    obj_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    print(f" device: {device}")
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_obj_mesh(obj_path)
    

    color1=[1.,0., 1.]
    color2=[0., 1., 0.]

    textures = assign_color_from_vertices(vertices, color1, color2)

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    # textures = torch.ones_like(vertices)  # (1, N_v, 3)
    # textures = textures * torch.tensor(color)  # (1, N_v, 3)

    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)


    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    NUM_VIEWS = 36
    # transform the camera around the obj 360 degrees
    azim = torch.linspace(-180, 180, NUM_VIEWS)

    #batch mesh for each view
    meshes = mesh.extend(NUM_VIEWS)

    # Prepare the camera:
    # specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3.0, azim=azim)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)

    # Render the obj.
    rend = renderer(meshes, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[:, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).

    return rend


if __name__ == "__main__":
    print(f" ------ initialzing script ------ ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_color_render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    images = render_GIF_from_obj(obj_path=args.obj_path, image_size=args.image_size)
    print(f" --- size of images: {len(images)} --- ")
    # Save the rendered images as GIF
    my_images = images  # List of images [(H, W, 3)]
    imageio.mimsave(args.output_path, my_images, fps=15)
    
    print(f" ------ ending script ------ ")
