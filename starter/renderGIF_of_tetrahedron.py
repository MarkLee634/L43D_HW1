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

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from starter.utils import get_device, get_mesh_renderer, load_obj_mesh


def get_tetrahedron_vertices_faces():
    # Manually create the vertices, faces
    vertices = torch.tensor([[0,0,1], [1.5,0,0], [-1,0,-1], [1,2,0]], dtype=torch.float32)
    faces = torch.tensor([[0,1,3], [1,2,3], [0,2,3], [0,1,2]], dtype=torch.int64)

    return vertices, faces

def get_cube_vertices_faces():
    # Manually create the vertices, faces
    vertices = torch.tensor([[-1,0,1], [1,0,1], [1,0,-1], [-1,0,-1], [-1,1,1], [1,1,1], [1,1,-1], [-1,1,-1]], dtype=torch.float32)
    faces = torch.tensor([[0,1,3], [1,2,3], [0,4,5], [0,5,1], [5,6,1], [1,2,6], [6,7,2], [2,3,7], [3,7,0], [0,3,4], [4,5,6], [6,7,4]], dtype=torch.int64)

    return vertices, faces

def render_GIF_from_tetrahedron(
     image_size=256, color=[0.2, 0.7, 0.3], device=None,
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

    # vertices, faces = get_tetrahedron_vertices_faces()
    vertices, faces = get_cube_vertices_faces()

   

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # # Render the plotly figure
    # fig = plot_scene({
    #     "subplot1": {
    #         "tetra_mesh": mesh
    #     }
    # })
    # fig.show()


    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -10]], device=device)

    NUM_VIEWS = 36
    # transform the camera around the obj 360 degrees
    azim = torch.linspace(-180, 180, NUM_VIEWS)

    #batch mesh for each view
    meshes = mesh.extend(NUM_VIEWS)

    # Prepare the camera:
    # specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = pytorch3d.renderer.look_at_view_transform(dist=5.0, azim=azim)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)

    # Render the obj.
    rend = renderer(meshes, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[:, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).

    return rend


if __name__ == "__main__":
    print(f" ------ initialzing script ------ ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="images/tetra_render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    images = render_GIF_from_tetrahedron( image_size=args.image_size)
    print(f" --- size of images: {len(images)} --- ")
    # # Save the rendered images as GIF
    my_images = images  # List of images [(H, W, 3)]
    imageio.mimsave(args.output_path, my_images, fps=15)
    
    print(f" ------ ending script ------ ")
