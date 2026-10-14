"""
Mesh renderer module for PyTorch3D-based scene rendering.
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    BlendParams,
    PointLights,
    Materials,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.ops import interpolate_face_attributes
from pathlib import Path
import yaml

from optimization.config.config import Config
from optimization.models.mesh_objects import MeshObject, HandMesh
from optimization.models.mano_wrapper import MANO


class SceneMeshRenderer(torch.nn.Module):
    """
    Unified scene renderer that manages multiple mesh objects (hand, object, etc.)
    and renders them together in a single pass.
    """

    def __init__(self, cfg: Config, device="cuda"):
        """
        Initialize scene mesh renderer.

        Args:
            cfg: Configuration object
            device: device to store tensors
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Dictionary to store mesh objects: {name: MeshObject}
        self.mesh_objects = {}

        # PyTorch3D renderer components (shared across all objects)
        self._setup_renderer()

    def _setup_renderer(self):
        """Setup PyTorch3D renderer with rasterizer and shader."""
        # Rasterization settings

        raster_settings = RasterizationSettings(
            image_size=self.cfg.rasterization_image_size,
            blur_radius=self.cfg.rasterization_blur_radius,
            faces_per_pixel=self.cfg.rasterization_faces_per_pixel,
            bin_size=self.cfg.rasterization_bin_size,
            max_faces_per_bin=self.cfg.rasterization_max_faces_per_bin,
        )

        # pytorch3d rasterizer
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)

        # Setup shader based on configuration
        # Always create both shaders so we can switch between them based on texture availability

        # Silhouette shader (for simple alpha rendering)
        silhouette_blend_params = BlendParams(sigma=self.cfg.rasterization_blend_sigma,
                                              gamma=self.cfg.rasterization_blend_gamma,
                                              background_color=(0.0, 0.0, 0.0))
        self.silhouette_shader = SoftSilhouetteShader(blend_params=silhouette_blend_params)

        # Phong shader (for RGB/texture rendering)

        lights = PointLights(
            ambient_color=(self.cfg.light_ambient_color,),
            diffuse_color=(self.cfg.light_diffuse_color,),
            specular_color=(self.cfg.light_specular_color,),
            location=((0.0, 0.0, 0.0),),
            device=self.device
        )

        materials = Materials(
            ambient_color=(self.cfg.initial_vertex_color,),
            diffuse_color=(self.cfg.initial_vertex_color,),
            specular_color=((0.2, 0.2, 0.2),),
            shininess=self.cfg.material_shininess,
            device=self.device
        )

        phong_blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
        self.phong_shader = SoftPhongShader(
            lights=lights,
            materials=materials,
            blend_params=phong_blend_params
        )

    def add_mesh_object(self, name, vertices, faces, initial_color=(0.7, 0.7, 0.7), optimize_colors=True,
                        uvs=None, texture_image=None, optimize_texture=False,
                        downsample_n_vert=5000):
        'Add mesh object.'
        # Convert to torch tensors if needed
        if not torch.is_tensor(vertices):
            vertices = torch.from_numpy(vertices).float()
        if not torch.is_tensor(faces):
            faces = torch.from_numpy(faces).long()

        vertices = vertices.to(self.device)
        faces = faces.to(self.device)

        # Process UV coordinates
        if uvs is not None and not torch.is_tensor(uvs):
            uvs = torch.from_numpy(uvs).float()

        # Process texture image
        texture_param = None
        if texture_image is not None:
            if not torch.is_tensor(texture_image):
                texture_image = torch.from_numpy(texture_image).float()

            # Normalize to [0, 1] if needed
            if texture_image.max() > 1.0:
                texture_image = texture_image / 255.0

            texture_image = texture_image.to(self.device)

            if optimize_texture:
                # Make texture optimizable

                texture_param = torch.nn.Parameter(texture_image.clone())
                self.register_parameter(f"{name}_texture", texture_param)
            else:
                texture_param = texture_image

        # Initialize vertex colors
        num_vertices = vertices.shape[0]
        if optimize_colors:
            init_colors = torch.tensor(initial_color, device=self.device).expand(num_vertices, 3)
            vertex_colors = torch.nn.Parameter(init_colors.clone())
            # Register as parameter
            self.register_parameter(f"{name}_vertex_colors", vertex_colors)
        else:
            vertex_colors = torch.tensor(initial_color, device=self.device).expand(num_vertices, 3)

        # Create and store mesh object
        if 'hand' in name.lower():
            mesh_obj = HandMesh(name, vertices, faces, vertex_colors, device=self.device,
                                uvs=uvs, texture_image=texture_param)
        else:
            mesh_obj = MeshObject(name, vertices, faces, vertex_colors, device=self.device,
                                  uvs=uvs, texture_image=texture_param,
                                  mesh_downsample_n_vert=downsample_n_vert)
        self.mesh_objects[name] = mesh_obj

        texture_info = ""
        if texture_param is not None:
            texture_info = f", texture: {texture_param.shape}"
        print(f"Added mesh object '{name}': {num_vertices} vertices, {faces.shape[0]} faces{texture_info}")
        return mesh_obj

    def get_mesh_object(self, name):
        """Get a mesh object by name."""
        return self.mesh_objects.get(name, None)

    def get_combined_mesh(self, include_objects=None):
        """
        Combine specified mesh objects into a single PyTorch3D Meshes object.

        Args:
            include_objects: Optional list of object names to include (e.g., ["hand"], ["object"]).
                           If None, includes all objects.

        Returns:
            meshes: PyTorch3D Meshes object containing specified scene objects
            vertex_colors: [total_vertices, 3] tensor of all vertex colors
            object_vertex_ranges: dict mapping object name to (start_idx, end_idx) in combined mesh
        """
        # Filter objects to include
        if include_objects is not None:
            objects_to_render = {name: obj for name, obj in self.mesh_objects.items()
                               if name in include_objects}
        else:
            objects_to_render = self.mesh_objects

        if len(objects_to_render) == 0:
            return None, None, {}

        all_vertices = []
        all_faces = []
        all_colors = []
        all_uvs = []
        all_texture_maps = []
        vertex_offset = 0
        face_offset = 0
        object_vertex_ranges = {}
        has_texture = False

        for name, mesh_obj in objects_to_render.items():
            # Get current transformed vertices
            verts = mesh_obj.get_transformed_vertices()
            all_vertices.append(verts)

            # Offset face indices for concatenation
            faces = mesh_obj.faces + vertex_offset
            all_faces.append(faces)

            # Get vertex colors
            all_colors.append(mesh_obj.vertex_colors)

            # Check for UV texture

            if mesh_obj.uvs is not None and mesh_obj.texture_image is not None:
                all_uvs.append(mesh_obj.uvs)
                all_texture_maps.append(mesh_obj.texture_image)
                has_texture = True
            else:
                # Create dummy UVs if this object doesn't have texture
                if has_texture:

                    # Pad with zeros to match dimension
                    dummy_uvs = torch.zeros(mesh_obj.num_vertices, 2, device=self.device)
                    all_uvs.append(dummy_uvs)
                    # Create a dummy 1x1 white texture for this object
                    dummy_texture = torch.ones(1, 1, 3, device=self.device)
                    all_texture_maps.append(dummy_texture)

            # Track vertex ranges for each object
            object_vertex_ranges[name] = (vertex_offset, vertex_offset + mesh_obj.num_vertices)
            vertex_offset += mesh_obj.num_vertices
            face_offset += faces.shape[0]

        # Concatenate all meshes
        combined_vertices = torch.cat(all_vertices, dim=0)  # [total_vertices, 3]
        combined_faces = torch.cat(all_faces, dim=0)  # [total_faces, 3]
        combined_colors = torch.cat(all_colors, dim=0)  # [total_vertices, 3]

        # Create PyTorch3D Meshes object with textures if available
        if has_texture and len(all_uvs) > 0:
            # Concatenate UVs - For TexturesUV, we need per-face UVs, not per-vertex
            # In PyTorch3D, TexturesUV expects:
            # - faces_uvs: [num_faces, 3] indices into verts_uvs
            # - verts_uvs: [num_verts_uvs, 2] UV coordinates
            # - texture_maps: list of [H, W, 3] texture images per mesh

            # For now, we'll create separate meshes for each object if they have different textures
            # This is necessary because PyTorch3D TexturesUV expects one texture map per mesh in the batch
            mesh_list = []
            vertex_offset_temp = 0

            for i, (name, mesh_obj) in enumerate(objects_to_render.items()):
                verts = all_vertices[i]
                faces = mesh_obj.faces  # Use original faces (not offset)

                if mesh_obj.uvs is not None and mesh_obj.texture_image is not None:
                    # Create TexturesUV for this object
                    # Need to ensure texture_image is [1, H, W, 3] format
                    texture_map = mesh_obj.texture_image
                    if len(texture_map.shape) == 3:
                        texture_map = texture_map.unsqueeze(0)  # [1, H, W, 3]

                    # faces_uvs should be the same as faces for vertex-based UVs
                    textures = TexturesUV(
                        maps=texture_map,  # [1, H, W, 3]
                        faces_uvs=[faces],  # List of [F, 3]
                        verts_uvs=[mesh_obj.uvs]  # List of [V, 2]
                    )

                    mesh = Meshes(
                        verts=[verts],
                        faces=[faces],
                        textures=textures
                    )
                else:
                    # Use vertex colors with TexturesUV (create uniform texture from vertex colors)
                    # Create a 1x1 texture map from the mean vertex color
                    mean_color = mesh_obj.vertex_colors.mean(dim=0, keepdim=True)  # [1, 3]
                    uniform_texture = mean_color.view(1, 1, 1, 3)  # [1, 1, 1, 3]

                    # Create dummy UVs (all pointing to [0.5, 0.5])
                    dummy_uvs = torch.ones(mesh_obj.num_vertices, 2, device=self.device) * 0.5

                    textures = TexturesUV(
                        maps=uniform_texture,  # [1, 1, 1, 3]
                        faces_uvs=[faces],  # List of [F, 3]
                        verts_uvs=[dummy_uvs]  # List of [V, 2]
                    )

                    mesh = Meshes(
                        verts=[verts],
                        faces=[faces],
                        textures=textures
                    )

                mesh_list.append(mesh)
                vertex_offset_temp += mesh_obj.num_vertices

            # Join all meshes as a scene (not batch)
            meshes = pytorch3d.structures.join_meshes_as_scene(mesh_list)
        else:
            # No texture, use vertex colors as TexturesVertex
            vertex_colors_rgb = combined_colors.unsqueeze(0)  # [1, total_vertices, 3]
            textures = TexturesVertex(verts_features=vertex_colors_rgb)

            meshes = Meshes(
                verts=[combined_vertices],
                faces=[combined_faces],
                textures=textures
            )

        return meshes, combined_colors, object_vertex_ranges

    def render(self, cameras, image_height, image_width,
               include_objects=None, soft_alpha=True, require_normal=False,
               force_hard_rasterization=False):
        """
        Render specified mesh objects in the scene.

        Args:
            cameras: PyTorch3D cameras object
            image_height: Target image height
            image_width: Target image width
            include_objects: Optional list of object names to render (e.g., ["hand"], ["object"]).
                           If None, renders all objects.

        Returns:
            Dictionary containing rendered outputs:
                - colors: [B, H, W, 3] RGB colors
                - alpha: [B, H, W, 1] silhouette/mask
                - depth: [B, H, W, 1] depth values
                - normals: [B, H, W, 3] surface normals in camera space (compatible with GT normals from depth)
                - fragments: PyTorch3D rasterization fragments
                - object_vertex_ranges: dict mapping object name to vertex ranges
        """
        # Get combined mesh with optional filtering
        meshes, vertex_colors, object_vertex_ranges = self.get_combined_mesh(include_objects)

        # Update rasterization settings for target image size
        if (image_height != self.cfg.rasterization_image_size or
            image_width != self.cfg.rasterization_image_size or
            force_hard_rasterization):
            if force_hard_rasterization:
                blur_radius = 0.0
            else:
                blur_radius = self.cfg.rasterization_blur_radius
            raster_settings = RasterizationSettings(
                image_size=(image_height, image_width),
                blur_radius=blur_radius,
                faces_per_pixel=self.cfg.rasterization_faces_per_pixel,
                bin_size=self.cfg.rasterization_bin_size,
                max_faces_per_bin=self.cfg.rasterization_max_faces_per_bin,
            )
            rasterizer = MeshRasterizer(raster_settings=raster_settings)
        else:
            rasterizer = self.rasterizer

        # Rasterization
        fragments = rasterizer(meshes, cameras=cameras)

        # Extract rasterization outputs
        pix_to_face = fragments.pix_to_face  # [B, H, W, K]
        zbuf = fragments.zbuf  # [B, H, W, K]
        bary_coords = fragments.bary_coords  # [B, H, W, K, 3]

        # Compute alpha (silhouette)
        if soft_alpha:
            silhouette_output = self.silhouette_shader(fragments, meshes, cameras=cameras)
            alpha = silhouette_output[..., 3:]  # [B, H, W, 1]
        else:
            alpha = (pix_to_face[..., 0] >= 0).float().unsqueeze(-1)  # [B, H, W, 1]

        # Compute depth from z-buffer (take closest face)
        depth = zbuf[..., 0:1]  # [B, H, W, 1]
        depth = torch.where(alpha > 0, depth, torch.zeros_like(depth))

        # Compute surface normals in world space
        # Get vertex positions and face indices
        verts_packed = meshes.verts_packed()  # [total_vertices, 3]
        faces_packed = meshes.faces_packed()  # [total_faces, 3]

        if require_normal:
            # Get face vertices for each face
            face_verts = verts_packed[faces_packed]  # [total_faces, 3, 3]

            # Compute face normals using cross product
            v0 = face_verts[:, 0, :]  # [total_faces, 3]
            v1 = face_verts[:, 1, :]  # [total_faces, 3]
            v2 = face_verts[:, 2, :]  # [total_faces, 3]

            # Compute edges
            e1 = v1 - v0  # [total_faces, 3]
            e2 = v2 - v0  # [total_faces, 3]

            # Compute face normals (cross product)
            face_normals = torch.cross(e1, e2, dim=1)  # [total_faces, 3]
            face_normals = F.normalize(face_normals, dim=1)  # Normalize to unit length

            # Map face normals to pixels using pix_to_face
            # pix_to_face: [B, H, W, K] contains face indices
            # We need to expand face_normals to match pixel dimensions
            batch_size = pix_to_face.shape[0]
            H, W = pix_to_face.shape[1], pix_to_face.shape[2]

            # Initialize normals with zeros
            normals = torch.zeros(batch_size, H, W, 3, device=self.device)

            # Get valid pixels (where a face is visible)
            valid_mask = pix_to_face[..., 0] >= 0  # [B, H, W]

            # Get face indices for valid pixels
            valid_face_idx = pix_to_face[..., 0].clamp(min=0)  # [B, H, W]

            # Index face normals with valid face indices
            pixel_normals = face_normals[valid_face_idx]  # [B, H, W, 3]

            # Apply valid mask
            normals = torch.where(valid_mask.unsqueeze(-1), pixel_normals, normals) # [B, H, W, 3]

            # Flip YZ axis to match image coordinate system
            normals[..., 1:] *= -1
        else:
            normals = None

        # Render colors - use texture if available, otherwise use vertex colors
        if meshes.textures is not None:
            # Use Phong shader to render with textures (required for proper texture rendering)
            colors = self.phong_shader(fragments, meshes, cameras=cameras)[..., :3]  # [B, H, W, 3]
        elif self.cfg.optimize_vertex_colors:
            # Get face colors by indexing vertex colors with face indices
            combined_faces = meshes.faces_packed()  # [total_faces, 3]
            face_colors = vertex_colors[combined_faces]  # [total_faces, 3, 3]

            # Interpolate face vertex colors to pixel colors
            pixel_colors = interpolate_face_attributes(
                pix_to_face, bary_coords, face_colors
            )  # [B, H, W, K, 3]

            # Take closest face colors
            colors = pixel_colors[..., 0, :]  # [B, H, W, 3]
            colors = torch.where(alpha.expand_as(colors) > 0, colors, torch.zeros_like(colors))
        else:
            colors = torch.ones_like(alpha.expand(-1, -1, -1, 3)) * torch.tensor(
                self.cfg.initial_vertex_color, device=self.device
            )
            colors = torch.where(alpha.expand_as(colors) > 0, colors, torch.zeros_like(colors))

        # def compute_visible_vertices(pix_to_face, faces_packed, num_vertices):
        # Get all visible faces (K=0 is the closest face at each pixel)
        visible_faces = pix_to_face[..., 0]  # [B, H, W]
        visible_faces = visible_faces[visible_faces >= 0]  # Filter out background (-1)

        num_vertices = verts_packed.shape[0]
        # if len(visible_faces) == 0:
        #     return torch.zeros(num_vertices, dtype=torch.bool, device=faces_packed.device)

        # Get unique visible faces
        unique_faces = torch.unique(visible_faces)  # [N_visible_faces]
        visible_faces_mask = torch.zeros(faces_packed.shape[0], dtype=torch.bool, device=faces_packed.device)
        visible_faces_mask[unique_faces] = True

        # Get vertices of visible faces
        visible_face_verts = faces_packed[unique_faces]  # [N_visible_faces, 3]
        visible_vertex_indices = visible_face_verts.flatten().unique()  # [N_visible_verts]

        # Create boolean mask
        vertices_visibility = torch.zeros(num_vertices, dtype=torch.bool, device=faces_packed.device)
        vertices_visibility[visible_vertex_indices] = True

        return {
            "colors": colors,   # [B, H, W, 3]
            "alpha": alpha,     # [B, H, W, 1]
            "depth": depth,     # [B, H, W, 1]
            "normals": normals, # [B, H, W, 3]
            "zbuf": zbuf,       # [B, H, W, K]
            "fragments": fragments,
            "object_vertex_ranges": object_vertex_ranges,
            "vertices_visibility": vertices_visibility,  # [total_vertices]
            "face_visibility": visible_faces_mask,  # [N_total_faces]
            "pix_to_face": pix_to_face,  # [B, H, W, K]
            "verts_packed": verts_packed,  # [total_vertices, 3]
        }

    @property
    def num_total_vertices(self):
        """Return total number of vertices across all objects."""
        return sum(obj.num_vertices for obj in self.mesh_objects.values())

    def state_dict(self):
        """Return state dictionary for checkpointing."""
        state = {
            'mesh_objects': {}
        }
        for name, mesh_obj in self.mesh_objects.items():
            state['mesh_objects'][name] = {
                'canonical_vertices': mesh_obj.canonical_vertices,
                'faces': mesh_obj.faces,
                'vertex_colors': mesh_obj.vertex_colors if isinstance(mesh_obj.vertex_colors, torch.nn.Parameter) else None,
            }
        return state

    def load_state_dict(self, state_dict):
        """Load state from dictionary for checkpoint restoration."""
        for name, obj_state in state_dict['mesh_objects'].items():
            if name in self.mesh_objects:
                mesh_obj = self.mesh_objects[name]
                mesh_obj.canonical_vertices = obj_state['canonical_vertices']
                mesh_obj.faces = obj_state['faces']
                if obj_state['vertex_colors'] is not None and isinstance(mesh_obj.vertex_colors, torch.nn.Parameter):
                    mesh_obj.vertex_colors.data = obj_state['vertex_colors'].data


def create_scene_renderer(
    parser,
    cfg: Config,
    mano_model: MANO = None,
    device: str = "cuda",
) -> SceneMeshRenderer:
    """
    Create a unified SceneMeshRenderer containing hand and object meshes.

    Args:
        parser: Dataset parser
        cfg: Configuration object
        mano_model: Optional MANO model for generating canonical hand mesh
        device: Device to use

    Returns:
        SceneMeshRenderer instance with hand and/or object added
    """
    # Create scene renderer
    scene_renderer = SceneMeshRenderer(cfg=cfg, device=device)

    # Add hand mesh if available
    assert mano_model is not None
    # Use MANO model to generate canonical T-pose hand (zero pose, zero betas)
    print("Initializing hand mesh from MANO T-pose (canonical space)")

    # Get hand faces from parser
    hand_faces = parser.mano_faces
    if torch.is_tensor(hand_faces):
        hand_faces = hand_faces.cpu().numpy()

    # Generate T-pose: zero betas (mean shape), zero pose (T-pose/rest pose)
    zero_betas = torch.zeros(1, 10, device=device)
    zero_hand_pose = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(1, 15, 1, 1)  # [1, 15, 3, 3] identity rotations

    # Forward pass through MANO to get T-pose vertices
    mano_output = mano_model(
        hand_pose=zero_hand_pose,
        betas=zero_betas,
        pose2rot=False  # Already in rotation matrix form
    )

    # Get canonical T-pose vertices
    hand_verts_canonical = mano_output.vertices[0].detach().cpu().numpy()  # (778, 3)

    # Add hand to scene
    scene_renderer.add_mesh_object(
        name="hand",
        vertices=hand_verts_canonical,
        faces=hand_faces,
        initial_color=cfg.initial_vertex_color,
        optimize_colors=cfg.optimize_vertex_colors
    )

    # Add object mesh if available
    assert parser.object_mesh is not None
    obj_verts = np.array(parser.object_mesh.vertices)
    obj_faces = np.array(parser.object_mesh.faces)

    # Try to load texture if enabled
    obj_uvs = None
    obj_texture = None

    if cfg.use_object_texture:
        # Check if texture data is available in parser
        if hasattr(parser, 'object_texture') and parser.object_texture is not None:
            obj_texture = parser.object_texture
            print(f"Loading object texture from parser: shape {obj_texture.shape}")

        # Check if UV coordinates are available in mesh
        if hasattr(parser.object_mesh.visual, 'uv') and parser.object_mesh.visual.uv is not None:
            obj_uvs = np.array(parser.object_mesh.visual.uv)
            print(f"Loading object UVs from mesh: shape {obj_uvs.shape}")

    scene_renderer.add_mesh_object(
        name="object",
        vertices=obj_verts,
        faces=obj_faces,
        initial_color=cfg.initial_vertex_color,
        optimize_colors=cfg.optimize_vertex_colors,
        uvs=obj_uvs,
        texture_image=obj_texture,
        optimize_texture=cfg.optimize_object_texture,
        downsample_n_vert=cfg.mesh_downsample_n_vert
    )

    return scene_renderer
