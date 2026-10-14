"""
Mesh objects module for handling individual mesh components.
"""

import torch
import open3d as o3d
import numpy as np
import trimesh

class MeshObject:
    """
    Represents a single mesh object (hand, object, etc.) with time-varying transformations.
    """
    def __init__(self, name, vertices, faces, vertex_colors, device="cuda",
                 uvs=None, texture_image=None, mesh_downsample_n_vert=5000):
        """
        Initialize a mesh object.

        Args:
            name: Name identifier for this mesh (e.g., "hand", "object")
            vertices: [N, 3] tensor of canonical vertex positions
            faces: [F, 3] tensor of face indices
            vertex_colors: [N, 3] tensor of vertex colors (learnable)
            device: device to store tensors
            uvs: Optional [N, 2] tensor of UV coordinates
            texture_image: Optional [H, W, 3] tensor of texture image
        """
        self.name = name
        self.device = device
        self.canonical_vertices = vertices.to(device)
        self.faces = faces.to(device)
        self.vertex_colors = vertex_colors  # This is a nn.Parameter

        # Texture data (optional)
        self.uvs = uvs.to(device) if uvs is not None else None
        self.texture_image = texture_image  # This can be a nn.Parameter if optimizable

        # Current frame state (updated per frame)
        self.current_vertices = self.canonical_vertices.clone()

        if name == "object" and vertices.shape[0] > mesh_downsample_n_vert:
            self.downsample_mesh(mesh_downsample_n_vert)

    def get_transformed_vertices(self):
        """Get current transformed vertices."""
        return self.current_vertices

    def apply_transform(self, R, t, scale:float=1.0):
        """
        Apply 6DoF transformation and scaling to current vertices.

        Args:
            pose_6dof: [6] tensor with rotation (first 3) and translation (last 3)
            scale: scalar scaling factor
        """

        if isinstance(scale, torch.Tensor) and scale.shape == (3,):
            scale = scale.to(self.device)[None, :]
        scaled = self.canonical_vertices * scale
        rotated = scaled @ R.T
        translated = rotated + t.unsqueeze(0)
        self.current_vertices = translated

    @property
    def num_vertices(self):
        return self.canonical_vertices.shape[0]

    def _interpolate_colors_nearest_neighbor(self, new_vertices, old_vertices, old_colors):
        from scipy.spatial import cKDTree

        tree = cKDTree(old_vertices)
        distances, indices = tree.query(new_vertices, k=1)
        new_colors = old_colors[indices]

        return new_colors

    def downsample_mesh(self, n_vert=5000):
        'Downsample mesh.'
        import pymeshlab

        print(f"[Downsample {self.name}] Starting downsampling to ~{n_vert} vertices...")


        ms = pymeshlab.MeshSet()


        vertices_np = self.canonical_vertices.cpu().numpy()
        faces_np = self.faces.cpu().numpy()


        if self.uvs is not None:
            uvs_np = self.uvs.cpu().numpy()


            wedge_texcoords = uvs_np[faces_np]  # [F, 3, 2]


            m = pymeshlab.Mesh(
                vertex_matrix=vertices_np,
                face_matrix=faces_np,
                v_tex_coords_matrix=uvs_np,  # ✅ per-vertex UV
                w_tex_coords_matrix=wedge_texcoords.reshape(-1, 2)  # ✅ per-wedge UV
            )
        else:
            m = pymeshlab.Mesh(
                vertex_matrix=vertices_np,
                face_matrix=faces_np
            )

        ms.add_mesh(m)


        target_faces = n_vert * 2
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            preserveboundary=True,
            planarquadric=True,
            qualitythr=0.3,
        )


        m_simplified = ms.current_mesh()

        new_vertices = torch.tensor(
            m_simplified.vertex_matrix(),
            dtype=torch.float32,
            device=self.device
        )
        new_faces = torch.tensor(
            m_simplified.face_matrix(),
            dtype=torch.long,
            device=self.device
        )


        if self.uvs is not None:

            if hasattr(m_simplified, 'vertex_tex_coord_matrix'):
                new_uvs_np = m_simplified.vertex_tex_coord_matrix()
                new_uvs = torch.tensor(new_uvs_np, dtype=torch.float32, device=self.device)


            elif hasattr(m_simplified, 'wedge_tex_coord_matrix'):
                wedge_texcoords = m_simplified.wedge_tex_coord_matrix()  # [F*3, 2]


                new_uvs_np = np.zeros((len(new_vertices), 2))
                uv_counts = np.zeros(len(new_vertices))

                face_flat = new_faces.cpu().numpy().flatten()
                for i, v_idx in enumerate(face_flat):
                    new_uvs_np[v_idx] += wedge_texcoords[i]
                    uv_counts[v_idx] += 1


                uv_counts[uv_counts == 0] = 1
                new_uvs_np /= uv_counts[:, None]

                new_uvs = torch.tensor(new_uvs_np, dtype=torch.float32, device=self.device)

            else:

                print("Warning: Could not extract UV from PyMeshLab, using nearest neighbor interpolation")
                new_uvs = self._interpolate_uvs_nearest_neighbor(
                    new_vertices.cpu().numpy(),
                    vertices_np,
                    self.uvs.cpu().numpy()
                )
                new_uvs = torch.tensor(new_uvs, dtype=torch.float32, device=self.device)
        else:
            new_uvs = None


        old_v_count = self.canonical_vertices.shape[0]
        old_f_count = self.faces.shape[0]

        self.canonical_vertices = new_vertices
        self.faces = new_faces
        self.uvs = new_uvs
        self.current_vertices = new_vertices.clone()


        if hasattr(self, 'vertex_colors') and self.vertex_colors is not None:
            old_colors = self.vertex_colors.detach().cpu().numpy()
            new_colors = self._interpolate_colors_nearest_neighbor(
                new_vertices.cpu().numpy(),
                vertices_np,
                old_colors
            )
            self.vertex_colors.data = torch.tensor(
                new_colors, dtype=torch.float32, device=self.device
            )

        print(f"[Downsample {self.name}] "
            f"Vertices: {old_v_count} → {new_vertices.shape[0]} "
            f"({new_vertices.shape[0]/old_v_count*100:.1f}%), "
            f"Faces: {old_f_count} → {new_faces.shape[0]} "
            f"({new_faces.shape[0]/old_f_count*100:.1f}%)")

        vertices_np = self.current_vertices.detach().cpu().numpy()
        faces_np = self.faces.detach().cpu().numpy()
        uvs_np = self.uvs.detach().cpu().numpy()

        mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            process=False
        )


        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uvs_np,
            image=None
        )

        print(f"Mesh watertight: {mesh.is_watertight}")


        if self.texture_image is not None:
            from PIL import Image
            texture_np = self.texture_image.detach().cpu().numpy()
            if texture_np.max() <= 1.0:
                texture_np = (texture_np * 255).astype(np.uint8)
            texture_pil = Image.fromarray(texture_np)
            mesh.visual.material.image = texture_pil


        import os
        os.makedirs("debug", exist_ok=True)
        mesh.export('debug/test_downsampled.obj')

        return True

import smplx

class HandMesh(MeshObject):
    """
    Specialized MeshObject for hand meshes with MANO parameters.
    """
    def __init__(self, name, vertices, faces, vertex_colors,
                 device="cuda", uvs=None, texture_image=None):
        super().__init__(name, vertices, faces, vertex_colors, device,
                         uvs, texture_image)

    def compute_one_frame_vertices(self, mano_model, betas, hand_pose, global_orient,
                                   mano_transform_R, mano_transform_t, transform_scale,
                                   full_R=None, full_t=None, compute_mano=True):
        """
        Compute hand vertices for one frame given MANO parameters and transformation.

        Args:
            betas: [1, 10] tensor of MANO shape parameters
            hand_pose: [1, 15, 3, 3] tensor of MANO hand pose parameters
            global_orient: [1, 1, 3, 3] tensor of global orientation
            transform_R: [3, 3] rotation matrix
            transform_t: [3] translation vector
            transform_scale: scalar scaling factor
        """
        # Get MANO vertices
        if compute_mano:
            mano_output = mano_model(betas=betas, hand_pose=hand_pose, pose2rot=False)  # [778, 3]

            # Get vertices and joints in MANO canonical space
            vertices = mano_output.vertices[0]  # [778, 3]
            joints = mano_output.joints[0]  # [21, 3]

            # Apply global orientation rotation around wrist (joint 0)
            wrist_pos = joints[0]  # [3]
            vertices_centered = vertices - wrist_pos
            joints_centered = joints - wrist_pos
            self.canonical_vertices = vertices_centered.clone()
            self.canonical_joints = joints_centered.clone()
        else:
            assert hasattr(self, 'canonical_joints'), "canonical_joints not found."
            assert hasattr(self, 'canonical_vertices'), "canonical_vertices not found."
            vertices_centered = self.canonical_vertices
            joints_centered = self.canonical_joints

        if full_R is not None and full_t is not None:
            # Use provided full rotation and translation
            composed_R = full_R
            mano_transform_t = full_t
        else:
            composed_R = mano_transform_R @ global_orient[0, 0]

        vertices_transformed = (vertices_centered * transform_scale @ composed_R.T) + mano_transform_t.unsqueeze(0)
        joints_transformed = (joints_centered * transform_scale @ composed_R.T) + mano_transform_t.unsqueeze(0)

        self.current_vertices = vertices_transformed
        self.current_joints = joints_transformed

        return composed_R, mano_transform_t
