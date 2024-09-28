def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, normal=None, depth=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


# Transform mesh vertices to COLMAP coordinates frame 
def blender_to_colmap_points(self, points):
        """
        Transform a set of 3D points from Blender's coordinate frame to COLMAP's coordinate frame.

        Args:
        - points (torch.Tensor): A tensor of shape (N, 3) representing N 3D points in Blender's coordinate frame.

        Returns:
        - torch.Tensor: A tensor of shape (N, 3) representing N 3D points in COLMAP's coordinate frame.
        """
        assert points.shape[-1] == 3, "Input points must be of shape (N, 3)."

        # Transformation matrix from Blender to COLMAP
        T_blender_to_colmap = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=points.dtype, device=points.device)

        # Apply the transformation
        points_colmap = points @ T_blender_to_colmap.T

        return points_colmap