import bpy
import json
import math
import mathutils
import os


def get_scaled_resolution():
    scale_percentage = bpy.context.scene.render.resolution_percentage / 100
    return (
        bpy.context.scene.render.resolution_x * scale_percentage,
        bpy.context.scene.render.resolution_y * scale_percentage,
    )


def method_name():
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    animation_duration = (
        bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
    ) / bpy.context.scene.render.fps
    print(
        f"Animation duration: {animation_duration} s at {bpy.context.scene.render.fps} FPS"
    )

    delete_all_cameras()
    cameras = create_cameras()
    intrinsic_matrices = []
    extrinsic_matrices = []
    file_names = []
    camera_ids = []
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        camera_ids.append([])
        file_names.append([])
        intrinsic_matrices.append([])
        extrinsic_matrices.append([])
        bpy.context.scene.frame_set(frame)
        for index, camera in enumerate(cameras):
            bpy.context.scene.camera = camera
            wefw = os.path.join(str(index), f"{frame - 1:06}.jpg")
            bpy.context.scene.render.filepath = os.path.join(
                output_directory_path, "ims", wefw
            )
            bpy.ops.render.render(write_still=True)
            intrinsic_matrices[-1].append(get_intrinsic_matrix(camera))
            extrinsic_matrices[-1].append(get_extrinsic_matrix(camera))
            file_names[-1].append(wefw)
            camera_ids[-1].append(index)
            with open(
                os.path.join(output_directory_path, "train_meta.json"), "w"
            ) as json_file:
                json.dump(
                    {
                        "w": resolution_x_in_px,
                        "h": resolution_y_in_px,
                        "k": intrinsic_matrices,
                        "w2c": extrinsic_matrices,
                        "fn": file_names,
                        "cam_id": camera_ids,
                    },
                    json_file,
                    indent=4,
                )


def delete_all_cameras():
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)


def create_cameras():
    cameras = []
    golden_angle = math.pi * (math.sqrt(5) - 1)
    for i in range(camera_count):
        # Calculate the z-coordinate (height) for the current point, ranging from radius to -radius
        z = radius * (1 - (i / float(camera_count - 1)) * 2)

        # Calculate the radius of the cross-section circle at the current height (z_coordinate)
        cross_section_radius = math.sqrt(radius * radius - z * z)

        # Calculate the angle for the current point on the cross-section circle
        angle = golden_angle * i

        x = math.cos(angle) * cross_section_radius
        y = math.sin(angle) * cross_section_radius

        camera = bpy.data.objects.new(
            name=f"Camera_{i}", object_data=(bpy.data.cameras.new(name=f"Camera_{i}"))
        )
        bpy.context.collection.objects.link(camera)
        camera.location = mathutils.Vector((x, y, z))
        rot_quat = camera.location.to_track_quat("Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()
        cameras.append(camera)
    return cameras


def get_intrinsic_matrix(camera):
    focal_length_in_mm = camera.data.lens

    pixel_aspect_ratio = (
        bpy.context.scene.render.pixel_aspect_x
        / bpy.context.scene.render.pixel_aspect_y
    )

    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height

    if camera.data.sensor_fit == "VERTICAL":
        focal_length_x_in_px = (
            resolution_y_in_px / sensor_height_in_mm
        ) * focal_length_in_mm
        focal_length_y_in_px = pixel_aspect_ratio * focal_length_x_in_px
    else:
        focal_length_x_in_px = (
            resolution_x_in_px / sensor_width_in_mm
        ) * focal_length_in_mm
        focal_length_y_in_px = focal_length_x_in_px / pixel_aspect_ratio

    principal_point_x_in_px = resolution_x_in_px * (0.5 - camera.data.shift_x)
    principal_point_y_in_px = resolution_y_in_px * (0.5 - camera.data.shift_y)
    skew_in_px = (
        camera.data.shift_x * focal_length_x_in_px
        if hasattr(camera.data, "shift_x")
        else 0
    )

    return [
        [focal_length_x_in_px, skew_in_px, principal_point_x_in_px],
        [0, focal_length_y_in_px, principal_point_y_in_px],
        [0, 0, 1],
    ]


def get_extrinsic_matrix(camera):
    extrinsic_matrix = camera.matrix_world.inverted()
    return [
        [
            extrinsic_matrix[0][0],
            extrinsic_matrix[0][1],
            extrinsic_matrix[0][2],
            extrinsic_matrix[0][3],
        ],
        [
            extrinsic_matrix[1][0],
            extrinsic_matrix[1][1],
            extrinsic_matrix[1][2],
            extrinsic_matrix[1][3],
        ],
        [
            extrinsic_matrix[2][0],
            extrinsic_matrix[2][1],
            extrinsic_matrix[2][2],
            extrinsic_matrix[2][3],
        ],
        [
            extrinsic_matrix[3][0],
            extrinsic_matrix[3][1],
            extrinsic_matrix[3][2],
            extrinsic_matrix[3][3],
        ],
    ]


blend_file_path = "/Users/bryanboateng/Uni/4d-gaussian-splatting/santa_pink.blend"
output_directory_path = "/Users/bryanboateng/Uni/4d-gaussian-splatting/data/santa"
camera_count = 12
radius = 10
resolution_x_in_px, resolution_y_in_px = get_scaled_resolution()
method_name()
