import bpy
import math
import mathutils
import os

blend_file_path = "/Users/bryanboateng/Uni/4d-gaussian-splatting/santa_pink.blend"
output_directory_path = "/Users/bryanboateng/Uni/4d-gaussian-splatting/data/santa"
camera_count = 12
radius = 10

bpy.ops.wm.open_mainfile(filepath=blend_file_path)

animation_duration = (
    bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
) / bpy.context.scene.render.fps
print(
    f"Animation duration: {animation_duration} s at {bpy.context.scene.render.fps} FPS"
)

for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

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

bpy.context.scene.render.image_settings.file_format = "JPEG"
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    for index, camera in enumerate(cameras):
        bpy.context.scene.camera = camera
        bpy.context.scene.render.filepath = os.path.join(
            output_directory_path, "ims", str(index), f"{frame - 1:06}.jpg"
        )
        bpy.ops.render.render(write_still=True)
