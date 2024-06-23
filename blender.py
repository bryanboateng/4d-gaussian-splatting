import bpy
import math
import os

blend_file_path = "/Users/bryanboateng/Uni/4d-gaussian-splatting/santa.blend"
output_directory_path = "/Users/bryanboateng/Uni/4d-gaussian-splatting/data/santa"

bpy.ops.wm.open_mainfile(filepath=blend_file_path)

animation_duration = (
    bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
) / bpy.context.scene.render.fps
print(
    f"Animation duration: {animation_duration} s at {bpy.context.scene.render.fps} FPS"
)

cameras = []
camera_count = 12
for i in range(camera_count):
    angle = 2 * math.pi * i / camera_count
    x = 5 * math.cos(angle)
    y = 5 * math.sin(angle)
    camera = bpy.data.objects.new(
        name=f"Camera_{i}", object_data=(bpy.data.cameras.new(name=f"Camera_{i}"))
    )
    bpy.context.collection.objects.link(camera)
    camera.location = (x, y, 0)
    camera.rotation_euler = (math.pi / 2, 0, angle + math.pi / 2)
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
