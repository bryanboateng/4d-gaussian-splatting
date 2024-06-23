import bpy
import math
import os

output_path = "/Users/bryanboateng/Uni/moin/4d-gaussian-splatting"

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

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
    camera.rotation_euler = (math.pi / 2,  0, angle + math.pi / 2)
    cameras.append(camera)


bpy.ops.mesh.primitive_monkey_add()
obj = bpy.context.object

light = bpy.data.objects.new(name="Sun_Light", object_data=(bpy.data.lights.new(name="Sun_Light", type='SUN')))
bpy.context.collection.objects.link(light)
light.location = (0, 0, 10)
light.rotation_euler = (math.radians(45), math.radians(-45), 0)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 2

bpy.context.scene.render.image_settings.file_format = "JPEG"

for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    for index, camera in enumerate(cameras):
        bpy.context.scene.camera = camera
        bpy.context.scene.render.filepath = os.path.join(
            output_path, "ims", str(index), f"{frame:06}.jpg"
        )
        bpy.ops.render.render(write_still=True)
