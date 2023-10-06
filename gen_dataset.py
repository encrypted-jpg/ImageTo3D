import os
import argparse
import time
import json
from tqdm import tqdm
from stdout_redirected import stdout_redirected
import concurrent.futures
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='final.json')

args = parser.parse_args()

rotation_sets = [
    (0, 0, 0),          # Set 1: No rotation
    (0, 45, 0),         # Set 2: 45-degree rotation around Y-axis
    (45, 0, 0),         # Set 3: 45-degree rotation around X-axis
    (30, 30, 30),       # Set 4: 30-degree rotation around X, Y, and Z axes
    (60, 60, 60),       # Set 5: 60-degree rotation around X, Y, and Z axes
    (90, 90, 90)        # Set 6: 90-degree rotation around X, Y, and Z axes
]

data_path = "../ShapeNet/"
base_dir = "./ShapeNetRender"


def generate_data(files_info):
    import bpy
    with stdout_redirected():
        for o in bpy.context.scene.objects:
            if o.name == "Cube":
                bpy.ops.object.delete(use_global=False)

        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"

    start = time.time()
    previous = ""
    base_key_name = ""
    for filename, cat_id, model_id in files_info:
        # print(filename, cat_id, model_id)
        try:
            with stdout_redirected():
                if previous != "":
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.data.objects[previous].select_set(True)
                    bpy.ops.object.delete()

                bpy.ops.import_scene.obj(filepath=filename)

                if base_key_name == "":
                    base_key_name = os.path.basename(
                        filename).replace(".obj", "")

                for key in bpy.data.objects.keys():
                    if key.startswith(base_key_name):
                        previous = key
                        break

                model = bpy.data.objects[previous]
                cam = bpy.data.objects['Camera']
                light = bpy.data.objects['Light']
                light.data.energy = 5000
                distance = 2

            path = os.path.join(base_dir, cat_id, model_id)
            os.makedirs(path, exist_ok=True)

            for rotation in tqdm(rotation_sets, desc=f"{cat_id}/{model_id}"):
                with stdout_redirected():
                    model.rotation_euler = rotation

                    cam.location = (0, distance, 0)
                    cam.rotation_euler = (1.570, 0, 3.1415)
                    light.location = cam.location

                    bpy.context.scene.render.resolution_x = 224
                    bpy.context.scene.render.resolution_y = 224
                    bpy.context.scene.render.resolution_percentage = 100
                    bpy.context.scene.render.engine = 'CYCLES'
                    bpy.context.scene.render.filepath = os.path.join(
                        path, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_0.png")
                    bpy.ops.render.render(write_still=True)

                    cam.location = (0, -distance, 0)
                    cam.rotation_euler = (-1.570, 0, 3.1415)
                    light.location = cam.location

                    bpy.context.scene.render.filepath = os.path.join(
                        path, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_1.png")
                    bpy.ops.render.render(write_still=True)

                    img = cv2.imread(os.path.join(
                        path, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_0.png"), cv2.COLOR_BGRA2RGB)
                    cv2.imwrite(os.path.join(
                        path, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_0.png"), img)
                    img = cv2.imread(os.path.join(
                        path, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_1.png"), cv2.COLOR_BGRA2RGB)
                    cv2.imwrite(os.path.join(
                        path, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_1.png"), img)
        except TypeError as e:
            print(e)
            continue

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return True


with open(args.json, "r") as f:
    final = json.load(f)


def check_data_gen(folder):
    for rotation in rotation_sets:
        pt0 = os.path.join(
            folder, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_0.png")
        pt1 = os.path.join(
            folder, f"render_{rotation[0]}_{rotation[1]}_{rotation[2]}_1.png")
        if not os.path.exists(pt0) or not os.path.exists(pt1):
            return False
    return True


existing_count = 0
filelist = []
for key in final.keys():
    for _mode in final[key]:
        for model in final[key][_mode]:
            if not check_data_gen(os.path.join(base_dir, key, model)):
                filelist.append((os.path.join(data_path, key,
                                key, model, "models", "model_normalized.obj"), key, model))
            else:
                existing_count += 1

print(f"Existing count: {existing_count}")
print(f"Filelist count: {len(filelist)}")
generate_data(filelist)

# flists = []
# diff = 10
# for i in range(0, len(filelist), diff):
#     flists.append(filelist[i:i+diff])

# with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#     results = [executor.submit(generate_data, flist)
#                for flist in flists]

# concurrent.futures.wait(results)

# outs = [future.result() for future in results]
# print("Done")
