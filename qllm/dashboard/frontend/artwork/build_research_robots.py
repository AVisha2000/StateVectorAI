"""Author the StateVectorAI robot cast in Blender; exports contain no cameras/lights.

Run with Blender --background --python this_file -- --output PUBLIC_DIR --source SOURCE_DIR.
Metres are miniature scene units; Blender +Z up/-Y forward exports to glTF +Y/+Z.
The gesture_arm empty preserves the existing shoulder animation contract.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--field", choices=["physics", "mathematics", "chemistry", "biology", "ai-safety", "machine-learning"],
                    help="Author only one discipline; leave every other export untouched.")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
output = Path(args.output)
source = Path(args.source)
output.mkdir(parents=True, exist_ok=True)
source.mkdir(parents=True, exist_ok=True)


def material(name, color, metallic=0, roughness=.42, emission=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    if emission:
        node.inputs["Emission Color"].default_value = (*color, 1)
        node.inputs["Emission Strength"].default_value = emission
    return mat


def attach(obj, name, mat, parent=None):
    obj.name = name
    obj.data.materials.append(mat)
    if parent:
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = world
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def sphere(name, at, size, mat, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=at)
    obj = bpy.context.object
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return attach(obj, name, mat, parent)


def rounded(name, at, size, mat, radius=.009, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=at)
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = obj.modifiers.new("Soft manufactured edges", "BEVEL")
    bevel.width = radius
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    normals = obj.modifiers.new("Weighted surface normals", "WEIGHTED_NORMAL")
    bpy.ops.object.modifier_apply(modifier=normals.name)
    return attach(obj, name, mat, parent)


def tube(name, start, end, radius, mat, parent=None):
    a, b = Vector(start), Vector(end)
    direction = b-a
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=direction.length, location=(a+b)/2)
    obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    return attach(obj, name, mat, parent)


def ring(name, at, radius, thickness, mat, parent=None):
    bpy.ops.mesh.primitive_torus_add(major_segments=24, minor_segments=8, location=at, major_radius=radius, minor_radius=thickness, rotation=(math.pi/2, 0, 0))
    return attach(bpy.context.object, name, mat, parent)


def glyph(name, text, at, size, mat):
    bpy.ops.object.select_all(action="DESELECT")
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = text
    curve.size = size
    curve.align_x = "CENTER"
    curve.extrude = .0006
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = at
    obj.rotation_euler = (math.pi/2, 0, 0)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    obj.select_set(False)
    return attach(obj, name, mat)


def parent_preserving_pose(obj, parent):
    bpy.context.view_layer.update()
    world = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = world
    bpy.context.view_layer.update()
    assert max(abs(obj.matrix_world[r][c]-world[r][c]) for r in range(4) for c in range(4)) < 1e-6


def fibre_lock(name, controls, width, depth, mat):
    """Closed cubic sweep with a tapered tip and broad manufactured fibre ribs.

    The tangent-aligned cross-section avoids the sphere-chain silhouette. Ribs are
    actual low-frequency geometry, not a shader that the glTF exporter drops.
    """
    points = [Vector(point) for point in controls]
    vertices, faces = [], []
    steps, sides = 12, 12
    for row in range(steps):
        t = row / steps
        center = (1-t)**3*points[0] + 3*(1-t)**2*t*points[1] + 3*(1-t)*t*t*points[2] + t**3*points[3]
        tangent = (3*(1-t)**2*(points[1]-points[0]) + 6*(1-t)*t*(points[2]-points[1]) + 3*t*t*(points[3]-points[2])).normalized()
        normal = tangent.cross(Vector((0, 1, 0))).normalized()
        binormal = tangent.cross(normal).normalized()
        radius = (.75 + .65*math.sin(math.pi*t)) * (1-t)**.65
        for column in range(sides):
            angle = 2*math.pi*column/sides
            rib = 1 + .11*math.cos(4*angle + .7*t)
            vertices.append(center + rib*radius*(width*math.cos(angle)*normal + depth*math.sin(angle)*binormal))
        if row:
            for column in range(sides):
                a = (row-1)*sides + column
                b = (row-1)*sides + (column+1)%sides
                faces.append((a, b, b+sides, a+sides))
    root, tip = len(vertices), len(vertices)+1
    vertices.extend([points[0], points[3]])
    for column in range(sides):
        nxt = (column+1)%sides
        faces.append((root, nxt, column))
        faces.append(((steps-1)*sides+column, (steps-1)*sides+nxt, tip))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return attach(obj, name, mat)


CAST = {
    "physics": ((.78, .81, .72), (.28, .78, .9), "PHYS"),
    "mathematics": ((.26, .33, .34), (.9, .65, .25), "MATH"),
    "chemistry": ((.82, .84, .76), (.65, .43, .92), "CHEM"),
    "biology": ((.29, .46, .32), (.3, .88, .61), "BIO"),
    "ai-safety": ((.23, .35, .45), (.35, .8, .95), "SAFE"),
    "machine-learning": ((.22, .25, .39), (.53, .61, 1), "ML"),
}

for field, (coat_color, glow_color, badge) in CAST.items():
    if args.field and field != args.field:
        continue
    # Factory-empty process only; no user .blend is loaded or modified.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    metal = material("Brushed titanium", (.48, .57, .59), .7, .32)
    dark = material("Graphite joints", (.04, .065, .08), .45, .38)
    face = material("Ceramic face shell", (.77, .83, .81), .35, .25)
    visor = material("Dark optical glass", (.018, .035, .043), .3, .2)
    coat = material("Discipline garment", coat_color, .02, .7)
    trim = material("Stitched trim", tuple(c*.76 for c in coat_color), .02, .68)
    light = material("AI luminous elements", glow_color, .2, .24, 2.1)
    white = material("Ivory fibre", (.87, .89, .82), 0, .75)
    brass = material("Warm instrument metal", (.62, .38, .13), .72, .3)

    for side in [-1, 1]:
        x = side*.044
        rounded("Boot sole", (x, -.009, .018), (.071, .115, .025), dark, .009)
        rounded("Boot shell", (x, -.016, .039), (.067, .094, .043), metal, .014)
        tube("Shin actuator", (x, 0, .055), (x, 0, .14), .021, metal)
        sphere("Knee joint", (x, -.003, .142), (.028, .027, .025), dark)
        rounded("Knee cap", (x, -.027, .143), (.035, .015, .029), face, .006)
        tube("Upper leg", (x, 0, .16), (x, 0, .218), .027, dark)
        rounded("Trouser panel", (x, -.022, .189), (.053, .046, .082), coat, .012)
    rounded("Pelvis housing", (0, 0, .222), (.139, .085, .068), dark, .019)
    rounded("Tailored torso", (0, 0, .286), (.171, .099, .136), coat, .025)
    rounded("Chest mechanism", (0, -.053, .288), (.05, .014, .089), dark, .007)
    ring("Core housing", (0, -.064, .29), .016, .003, metal)
    sphere("AI core", (0, -.066, .29), (.012, .005, .012), light)
    for side in [-1, 1]:
        lapel = rounded("Collar lapel", (side*.043, -.057, .325), (.043, .008, .049), face if field in ["physics", "chemistry"] else trim, .004)
        lapel.rotation_euler.y = side*-.28
        rounded("Utility pocket", (side*.056, -.054, .256), (.043, .012, .034), trim, .005)
    tube("Pen", (.059, -.065, .272), (.059, -.065, .301), .0027, light)
    glyph("Field badge", badge, (-.047, -.063, .288), .0085, light)
    tube("Neck spindle", (0, 0, .347), (0, 0, .37), .022, metal)
    ring("Neck front bearing", (0, -.022, .352), .013, .003, dark)

    # A separate semantic shoulder pivot; the application owns its pitch.
    arm = bpy.data.objects.new("gesture_arm", None)
    bpy.context.collection.objects.link(arm)
    arm.location = (.097, 0, .323)
    bpy.context.view_layer.update()
    for side in [-1, 1]:
        parent = arm if side == 1 else None
        x = side*.098
        sphere("Shoulder bearing", (x, 0, .319), (.027, .028, .027), metal, parent)
        tube("Coat sleeve", (x, 0, .313), (side*.111, 0, .255), .024, coat, parent)
        sphere("Elbow bearing", (side*.112, 0, .25), (.019, .02, .02), dark, parent)
        tube("Forearm", (side*.112, 0, .246), (side*.113, -.004, .208), .018, metal, parent)
        rounded("Mechanical palm", (side*.113, -.008, .193), (.034, .027, .035), face, .006, parent)
        for n in range(3):
            rounded("Finger", (side*.113+(n-1)*.009, -.013, .169), (.007, .018, .022), metal, .003, parent)
        sphere("Thumb", (side*.096, -.02, .192), (.009, .013, .014), metal, parent)

    rounded("Robot cranium", (0, 0, .409), (.151, .116, .111), face, .028)
    rounded("Optical visor", (0, -.06, .412), (.129, .018, .058), visor, .015)
    for side in [-1, 1]:
        rounded("Expressive eye", (side*.031, -.072, .416), (.025, .009, .019), light, .007)
        tube("Temple joint", (side*.074, 0, .405), (side*.086, 0, .405), .02, metal)
        sphere("Temple indicator", (side*.087, -.008, .405), (.005, .012, .012), light)
        for i in range(3):
            rounded("Cheek cooling vent", (side*(.039+i*.009), -.053, .377), (.004, .006, .015), dark, .001)
    rounded("Mouth speaker", (0, -.06, .375), (.043, .008, .009), dark, .003)
    for i in [-1, 0, 1]:
        rounded("Voice diode", (i*.01, -.066, .375), (.005, .003, .003), light, .001)

    if field == "physics":
        # Einstein-inspired silhouette, plainly synthetic beneath the fibre hair.
        for side in [-1, 1]:
            for i in range(7):
                # Fan from behind each temple: lower locks sweep outward,
                # upper locks rise. The optical visor stays unobstructed.
                height = .424 + i*.007
                rear = .015 + .017*math.sin(i*2.4)
                rise = -.007 + i*.005
                reach = .109 + .013*math.sin(i*1.8 + side*.4)
                fibre_lock("Wild white hair", [
                    (side*.061, rear, height),
                    (side*.087, rear-.002, height+.005),
                    (side*(reach+.009), rear-.013, height+rise-.008),
                    (side*reach, rear-.023, height+rise+.011),
                ], .012, .017, white)
            fibre_lock("Bushy white brow", [
                (side*.004, -.077, .445), (side*.025, -.082, .451),
                (side*.048, -.082, .45), (side*.064, -.074, .446),
            ], .007, .006, white)
            fibre_lock("Fibre moustache", [
                (side*.002, -.075, .386), (side*.012, -.08, .389),
                (side*.028, -.08, .38), (side*.034, -.074, .384),
            ], .006, .005, white)
        for i in range(4):
            x = (i-1.5)*.021
            fibre_lock("Crown tuft", [
                (x, .018, .46), (x-.012, .017, .48),
                (x+.013, .005, .501+(i%2)*.008), (x+.022, -.013, .495+(i%2)*.007),
            ], .009, .013, white)
    elif field == "mathematics":
        for x in [-.032, .032]:
            ring("Round spectacles", (x, -.082, .416), .025, .0023, brass)
        tube("Spectacle bridge", (-.008, -.083, .418), (.008, -.083, .418), .002, brass)
        sphere("Soft beret", (0, .004, .468), (.086, .062, .026), trim)
        sphere("Beret top", (.017, 0, .494), (.009, .01, .005), trim)
        glyph("Chalk symbol", "Σ", (.043, -.065, .305), .019, white)
    elif field == "chemistry":
        for x in [-.033, .033]:
            ring("Protective goggle rim", (x, -.082, .416), .029, .006, brass)
        tube("Goggle bridge", (-.008, -.081, .42), (.008, -.081, .42), .006, brass)
        rounded("Lab hair cap", (0, .004, .467), (.137, .101, .025), white, .012)
        for i in [-1, 1]:
            tube("Sample vial", (i*.061, -.065, .25), (i*.061, -.065, .285), .007, light)
            rounded("Vial cap", (i*.061, -.065, .287), (.017, .016, .009), dark, .002)
    elif field == "biology":
        sphere("Field hat crown", (0, .003, .47), (.072, .062, .034), trim)
        sphere("Field hat brim", (0, -.009, .46), (.103, .089, .009), coat)
        tube("Botanical antenna", (.047, .016, .47), (.065, .016, .52), .003, brass)
        leaf = sphere("Leaf antenna", (.067, .016, .511), (.022, .006, .009), light)
        leaf.rotation_euler.y = -.5
        rounded("Field scanner", (-.06, -.063, .272), (.035, .015, .054), metal, .005)
        rounded("Scanner screen", (-.06, -.073, .28), (.024, .004, .026), light, .003)
    elif field == "ai-safety":
        rounded("Protective helmet", (0, .005, .469), (.165, .12, .033), coat, .014)
        rounded("Helmet signal band", (0, -.058, .465), (.121, .008, .009), light, .003)
        shield = rounded("Evaluation shield", (0, -.073, .296), (.047, .009, .051), metal, .008)
        glyph("Shield check", "✓", (0, -.08, .281), .033, light)
        for x in [-.065, .065]:
            rounded("Safety shoulder stripe", (x, -.044, .335), (.024, .008, .013), brass, .003)
    else:
        for side in [-1, 1]:
            sphere("Headphone cup", (side*.091, .001, .414), (.017, .033, .038), coat)
            sphere("Headphone ring", (side*.102, -.008, .414), (.008, .021, .025), light)
        ring("Headphone arch", (0, .002, .428), .09, .008, dark)
        for i in range(3):
            sphere("Neural indicator", ((i-1)*.032, -.018, .472+(.01 if i==1 else 0)), (.007, .007, .007), light)
        for x in [-.023, .023]:
            tube("Hoodie cord", (x, -.063, .337), (x, -.065, .312), .0025, white)

    # Neck and eye pivots add expression without changing the authored rest pose.
    head = bpy.data.objects.new("attention_head", None)
    bpy.context.collection.objects.link(head)
    head.location = (0, 0, .362)
    head_names = {
        "Robot cranium", "Optical visor", "Expressive eye", "Temple joint",
        "Temple indicator", "Cheek cooling vent", "Mouth speaker", "Voice diode",
        "Wild white hair", "Bushy white brow", "Fibre moustache", "Crown tuft",
        "Round spectacles", "Spectacle bridge", "Soft beret", "Beret top",
        "Protective goggle rim", "Goggle bridge", "Lab hair cap", "Field hat crown",
        "Field hat brim", "Botanical antenna", "Leaf antenna", "Protective helmet",
        "Helmet signal band", "Headphone cup", "Headphone ring", "Headphone arch",
        "Neural indicator",
    }
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and obj.name.split(".")[0] in head_names:
            parent_preserving_pose(obj, head)
            if obj.name.startswith("Expressive eye"):
                world_position = obj.matrix_world.translation.copy()
                eye = bpy.data.objects.new("eye_left" if world_position.x < 0 else "eye_right", None)
                bpy.context.collection.objects.link(eye)
                eye.location = world_position
                parent_preserving_pose(eye, head)
                parent_preserving_pose(obj, eye)

    # Join by material and animation parent: keeps the authored shapes but avoids
    # hundreds of draw calls. Each output primitive retains its PBR identity.
    bpy.ops.object.select_all(action="DESELECT")
    groups = {}
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH":
            groups.setdefault((obj.parent, obj.data.materials[0].name), []).append(obj)
    for (parent, mat_name), objects in groups.items():
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        if len(objects) > 1:
            bpy.ops.object.join()
        objects[0].name = (parent.name + " " if parent else "Body ") + mat_name
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.wm.save_as_mainfile(filepath=str(source / f"{field}-robot.blend"))
    bpy.ops.export_scene.gltf(filepath=str(output / f"{field}-robot.glb"), export_format="GLB", use_selection=True, export_cameras=False, export_lights=False, export_animations=False, export_yup=True, export_apply=True)
    print(f"ROBOT_EXPORTED {field} {(output / (field+'-robot.glb')).stat().st_size} bytes")
