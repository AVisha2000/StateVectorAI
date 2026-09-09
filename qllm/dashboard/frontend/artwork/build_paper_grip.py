"""Author a static paper-holding hand in Blender (CPU, no external assets).

Blender --background --factory-startup --python this_file -- --output FILE --source FILE
The exported module is synchronous, texture-free indexed geometry: no new loader,
request, rig or transform owner. Coordinates are paper-local Three.js Y-up units.
Only the source .blend stays outside the frontend. Regenerate, do not hand-edit.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--source", required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def xyz(p):
    return Vector((p[0], -p[2], p[1]))


def ellipsoid(name, at, radii):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=xyz(at))
    obj = bpy.context.object
    obj.name = name
    obj.scale = (radii[0], radii[2], radii[1])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def link(name, start, end, radius):
    a, b = xyz(start), xyz(end)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, location=(a+b)/2)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (radius, radius, (b-a).length/2+radius)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (b-a).to_track_quat("Z", "Y")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


skin = [ellipsoid("Palm", (.112, -.019, .095), (.021, .014, .033)),
        ellipsoid("Wrist", (.110, -.018, .131), (.016, .013, .022))]
# Four different-length fingers curl behind the sheet; gaps remain at the tips.
for i, (z, reach) in enumerate([(.065, .059), (.080, .050), (.095, .056), (.109, .071)]):
    base = (.113, -.020, z)
    knuckle = (.102, -.027, z-.002)
    tip = (reach, -.009, z-.005)
    skin.extend([link(f"Finger {i+1} base", base, knuckle, .0065),
                 link(f"Finger {i+1} curl", knuckle, tip, .006)])
# Opposing thumb comes around the outside edge and rests on the paper's face.
thumb = [(.125, -.007, .106), (.121, .0004, .091), (.107, .0074, .077), (.078, -.0006, .071)]
for i in range(len(thumb)-1):
    skin.append(link(f"Thumb {i+1}", thumb[i], thumb[i+1], .0085-i*.0008))

bpy.ops.object.select_all(action="DESELECT")
for obj in skin:
    obj.select_set(True)
bpy.context.view_layer.objects.active = skin[0]
bpy.ops.object.join()
hand = bpy.context.object
hand.name = "Sculpted palm and five digits"
remesh = hand.modifiers.new("Unify the hand surface", "REMESH")
remesh.mode = "VOXEL"
remesh.voxel_size = .0013
bpy.ops.object.modifier_apply(modifier=remesh.name)
smooth = hand.modifiers.new("Soften palm transitions", "SMOOTH")
smooth.factor = .7
smooth.iterations = 5
bpy.ops.object.modifier_apply(modifier=smooth.name)
decimate = hand.modifiers.new("Browser mesh", "DECIMATE")
decimate.ratio = .16
bpy.ops.object.modifier_apply(modifier=decimate.name)


def sleeve(name, profile):
    vertices, faces = [], []
    sides = 32
    for z, rx, ry in profile:
        for j in range(sides):
            angle = 2*math.pi*j/sides
            # Restrained longitudinal cloth folds, rounded rather than a box.
            fold = 1+.025*math.cos(6*angle)
            vertices.append(xyz((.110+rx*math.cos(angle)*fold,
                                 -.018+ry*math.sin(angle)*fold, z)))
    for ring in range(len(profile)-1):
        for j in range(sides):
            a, b = ring*sides+j, ring*sides+(j+1)%sides
            faces.append((a, b, b+sides, a+sides))
    faces.extend([tuple(reversed(range(sides))),
                  tuple((len(profile)-1)*sides+j for j in range(sides))])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


cuff = sleeve("Ribbed knit cuff", [(.130,.016,.013),(.132,.018,.015),
    (.135,.0185,.0155),(.138,.018,.015),(.141,.0185,.0155),(.144,.019,.016)])
coat = sleeve("Soft blue sleeve", [(.141,.0185,.0155),(.145,.022,.018),
    (.150,.0225,.0185),(.154,.023,.019),(.158,.024,.020),(.160,.022,.018)])
nail = ellipsoid("Thumbnail", (.083,.0059,.0715), (.0065,.0016,.0042))

parts = [(hand, "#e0b591", .72), (cuff, "#537d89", .96),
         (coat, "#86afba", .91), (nail, "#e8c8ae", .48)]
exported = []
for obj, color, roughness in parts:
    mat = bpy.data.materials.new(obj.name)
    mat.diffuse_color = (*[int(color[i:i+2],16)/255 for i in (1,3,5)],1)
    mat.roughness = roughness
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.calc_loop_triangles()
    world = obj.matrix_world
    normals = world.to_3x3().inverted().transposed()
    positions, normal_values = [], []
    for vertex in obj.data.vertices:
        p = world @ vertex.co
        n = (normals @ vertex.normal).normalized()
        positions.extend(round(v,6) for v in (p.x,p.z,-p.y))
        normal_values.extend(round(v,6) for v in (n.x,n.z,-n.y))
    indices = [i for tri in obj.data.loop_triangles for i in tri.vertices]
    exported.append(dict(name=obj.name,color=color,roughness=roughness,
                         positions=positions,normals=normal_values,indices=indices))

output, source = Path(args.output), Path(args.source)
output.parent.mkdir(parents=True, exist_ok=True)
source.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(source))
payload = dict(author="StateVectorAI / Blender 4.5 LTS",frame="paper-local Y-up",parts=exported)
output.write_text("// Generated by artwork/build_paper_grip.py. Do not hand-edit.\nexport default "
                  +json.dumps(payload,separators=(",", ":"))+";\n",encoding="utf-8")
print(json.dumps({"output":str(output),"source":str(source),
                  "triangles":sum(len(p["indices"])//3 for p in exported),
                  "bytes":output.stat().st_size}))
