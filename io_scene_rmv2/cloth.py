"""Cloth: a weighted cloth's own skeleton as an armature, and back.

A weighted cloth (material 58) is skinned, but not to the model's
skeleton: its material block carries a skeleton of its own, one bone per
particle of the cloth (material 60) that drives it (see
rmv2_format.decode_cloth_skeleton).  The game rebuilds each bone from its
particle every frame and skins the drawn mesh with the bind matrices in
the file, so those matrices have to move whenever the mesh does - kept
raw, a moved sail is skinned from where it used to be.

In Blender that skeleton is an ordinary armature with the mesh bound to
it, and the cloth parented to it too, so moving the armature moves the
whole piece and export reads the bind matrices back off the bones.  A
bone that is still where the file put it writes the file's own bytes.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import rmv2_format as rf
from . import skeleton
from . import utils

SKELETON_PROP = skeleton.CLOTH_SKELETON_PROP
is_cloth_skeleton = skeleton.is_cloth_skeleton

_DEFAULT_BONE_LENGTH = 0.1


def _bone_frames(skel: rf.ClothSkeleton) -> np.ndarray:
    """(n, 4, 4) bone -> game frames, the inverses of the bind matrices."""
    inv_bind = skel.bones["inv_bind"].astype(np.float64)
    frames = np.tile(np.eye(4), (len(inv_bind), 1, 1))
    frames[:, :3, :] = inv_bind
    return np.linalg.inv(frames)


def bone_origins(skel: rf.ClothSkeleton) -> np.ndarray:
    """(n, 3) game-space position of every bone."""
    return _bone_frames(skel)[:, :3, 3]


def build_armature(context, name: str, skel: rf.ClothSkeleton,
                   scale: float, collection):
    """The skeleton as an armature linked into `collection`."""
    conv = utils.GAME_TO_BLENDER_M3.astype(np.float64)
    frames = _bone_frames(skel)
    world = []
    for frame in frames:
        m = np.eye(4)
        # Conjugating by the reflection keeps the rotation proper.
        m[:3, :3] = conv @ frame[:3, :3] @ conv.T
        m[:3, 3] = conv @ frame[:3, 3] * scale
        world.append(Matrix(m.tolist()))

    # Long enough to see, short of reaching the next bone.
    heads = np.array([m.translation[:] for m in world])
    length = _DEFAULT_BONE_LENGTH * scale
    if len(heads) > 1:
        gaps = np.linalg.norm(heads[:, None] - heads[None], axis=2)
        np.fill_diagonal(gaps, np.inf)
        nearest = float(np.median(gaps.min(axis=1)))
        if np.isfinite(nearest) and nearest > 1e-4:
            length = nearest * 0.5

    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    collection.objects.link(arm_obj)

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    names = []
    try:
        for k, matrix in enumerate(world):
            eb = arm_data.edit_bones.new(skel.bone_name(k) or f"bone_{k}")
            eb.head = (0.0, 0.0, 0.0)
            eb.tail = (0.0, length, 0.0)
            eb.matrix = matrix
            names.append(eb.name)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    for k, bone_name in enumerate(names):
        arm_data.bones[bone_name][skeleton.BONE_INDEX_PROP] = k
    arm_data[SKELETON_PROP] = True
    arm_data.display_type = "STICK"
    arm_obj.show_in_front = True
    return arm_obj


def bone_names(arm_obj) -> dict:
    return skeleton.bone_name_by_index(arm_obj)


def parent_keep_transform(obj, parent) -> None:
    """Parent `obj` to `parent` without moving it."""
    obj.parent = parent
    obj.parent_type = "OBJECT"
    obj.matrix_parent_inverse = parent.matrix_world.inverted_safe()


def drives(skel: rf.ClothSkeleton, cloth: rf.ClothGraph,
           cloth_positions: np.ndarray, tolerance: float = 0.02) -> bool:
    """Whether `cloth` is the cloth this skeleton rides on: one bone per
    particle, each sitting on its particle (to half precision - the
    vertices are stored at half precision and the bones are not)."""
    particles = skel.bones["particle"].astype(np.int64)
    if (len(particles) != cloth.particle_count
            or cloth.particle_count > len(cloth_positions)
            or int(particles.max()) >= cloth.particle_count):
        return False
    gaps = np.linalg.norm(bone_origins(skel)
                          - cloth_positions[particles], axis=1)
    return bool(gaps.max() <= tolerance)


def find_armature(obj):
    for mod in obj.modifiers:
        if mod.type == "ARMATURE" and is_cloth_skeleton(mod.object):
            return mod.object
    if is_cloth_skeleton(obj.parent):
        return obj.parent
    return None


def warn_if_moved_apart(obj, warnings: list) -> None:
    """A weighted cloth or cloth moved on its own, rather than with its
    skeleton, no longer lines up with the bones: the drawn mesh is then
    skinned to particles somewhere else, and twists apart as soon as the
    cloth moves.  That is almost always a sail grabbed instead of its
    armature, so it is worth saying."""
    arm = find_armature(obj) or (obj.parent
                                 if is_cloth_skeleton(obj.parent) else None)
    if arm is None:
        return
    relative = arm.matrix_world.inverted_safe() @ obj.matrix_world
    if np.allclose(np.array(relative), np.eye(4), rtol=0.0, atol=1e-5):
        return
    warnings.append(
        f"{obj.name} has been moved apart from its cloth skeleton "
        f"'{arm.name}'. The game rebuilds that skeleton from the cloth's "
        "particles, so the sail will tear away from where it is drawn. "
        "Move the skeleton instead - the sail and its cloth both follow "
        "it. Clearing this object's location, rotation and scale "
        "(Alt+G, Alt+R, Alt+S) puts it back on the skeleton")


def _inverse_rigid(rotation: np.ndarray, translation: np.ndarray):
    out = np.zeros((3, 4))
    out[:, :3] = rotation.T
    out[:, 3] = -rotation.T @ translation
    return out


def _keep_if_close(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """The file's own matrix while the bone has not really moved: the
    bone's rest matrix goes through head, tail and roll on the way in and
    out, which a float32 bind matrix survives only to within rounding."""
    tolerance = 1e-4 * max(1.0, float(np.abs(old[:, 3]).max()))
    if np.allclose(new, old, rtol=0.0, atol=tolerance):
        return old
    return new.astype(np.float32)


def write_skeleton(mesh_obj, original: rf.ClothSkeleton, scale: float,
                   warnings: list) -> rf.ClothSkeleton:
    """The skeleton to write for `mesh_obj`: the file's, with each bind
    matrix and name read back off the armature it is bound to.

    Without that armature (a scene from before it existed, or one where
    it was deleted) the object's own transform moves the skeleton
    instead, on the assumption that the object was imported where the
    file put it - which for every vanilla cloth is the origin."""
    conv = utils.GAME_TO_BLENDER_M3.astype(np.float64)
    out = rf.ClothSkeleton(bones=original.bones.copy(),
                           unknown=original.unknown)
    bones = out.bones
    arm = find_armature(mesh_obj)

    if arm is None:
        loc, rot, _ = mesh_obj.matrix_world.decompose()
        moved_r = conv.T @ np.array(rot.to_matrix(), np.float64) @ conv
        moved_t = conv.T @ np.array(loc[:], np.float64) * scale
        if np.allclose(moved_r, np.eye(3), atol=1e-7) and \
                np.allclose(moved_t, 0.0, atol=1e-7):
            return out
        for k, frame in enumerate(_bone_frames(original)):
            rotation = moved_r @ frame[:3, :3]
            translation = moved_r @ frame[:3, 3] + moved_t
            bones["inv_bind"][k] = _inverse_rigid(rotation, translation)
        return out

    by_index = dict(skeleton.bone_index_pairs(arm))
    missing = [k for k in range(len(bones)) if k not in by_index]
    if missing:
        warnings.append(
            f"{mesh_obj.name}: the cloth skeleton '{arm.name}' has no bone "
            f"for {len(missing)} of the {len(bones)} particles; those keep "
            "their bind matrices from the file")
    extra = sorted(set(by_index) - set(range(len(bones))))
    if extra:
        warnings.append(
            f"{mesh_obj.name}: the cloth skeleton '{arm.name}' has "
            f"{len(extra)} more bones than the cloth has particles; the "
            "extra bones are not written - a weighted cloth needs exactly "
            "one bone per particle")

    for k in range(len(bones)):
        bone = by_index.get(k)
        if bone is None:
            continue
        # The game rebuilds the frame from the particles, so it is never
        # scaled: an armature's scale lands in the positions, not here.
        loc, rot, _ = (arm.matrix_world @ bone.matrix_local).decompose()
        rotation = conv.T @ np.array(rot.to_matrix(), np.float64) @ conv
        translation = conv.T @ np.array(loc[:], np.float64) * scale
        bones["inv_bind"][k] = _keep_if_close(
            _inverse_rigid(rotation, translation),
            original.bones["inv_bind"][k].astype(np.float64))
        bones["name"][k] = np.void(rf.encode_cloth_bone_name(
            bone.name, bytes(original.bones["name"][k])))
    return out


def write_cloth(original: rf.ClothGraph, positions: np.ndarray,
                obj_name: str, warnings: list) -> rf.ClothGraph:
    """The cloth graph to write, with each rest length that no longer
    matches its edge measured again.

    `positions` are the vertices about to be written, in game space.
    Rest lengths are the authored lengths and the vertices are stored at
    half precision, so an edge is only measured again when it has moved
    by more than that precision can account for - an untouched cloth
    writes the file's own numbers."""
    out = rf.ClothGraph(edges=original.edges.copy(),
                        particle_words=original.particle_words.copy(),
                        triangles=original.triangles.copy())
    p = original.particle_count
    if len(positions) < p:
        warnings.append(
            f"{obj_name}: the cloth has {len(positions)} vertices but its "
            f"simulation names {p} particles; the simulation is written "
            "as it was imported and will not match the mesh")
        return out
    pos = np.asarray(positions[:p], np.float64)
    i = out.edges["i"].astype(np.int64)
    j = out.edges["j"].astype(np.int64)
    measured = np.linalg.norm(pos[i] - pos[j], axis=1)
    # A half float has 10 fraction bits; two endpoints each off by half a
    # step, along three axes, stay inside this.
    tolerance = float(np.abs(pos).max(initial=1.0)) * 2.0 ** -9
    moved = np.abs(measured - out.edges["rest"]) > tolerance
    out.edges["rest"][moved] = measured[moved]
    return out


def write_collider(original: rf.SphereCollider, obj, extra: dict,
                   scale: float) -> rf.SphereCollider:
    """The sphere, carried by the object's transform from where the
    importer stored it in the object's own space."""
    centre_local = extra.get("collider_centre")
    radius_local = extra.get("collider_radius")
    if centre_local is None or radius_local is None:
        return original
    world = obj.matrix_world
    centre_b = world @ Vector(centre_local)
    centre = np.array(utils.blender_to_game_v(centre_b), np.float64) * scale
    m3 = np.array(world.to_3x3(), np.float64)
    radius = (float(radius_local) * abs(np.linalg.det(m3)) ** (1.0 / 3.0)
              * scale)
    old_centre = np.array(original.centre, np.float64)
    tolerance = 1e-5 * max(1.0, float(np.abs(old_centre).max()))
    out = rf.SphereCollider(centre=original.centre, radius=original.radius,
                            unknown=original.unknown)
    if not np.allclose(centre, old_centre, rtol=0.0, atol=tolerance):
        out.centre = tuple(float(v) for v in centre)
    if abs(radius - original.radius) > 1e-5 * max(1.0, original.radius):
        out.radius = radius
    return out
