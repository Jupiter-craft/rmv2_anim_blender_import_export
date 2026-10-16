"""Ropes: a chain of simulation nodes in Blender, and back.

A rope mesh ships no vertices - the game sweeps a twelve-sided tube along
the node chain in its material block (see rmv2_format.decode_rope).  In
Blender the chain is a mesh of loose vertices and edges, one vertex per
node, so it can be edited like any other polyline; the per-node inverse
mass and texture coordinate ride along as point attributes.  A Geometry
Nodes modifier sweeps the same tube the game does, purely for display:
export switches it off before reading the chain, so the preview is never
written as geometry.

A rope does not hang from a bone.  Each end the file attaches names a
particle - a vertex of one of the LOD's cloths (60) - and the game holds
that end of the rope on it.  So in Blender an attached end is hooked to
the cloth object, easing off along the chain to nothing at the far end:
move a sail's skeleton and the cloth goes with it, dragging the rope.
Export reads the hooked chain, and then puts any attached end that is
still not on its particle there (a scene imported before the hooks, or
a hook removed), easing the rest of the rope after it the same way.
"""

from __future__ import annotations

import bpy
import numpy as np

from . import rmv2_format as rf
from . import utils

INV_MASS_ATTRIBUTE = "rmv2_rope_inv_mass"
U_ATTRIBUTE = "rmv2_rope_u"
PREVIEW_GROUP = "RMV2 Rope Preview"
PREVIEW_MODIFIER = "Rope Preview"
HOOK_GROUPS = ("rmv2_rope_start", "rmv2_rope_end")


def build_mesh(name: str, rope: rf.RopeGraph, scale: float):
    """The node chain as a Blender mesh of vertices and edges."""
    positions = utils.game_to_blender(rope.nodes["position"]) * scale
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(positions))
    me.vertices.foreach_set(
        "co", np.ascontiguousarray(positions, np.float32).ravel())
    edges = np.stack([rope.links["i"], rope.links["j"]], axis=1)
    me.edges.add(len(edges))
    me.edges.foreach_set("vertices",
                         np.ascontiguousarray(edges, np.int32).ravel())
    for attr_name, field in ((INV_MASS_ATTRIBUTE, "inv_mass"),
                             (U_ATTRIBUTE, "u")):
        attr = me.attributes.new(name=attr_name, type="FLOAT",
                                 domain="POINT")
        attr.data.foreach_set(
            "value", np.ascontiguousarray(rope.nodes[field], np.float32))
    me.validate(verbose=False, clean_customdata=False)
    return me


def _preview_group():
    group = bpy.data.node_groups.get(PREVIEW_GROUP)
    if group is not None:
        return group
    group = bpy.data.node_groups.new(PREVIEW_GROUP, "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT",
                               socket_type="NodeSocketGeometry")
    radius = group.interface.new_socket("Radius", in_out="INPUT",
                                        socket_type="NodeSocketFloat")
    radius.default_value = 0.03125
    radius.min_value = 0.0
    group.interface.new_socket("Geometry", in_out="OUTPUT",
                               socket_type="NodeSocketGeometry")

    nodes, links = group.nodes, group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    to_curve = nodes.new("GeometryNodeMeshToCurve")
    circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
    circle.mode = "RADIUS"
    circle.inputs["Resolution"].default_value = 12  # as the game does
    sweep = nodes.new("GeometryNodeCurveToMesh")
    sweep.inputs["Fill Caps"].default_value = True
    for i, node in enumerate((group_in, to_curve, circle, sweep,
                              group_out)):
        node.location = (200.0 * i, 0.0)

    links.new(group_in.outputs["Geometry"], to_curve.inputs["Mesh"])
    links.new(group_in.outputs["Radius"], circle.inputs["Radius"])
    links.new(to_curve.outputs["Curve"], sweep.inputs["Curve"])
    links.new(circle.outputs["Curve"], sweep.inputs["Profile Curve"])
    links.new(sweep.outputs["Mesh"], group_out.inputs["Geometry"])
    return group


def _radius_input(mod):
    """(get, set) for the modifier's Radius input.  Blender 5 moved
    modifier inputs from ID properties on the modifier itself to
    `mod.properties.inputs`; 4.x still has only the former."""
    identifier = mod.node_group.interface.items_tree["Radius"].identifier
    properties = getattr(mod, "properties", None)
    socket = getattr(getattr(properties, "inputs", None), identifier, None)
    if socket is not None:
        return (lambda: socket.value,
                lambda value: setattr(socket, "value", value))
    return (lambda: mod[identifier],
            lambda value: mod.__setitem__(identifier, value))


def add_preview(obj, radius: float) -> None:
    """Sweep the tube the game would, as a display-only modifier."""
    mod = obj.modifiers.new(PREVIEW_MODIFIER, "NODES")
    mod.node_group = _preview_group()
    _radius_input(mod)[1](float(radius))


def is_preview(mod) -> bool:
    return (mod.type == "NODES" and mod.node_group is not None
            and mod.node_group.name.startswith(PREVIEW_GROUP))


def preview_radius(obj):
    """The radius set on the object's preview modifier, if it has one."""
    for mod in obj.modifiers:
        if is_preview(mod):
            try:
                return float(_radius_input(mod)[0]())
            except (KeyError, TypeError):
                return None
    return None


def is_rope_object(obj) -> bool:
    """An object export should write as a rope: one whose material is the
    rope material and whose mesh is a chain with no faces."""
    if obj.type != "MESH" or len(obj.data.polygons):
        return False
    settings = obj.rmv2
    if settings.material_id == "OTHER":
        return settings.material_id_raw == rf.MAT_ROPE
    return rf.MATERIAL_IDS.get(settings.material_id) == rf.MAT_ROPE


def _chain_order(count: int, edges: np.ndarray):
    """The vertices of a single unbranched chain, end to end, starting
    from the lower-numbered end - or None if the edges are anything else.
    Walking from the lower end keeps an imported rope in its file order,
    and a vertex extruded off either end lands where it belongs."""
    if count < 2 or len(edges) != count - 1:
        return None
    neighbours = [[] for _ in range(count)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    if any(len(n) > 2 or not n for n in neighbours):
        return None
    ends = [v for v in range(count) if len(neighbours[v]) == 1]
    if len(ends) != 2:
        return None
    order = [min(ends)]
    previous = -1
    while len(order) < count:
        here = order[-1]
        step = [v for v in neighbours[here] if v != previous]
        if not step:
            return None
        previous = here
        order.append(step[0])
    return np.array(order, np.int64)


def _point_floats(me, name: str):
    attr = me.attributes.get(name)
    if attr is None or attr.domain != "POINT" or attr.data_type != "FLOAT":
        return None
    values = np.empty(len(me.vertices), np.float32)
    attr.data.foreach_get("value", values)
    return values


def falloff(positions: np.ndarray, from_start: bool) -> np.ndarray:
    """How much of a pull on one end of the chain each node takes: all
    of it at that end, none at the other, in between by distance along
    the rope - so a moved end drags the rope without kinking it."""
    segment = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    run = np.concatenate([[0.0], np.cumsum(segment)])
    if run[-1] > 0.0:
        along = run / run[-1]
    else:
        along = np.linspace(0.0, 1.0, len(positions))
    return 1.0 - along if from_start else along


def _attached_ends(rope: rf.RopeGraph):
    """(vertex, cloth, from_start) for each end hung from a cloth."""
    for (vertex, cloth), from_start in ((rope.start_attachment, True),
                                        (rope.end_attachment, False)):
        if vertex >= 0 and cloth >= 0:
            yield int(vertex), int(cloth), from_start


def hook_to_cloths(obj, rope: rf.RopeGraph, cloth_objs: list) -> None:
    """Hook each attached end of the rope to the cloth it hangs from.
    `cloth_objs` are the LOD's cloths in file order, which is what the
    attachment's cloth number counts.  Nothing moves now: the hook takes
    the cloth's transform from here on."""
    positions = rope.nodes["position"].astype(np.float64)
    hooks = []
    for vertex, cloth, from_start in _attached_ends(rope):
        if cloth >= len(cloth_objs):
            continue
        target = cloth_objs[cloth]
        group = obj.vertex_groups.new(
            name=HOOK_GROUPS[0 if from_start else 1])
        for k, weight in enumerate(falloff(positions, from_start)):
            if weight > 0.0:
                group.add([k], float(weight), "REPLACE")
        mod = obj.modifiers.new(f"Hang from {target.name}", "HOOK")
        mod.object = target
        mod.vertex_group = group.name
        mod.falloff_type = "NONE"
        mod.strength = 1.0
        mod.matrix_inverse = (target.matrix_world.inverted_safe()
                              @ obj.matrix_world)
        hooks.append(mod.name)
    # The hooks bend the chain and the preview sweeps the bent one.
    for k, name in enumerate(hooks):
        obj.modifiers.move(obj.modifiers.find(name), k)


def follow_cloths(models: list, warnings: list) -> None:
    """Put every attached rope end in a LOD's models on its particle, as
    written: an end left behind by a moved sail is tied to a particle
    somewhere else, and the game's rope - its links still the old
    lengths - hauls the sail back toward it."""
    cloths = [m for m in models
              if getattr(m.material, "material_id", None) == rf.MAT_CLOTH]
    for model in models:
        material = model.material
        if getattr(material, "material_id", None) != rf.MAT_ROPE:
            continue
        rope = rf.decode_rope(getattr(material, "trailing", b""))
        if rope is None:
            continue
        positions = rope.nodes["position"].astype(np.float64)
        pulled = False
        for vertex, cloth, from_start in _attached_ends(rope):
            if (cloth >= len(cloths)
                    or vertex >= cloths[cloth].mesh.vertex_count):
                warnings.append(
                    f"{material.model_name}: hangs from particle {vertex} "
                    f"of cloth {cloth}, which this LOD does not have")
                continue
            target = cloths[cloth].mesh.positions[vertex].astype(np.float64)
            gap = target - positions[0 if from_start else -1]
            # The particle is written at half precision; the rope is not.
            tolerance = max(0.02, float(np.abs(target).max()) * 2.0 ** -9)
            distance = float(np.linalg.norm(gap))
            if distance <= tolerance:
                continue
            positions += falloff(positions, from_start)[:, None] * gap
            pulled = True
            warnings.append(
                f"{material.model_name}: its "
                f"{'start' if from_start else 'end'} hangs from particle "
                f"{vertex} of {cloths[cloth].material.model_name}, which "
                f"is {distance:.3g} away from it, so the rope was pulled "
                "after it. Import the file again to see ropes follow "
                "their sails in Blender")
        if not pulled:
            continue
        rope.nodes["position"] = positions
        written = rope.nodes["position"].astype(np.float64)
        rope.links["rest"] = np.linalg.norm(
            written[rope.links["j"]] - written[rope.links["i"]], axis=1)
        material.trailing = rope.write()
        model.bbox_min = tuple((written.min(axis=0) - rope.radius).tolist())
        model.bbox_max = tuple((written.max(axis=0) + rope.radius).tolist())


def read_graph(obj, scale: float, original, warnings: list, me=None):
    """The rope graph to write for `obj`, or None if its mesh is not a
    chain.  `me` is the chain to read, the object's own mesh by default.
    `original` is the graph it was imported with, if any: what the edit
    left alone comes back from it exactly, so an untouched rope
    re-exports byte for byte."""
    if me is None:
        me = obj.data
    count = len(me.vertices)
    co = np.empty(count * 3, np.float32)
    me.vertices.foreach_get("co", co)
    edges = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", edges)
    order = _chain_order(count, edges.reshape(-1, 2))
    if order is None:
        warnings.append(
            f"{obj.name}: a rope must be a single chain of vertices "
            f"joined end to end by edges ({count} vertices, "
            f"{len(me.edges)} edges)")
        return None

    # The whole transform goes into the positions.  The game ignores a
    # rope's pivot, like every physics material's, so the export writes
    # it as zero and the translation has to be here.
    mw = np.array(obj.matrix_world, np.float64)
    positions = utils.blender_to_game(
        co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T
        + mw[:3, 3]) * scale
    positions = positions[order].astype(np.float32)

    inv_mass = _point_floats(me, INV_MASS_ATTRIBUTE)
    inv_mass = (inv_mass[order] if inv_mass is not None
                else np.ones(count, np.float32))
    segment = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    u = _point_floats(me, U_ATTRIBUTE)
    u = (u[order] if u is not None
         else np.concatenate([[0.0], np.cumsum(segment)]).astype(np.float32))

    nodes = np.zeros(count, rf._ROPE_NODE)
    nodes["position"] = positions
    nodes["inv_mass"] = inv_mass
    nodes["u"] = u
    links = np.zeros(count - 1, rf._ROPE_LINK)
    links["i"] = np.arange(count - 1)
    links["j"] = np.arange(1, count)
    links["rest"] = segment

    graph = rf.RopeGraph(nodes=nodes, links=links)
    if original is not None:
        graph.start_attachment = original.start_attachment
        graph.end_attachment = original.end_attachment
        graph.radius = original.radius
        graph.unknown = original.unknown
        _keep_untouched(graph, original, order)
    radius = preview_radius(obj)
    if radius is not None:
        radius = np.float32(radius * scale)
        if not np.isclose(radius, graph.radius, rtol=0, atol=1e-7):
            graph.radius = float(radius)
    return graph


def _keep_untouched(graph, original, order) -> None:
    """Put back the original values wherever the edit left them alone.
    Positions pass through a reflection and a float64 matrix product on
    the way in and out, which a node that never moved survives only to
    within rounding; the rest lengths in the file are the authored ones,
    not quite what float32 arithmetic recomputes from the positions."""
    old_pos = original.nodes["position"]
    new_pos = graph.nodes["position"]
    same = np.zeros(len(order), bool)
    for k, v in enumerate(order):
        if v < len(old_pos) and np.allclose(new_pos[k], old_pos[v], rtol=0,
                                            atol=1e-5):
            new_pos[k] = old_pos[v]
            same[k] = True
    rest = {(int(a), int(b)): float(r) for a, b, r in
            zip(original.links["i"], original.links["j"],
                original.links["rest"])}
    for k in range(len(order) - 1):
        if not (same[k] and same[k + 1]):
            continue
        a, b = int(order[k]), int(order[k + 1])
        r = rest.get((a, b), rest.get((b, a)))
        if r is not None:
            graph.links["rest"][k] = r
