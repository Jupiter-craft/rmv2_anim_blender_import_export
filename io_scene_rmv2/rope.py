"""Ropes: a chain of simulation nodes in Blender, and back.

A rope mesh ships no vertices - the game sweeps a twelve-sided tube along
the node chain in its material block (see rmv2_format.decode_rope).  In
Blender the chain is a mesh of loose vertices and edges, one vertex per
node, so it can be edited like any other polyline; the per-node inverse
mass and texture coordinate ride along as point attributes.  A Geometry
Nodes modifier sweeps the same tube the game does, purely for display:
export reads the chain off the object's own mesh, never the evaluated
one, so the preview is never written as geometry.
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


def preview_radius(obj):
    """The radius set on the object's preview modifier, if it has one."""
    for mod in obj.modifiers:
        if (mod.type == "NODES" and mod.node_group is not None
                and mod.node_group.name.startswith(PREVIEW_GROUP)):
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


def read_graph(obj, scale: float, original, warnings: list):
    """The rope graph to write for `obj`, or None if its mesh is not a
    chain.  `original` is the graph it was imported with, if any: what
    the edit left alone comes back from it exactly, so an untouched rope
    re-exports byte for byte."""
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

    # Rotation and scale go into the positions; the translation is the
    # pivot, written separately, exactly as for a mesh's vertices.
    m3 = np.array(obj.matrix_world, np.float64)[:3, :3]
    positions = utils.blender_to_game(
        (co.reshape(-1, 3).astype(np.float64) @ m3.T)) * scale
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
