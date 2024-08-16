from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Pln, gp_Vec, gp_Ax3, gp_Ax2, gp_Lin
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from copy import copy
from .extrude import *
from .sketch import Loop, Profile
from .curves import *
import os
import trimesh
from trimesh.sample import sample_surface
import random

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen
from OCC.Core.AIS import AIS_Shape

# 创建不同颜色
red = Quantity_Color(1.0, 0.1, 0.1, Quantity_TOC_RGB)
green = Quantity_Color(0.1, 1.0, 0.1, Quantity_TOC_RGB)
blue = Quantity_Color(0.1, 0.1, 1.0, Quantity_TOC_RGB)
white = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
gray = Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)

import random
def generate_random_color():
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)
import numpy as np
def color_distance(color1, color2):
    rgb1 = np.array([color1.Red(), color1.Green(), color1.Blue()])
    rgb2 = np.array([color2.Red(), color2.Green(), color2.Blue()])
    return np.linalg.norm(rgb1 - rgb2)

def vec2CADsolid(vec, is_numerical=True, n=256):
    cad = CADSequence.from_vector(vec, is_numerical=is_numerical, n=256)
    cad = create_CAD(cad)
    return cad


from copy import deepcopy
def create_CAD_index(doc: TDocStd_Document, cad_seq: CADSequence, index = 0, color = red):
    """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
    if len(cad_seq.seq) != 1:
        _ = create_CAD(doc, cad_seq)
    extrude_op = cad_seq.seq[index]
    profile = copy(extrude_op.profile) # use copy to prevent changing extrude_op internally
    profile.denormalize(extrude_op.sketch_size)

    sketch_plane = copy(extrude_op.sketch_plane)
    #sketch_plane.origin = extrude_op.sketch_pos

    face = create_profile_face(profile, sketch_plane)
    normal = gp_Dir(*extrude_op.sketch_plane.normal)
    ext_vec = gp_Vec(normal).Multiplied(extrude_op.extent_one)
    body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    label = shape_tool.AddShape(body)
    color_tool.SetColor(label, red, XCAFDoc_ColorGen)
    
    if extrude_op.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
        body_sym = BRepPrimAPI_MakePrism(face, ext_vec.Reversed()).Shape()
        body = BRepAlgoAPI_Fuse(body, body_sym).Shape()
        label_sym = shape_tool.AddShape(body_sym)
        color_tool.SetColor(label_sym, red, XCAFDoc_ColorGen)
    if extrude_op.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
        ext_vec = gp_Vec(normal.Reversed()).Multiplied(extrude_op.extent_two)
        body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        body = BRepAlgoAPI_Fuse(body, body_two).Shape()
        label_two = shape_tool.AddShape(body_two)
        color_tool.SetColor(label_two, red, XCAFDoc_ColorGen)
    return body

def create_CAD(doc: TDocStd_Document, cad_seq: CADSequence):
    """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
    body = create_by_extrude(doc, cad_seq.seq[0])

    for extrude_op in cad_seq.seq[1:]:
        new_body = create_by_extrude(doc, extrude_op)

        if extrude_op.operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation") or \
                extrude_op.operation == EXTRUDE_OPERATIONS.index("JoinFeatureOperation"):
            body = BRepAlgoAPI_Fuse(body, new_body).Shape()
        elif extrude_op.operation == EXTRUDE_OPERATIONS.index("CutFeatureOperation"):
            body = BRepAlgoAPI_Cut(body, new_body).Shape()
        elif extrude_op.operation == EXTRUDE_OPERATIONS.index("IntersectFeatureOperation"):
            body = BRepAlgoAPI_Common(body, new_body).Shape()

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    _ = shape_tool.AddShape(body)
    return body


def create_by_extrude(doc: TDocStd_Document, extrude_op: Extrude):
    """create a solid body from Extrude instance."""
    profile = copy(extrude_op.profile) # use copy to prevent changing extrude_op internally
    profile.denormalize(extrude_op.sketch_size)

    sketch_plane = copy(extrude_op.sketch_plane)
    #sketch_plane.origin = extrude_op.sketch_pos

    face = create_profile_face(profile, sketch_plane)
    normal = gp_Dir(*extrude_op.sketch_plane.normal)
    ext_vec = gp_Vec(normal).Multiplied(extrude_op.extent_one)
    body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
    if extrude_op.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
        body_sym = BRepPrimAPI_MakePrism(face, ext_vec.Reversed()).Shape()
        body = BRepAlgoAPI_Fuse(body, body_sym).Shape()
    if extrude_op.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
        ext_vec = gp_Vec(normal.Reversed()).Multiplied(extrude_op.extent_two)
        body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        body = BRepAlgoAPI_Fuse(body, body_two).Shape()
    
    return body


def create_profile_face(profile: Profile, sketch_plane: CoordSystem):
    """create a face from a sketch profile and the sketch plane"""
    origin = gp_Pnt(*sketch_plane.origin)
    normal = gp_Dir(*sketch_plane.normal)
    x_axis = gp_Dir(*sketch_plane.x_axis)
    gp_face = gp_Pln(gp_Ax3(origin, normal, x_axis))

    all_loops = [create_loop_3d(loop, sketch_plane) for loop in profile.children]
    topo_face = BRepBuilderAPI_MakeFace(gp_face, all_loops[0])
    for loop in all_loops[1:]:
        topo_face.Add(loop.Reversed())
    return topo_face.Face()


def create_loop_3d(loop: Loop, sketch_plane: CoordSystem):
    """create a 3D sketch loop"""
    topo_wire = BRepBuilderAPI_MakeWire()
    for curve in loop.children:
        topo_edge = create_edge_3d(curve, sketch_plane)
        if topo_edge == -1: # omitted
            continue
        topo_wire.Add(topo_edge)
    return topo_wire.Wire()


def create_edge_3d(curve: CurveBase, sketch_plane: CoordSystem):
    """create a 3D edge"""
    if isinstance(curve, Line):
        if np.allclose(curve.start_point, curve.end_point):
            return -1
        start_point = point_local2global(curve.start_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point)
    elif isinstance(curve, Circle):
        center = point_local2global(curve.center, sketch_plane)
        axis = gp_Dir(*sketch_plane.normal)
        gp_circle = gp_Circ(gp_Ax2(center, axis), abs(float(curve.radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)
    elif isinstance(curve, Arc):
        # print(curve.start_point, curve.mid_point, curve.end_point)
        start_point = point_local2global(curve.start_point, sketch_plane)
        mid_point = point_local2global(curve.mid_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        arc = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc)
    else:
        raise NotImplementedError(type(curve))
    return topo_edge.Edge()


def point_local2global(point, sketch_plane: CoordSystem, to_gp_Pnt=True):
    """convert point in sketch plane local coordinates to global coordinates"""
    g_point = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis + sketch_plane.origin
    if to_gp_Pnt:
        return gp_Pnt(*g_point)
    return g_point


def CADsolid2pc(shape, n_points, name=None):
    """convert opencascade solid to point clouds"""
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    if bbox.IsVoid():
        raise ValueError("box check failed")

    if name is None:
        name = random.randint(100000, 999999)
    write_stl_file(shape, "tmp_out_{}.stl".format(name))
    out_mesh = trimesh.load("tmp_out_{}.stl".format(name))
    os.system("rm tmp_out_{}.stl".format(name))
    out_pc, _ = sample_surface(out_mesh, n_points)
    return out_pc
