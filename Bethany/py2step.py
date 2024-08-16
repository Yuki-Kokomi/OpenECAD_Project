import os
import glob
import numpy as np
import argparse
import sys
sys.path.append(".")

from lib.DataExchange import write_step_file
from lib.visualize import create_CAD
from lib.curves import Line, Arc, Circle
from lib.sketch import Loop, Profile
from lib.math_utils import *
from lib.extrude import CoordSystem, Extrude, CADSequence

# Curves
def add_line(start, end):
    start = np.array(start)
    end = np.array(end)
    return Line(start, end)

def add_arc(start, end, mid):
    # get radius and center point
    start = np.array(start)
    end = np.array(end)
    mid = np.array(mid)
    center, radius = find_circle_center_and_radius(start, end, mid)
    def get_angles_counterclockwise(eps=1e-8):
        c2s_vec = (start - center) / (np.linalg.norm(start - center) + eps)
        c2m_vec = (mid - center) / (np.linalg.norm(mid - center) + eps)
        c2e_vec = (end - center) / (np.linalg.norm(end - center) + eps)
        angle_s, angle_m, angle_e = angle_from_vector_to_x(c2s_vec), angle_from_vector_to_x(c2m_vec), \
                                    angle_from_vector_to_x(c2e_vec)
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s
        return angle_s, angle_e
    angle_s, angle_e = get_angles_counterclockwise()
    return Arc(start, end, center, radius, start_angle=angle_s, end_angle=angle_e, mid_point=mid)

def add_circle(center, radius):
    center = np.array(center)
    return Circle(center, radius)

# Loops
def add_loop(curves):
    res =  Loop(curves)
    res.reorder()
    def autofix(loop):
        if len(loop.children) <= 1:
            return
        if isinstance(loop.children[0], Circle):
            return
        for i in range(0, len(loop.children) - 1):
            if not np.allclose(loop.children[i].end_point, loop.children[i+1].start_point):
                loop.children[i+1].start_point = loop.children[i].end_point
                print("warning: fixing loop")
        if not np.allclose(loop.children[len(loop.children) - 1].end_point, loop.children[0].start_point):
            loop.children[len(loop.children) - 1].start_point = loop.children[0].end_point
            print("warning: fixing loop")
            
    autofix(res)
    return res

# Sketch-Profile
def add_profile(loops):
    return Profile(loops)

def add_sketchplane(origin, normal, x_axis):#, y_axis):
    #print(origin, normal, x_axis)
    origin = np.array(origin)
    normal = np.array(normal)
    x_axis = np.array(x_axis)
    y_axis = find_n_from_x_and_y(normal, x_axis)
    # get theta and phi
    theta, phi, gamma = polar_parameterization(normal, x_axis)
    #print(normal_axis, x_axis)
    #print(theta, phi, gamma)
    return CoordSystem(origin, theta, phi, gamma, y_axis=cartesian2polar(y_axis))

def add_sketchplane_ref(extrude: Extrude, origin, type: str, line: Line = None, reverse = False, angle=0):
    origin = np.array(origin)
    types_dict = ["sameplane", "extent_one", "extent_two", "line"]
    """
    sameplane: 参考Extrude的SketchPlane，angle是以normal为轴，向下看逆时针角度，原点是在SketchPlane内的2D相对坐标。reverse只在最后反转normal，其他参考并不反转。
    line：默认normal为y轴，line start to end 为x轴，它们的叉积为方向向量normal'，原点是在该默认平面的2D相对坐标。angle是以normal'为轴，向下看逆时针角度。reverse只在最后反转normal'，其他参考并不反转。
    """
    if type not in types_dict:
        raise ValueError
    ref_plane = extrude.sketch_plane
    ref_x_axis = unit_vector(ref_plane.x_axis)
    ref_y_axis = unit_vector(ref_plane.y_axis)
    ref_n_axis = unit_vector(ref_plane.normal)
    ref_origin = ref_plane.origin      
    if type == "sameplane":
        real_origin = map_2d_to_3d(ref_origin, ref_x_axis, ref_y_axis, origin)
    elif type == "extent_one":
        ref_origin_ = ref_origin + extrude.extent_one * ref_n_axis
        real_origin = map_2d_to_3d(ref_origin_, ref_x_axis, ref_y_axis, origin)
    elif type == "extent_two":
        ref_origin_ = ref_origin - extrude.extent_two * ref_n_axis
        real_origin = map_2d_to_3d(ref_origin_, ref_x_axis, ref_y_axis, origin)
    if type in types_dict[:3]:
        real_x_axis = rotate_vector(ref_x_axis, ref_n_axis, angle)
        return add_sketchplane(real_origin, ref_n_axis if not reverse else -ref_n_axis, real_x_axis)
    if type == "line":
        if line is None:
            raise TypeError
        start_point = map_2d_to_3d(ref_origin, ref_x_axis, ref_y_axis, line.start_point)
        end_point = map_2d_to_3d(ref_origin, ref_x_axis, ref_y_axis, line.end_point)
        default_x_axis = unit_vector(end_point - start_point)
        real_origin = map_2d_to_3d(start_point, default_x_axis, ref_n_axis, origin) # ref_n_axis is y axis of default plane
        default_normal = find_n_from_x_and_y(default_x_axis, ref_n_axis)
        real_x_axis = rotate_vector(default_x_axis, default_normal, angle)
        return add_sketchplane(real_origin, default_normal if not reverse else -default_normal, real_x_axis)
    raise ValueError

class Sketch(object):
    def __init__(self, sketch_plane, profile, sketch_position, sketch_size):
        self.sketch_plane = sketch_plane
        self.profile = profile
        self.sketch_position = sketch_position
        self.sketch_size = sketch_size

def add_sketch(sketch_plane, profile, sketch_position=[0.0,0.0,0.0], sketch_size=0):
    return Sketch(sketch_plane, profile, np.array(sketch_position), sketch_size)

cad_seq = []
def add_extrude(sketch: Sketch, operation, type, extent_one, extent_two):
    res = Extrude(
        sketch.profile, 
        sketch.sketch_plane,
        np.intc(operation), np.intc(type), np.double(extent_one), np.double(extent_two),
        np.double(sketch.sketch_position),
        np.double(sketch.sketch_size)
    )
    cad_seq.append(res)
    return res

from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TCollection import TCollection_ExtendedString

def _process(path, path_o):
    global cad_seq
    cad_seq.clear()
    with open(path, 'r') as file:
        codes = file.read()
    # 执行读取到的代码
    
    codes = codes.split("\n")
    i = 0
    last_curves_list, last_loops_list = None, None
    while i < len(codes):
        code = codes[i]
        last_curve_name, last_loop_name = None, None
        if codes[i].startswith('Curves'):
            last_curves_list = codes[i].split('=', 1)[0].split()[0]
        elif codes[i].startswith('Loops'):
            last_loops_list = codes[i].split('=', 1)[0].split()[0]
        elif codes[i].startswith('Arc') or codes[i].startswith('Line') or codes[i].startswith('Circle'):
            last_curve_name = codes[i].split('=', 1)[0].split()[0]
        elif codes[i].startswith('Loop'):
            last_loop_name = codes[i].split('=', 1)[0].split()[0]
        for j in range(i + 1, len(codes)):
            if codes[j].startswith('\t') or codes[j].startswith(' '):
                code += '\n' + codes[j]
                i += 1
            else:
                break
        exec(code)
        if last_curve_name is not None:
            exec(f"{last_curves_list}.append({last_curve_name})")
        elif last_loop_name is not None:
            exec(f"{last_loops_list}.append({last_loop_name})")

        i += 1
    doc_name = TCollection_ExtendedString("pythonocc-doc")
    doc = TDocStd_Document(doc_name)
    cad = CADSequence(cad_seq)
    #print(cad)
    out_shape = create_CAD(doc, cad)
        
    write_step_file(doc, path_o)

error_list = []

from lib.file_utils import ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format("py"))))
save_dir = args.src + "_step" if args.outputs is None else args.outputs
ensure_dir(save_dir)

import traceback

for path in out_paths:
    name = path.split("/")[-1].split(".")[0]
    try:
        save_path = os.path.join(save_dir, name + ".step")
        _process(path, save_path)
        
    except Exception as e:
        print("load and create failed.")
        traceback.print_exc()
        print(name)
        error_list.append(name)

for error in error_list:
    print(error)