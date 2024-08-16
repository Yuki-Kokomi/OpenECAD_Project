import numpy as np
from lib.extrude import CADSequence
import lib.curves
from lib.math_utils import *

tol = 1e-10
def get_sketchplane(cad_seq, index):
    ext = cad_seq.seq[index]
    res = f"SketchPlane{index} = add_sketchplane(\n"
    res += "\torigin= {}, normal= {}, x_axis= {})\n".format(
            np.array2string(ext.sketch_plane.origin.round(4), separator=', '), np.array2string(ext.sketch_plane.normal.round(4), separator=', '), 
            np.array2string(ext.sketch_plane.x_axis.round(4), separator=', '))
    return res

def get_sketchplane_ref(cad_seq, index):
    if index == 0: return get_sketchplane(cad_seq, index) # 第一个没有参考
    target_ext = cad_seq.seq[index]
    target_plane = target_ext.sketch_plane
    target_x_axis = target_plane.x_axis
    target_n_axis = target_plane.normal
    target_origin = target_plane.origin
    for i in range(0, index):
        ref_ext = cad_seq.seq[i]
        ref_plane = ref_ext.sketch_plane
        ref_x_axis = ref_plane.x_axis
        ref_y_axis = ref_plane.y_axis
        ref_n_axis = ref_plane.normal
        ref_origin = ref_plane.origin
        if are_parallel(target_n_axis, ref_n_axis):
            # sameplane or extent_one/two
            reverse = False if np.dot(target_n_axis, ref_n_axis) > 0 else True
            vec_origin = target_origin - ref_origin
            #if (np.linalg.norm(vec_origin) < tol) or (np.dot(vec_origin, ref_n_axis) < tol and np.abs(np.dot(vec_origin, ref_x_axis)) > tol):
            if is_point_on_plane(ref_origin, ref_x_axis, ref_y_axis, target_origin):
                # sameplane if OO' is same point or OO' and normal_axis are vertical.
                typeop = "sameplane"
                origin = map_3d_to_2d(ref_origin, ref_x_axis, ref_y_axis, target_origin)
                angle = calculate_rotation_angle(ref_x_axis, target_x_axis, ref_n_axis)
                res = f"SketchPlane{index} = add_sketchplane_ref(\n"
                res += f"\tExtrude{i}, origin = {np.array2string(origin.round(4), separator=', ')}, type = \"{typeop}\""
                if np.abs(angle) > tol:
                    res += f", angle = {number_to_pi_string(angle)}"
                if reverse:
                    res += ", reverse = True"
                res += ")\n"
                return res                
            #elif (are_parallel(vec_origin, ref_n_axis) and np.dot(vec_origin, ref_n_axis) > 0) or (np.dot(vec_origin - ref_ext.extent_one * ref_n_axis, ref_n_axis) < tol):
            elif (are_parallel(vec_origin, ref_n_axis) and np.dot(vec_origin, ref_n_axis) > 0) and np.abs(distance_of_point_and_plane(ref_origin, ref_x_axis, ref_y_axis, target_origin) - np.abs(ref_ext.extent_one)) < tol:
                typeop = "extent_one"
                ref_origin_ = ref_origin + ref_n_axis * ref_ext.extent_one
                origin = map_3d_to_2d(ref_origin_, ref_x_axis, ref_y_axis, target_origin)
                angle = calculate_rotation_angle(ref_x_axis, target_x_axis, ref_n_axis)
                res = f"SketchPlane{index} = add_sketchplane_ref(\n"
                res += f"\tExtrude{i}, origin = {np.array2string(origin.round(4), separator=', ')}, type = \"{typeop}\""
                if np.abs(angle) > tol:
                    res += f", angle = {number_to_pi_string(angle)}"
                if reverse:
                    res += ", reverse = True"
                res += ")\n"
                return res
            #elif  (are_parallel(vec_origin, ref_n_axis) and np.dot(vec_origin, ref_n_axis) < 0) or (np.dot(vec_origin + ref_ext.extent_two * ref_n_axis, ref_n_axis) < tol):
            elif (are_parallel(vec_origin, ref_n_axis) and np.dot(vec_origin, ref_n_axis) < 0) and np.abs(distance_of_point_and_plane(ref_origin, ref_x_axis, ref_y_axis, target_origin) - np.abs(ref_ext.extent_two)) < tol:
                typeop = "extent_two"
                ref_origin_ = ref_origin - ref_n_axis * ref_ext.extent_two
                origin = map_3d_to_2d(ref_origin_, ref_x_axis, ref_y_axis, target_origin)
                angle = calculate_rotation_angle(ref_x_axis, target_x_axis, ref_n_axis)
                res = f"SketchPlane{index} = add_sketchplane_ref(\n"
                res += f"\tExtrude{i}, origin = {np.array2string(origin.round(4), separator=', ')}, type = \"{typeop}\""
                if np.abs(angle) > tol:
                    res += f", angle = {number_to_pi_string(angle)}"
                if reverse:
                    res += ", reverse = True"
                res += ")\n"
                return res
            else:
                pass # next situation 
        elif np.dot(target_n_axis, ref_n_axis) < tol:
            typeop = "line"
            # line
            for j, loop in enumerate(ref_ext.profile.children):
                for k, curve in enumerate(loop.children):
                    if type(curve) == lib.curves.Line:
                        start_point = map_2d_to_3d(ref_origin, ref_x_axis, ref_y_axis, curve.start_point)
                        end_point = map_2d_to_3d(ref_origin, ref_x_axis, ref_y_axis, curve.end_point)
                        ref_x_axis_ = unit_vector(end_point - start_point)
                        ref_y_axis_ = ref_n_axis
                        ref_n_axis_ = find_n_from_x_and_y(ref_x_axis_, ref_y_axis_)
                        ref_origin_ = start_point
                        reverse = False if np.dot(target_n_axis, ref_n_axis_) > 0 else True
                        vec_origin = target_origin - ref_origin_
                        if are_parallel(target_n_axis, ref_n_axis_):
                            #if (np.linalg.norm(vec_origin) < tol) or (np.dot(vec_origin, ref_n_axis_) < tol  and np.abs(np.dot(vec_origin, ref_x_axis_)) > tol):

                            if is_point_on_plane(ref_origin_, ref_x_axis_, ref_y_axis_, target_origin):
                            # if SO' is same point or SO' and normal_axis are vertical.
                                origin = map_3d_to_2d(ref_origin_, ref_x_axis_, ref_y_axis_, target_origin)
                                angle = calculate_rotation_angle(ref_x_axis_, target_x_axis, ref_n_axis_)
                                res = f"SketchPlane{index} = add_sketchplane_ref(\n"
                                res += f"\tExtrude{i}, origin= {np.array2string(origin.round(4), separator=', ')}, type= \"{typeop}\", line= Line{i}_{j}_{k}"
                                if np.abs(angle) > tol:
                                    res += f", angle= {number_to_pi_string(angle)}"
                                if reverse:
                                    res += ", reverse= True"
                                res += ")\n"
                                return res

    return  get_sketchplane(cad_seq, index) # 查找失败，使用绝对坐标定义

        
    



def get_cad_code(cad_seq):
    cad_code = ""
    for i, ext in enumerate(cad_seq.seq):                   
        # SketchPlane
        cad_code += get_sketchplane_ref(cad_seq, i)
        # Loops
        cad_code += f"Loops{i} = []\n"
        for j, loop in enumerate(ext.profile.children):
            # Curves
            cad_code += f"Curves{i}_{j} = []\n"
            for k, curve in enumerate(loop.children):
                if type(curve) == lib.curves.Line:
                    cad_code += f"Line{i}_{j}_{k} = add_line(start= {np.array2string(curve.start_point.round(4), separator=', ')}, end= {np.array2string(curve.end_point.round(4), separator=', ')})\n"
                    #cad_code += f"Curves{i}_{j}.append(Line{i}_{j}_{k})\n"
                elif type(curve) == lib.curves.Arc:
                    cad_code += f"Arc{i}_{j}_{k} = add_arc(start= {np.array2string(curve.start_point.round(4), separator=', ')}, "
                    cad_code += f"end= {np.array2string(curve.end_point.round(4), separator=', ')}, mid= {np.array2string(curve.mid_point.round(4), separator=', ')})\n"
                    #cad_code += f"Curves{i}_{j}.append(Arc{i}_{j}_{k})\n"
                elif type(curve) == lib.curves.Circle:
                    cad_code += f"Circle{i}_{j}_{k} = add_circle(center= {np.array2string(curve.center.round(4), separator=', ')}, radius= {np.array2string(np.float64(curve.radius).round(4), separator=', ')})\n"
                    #cad_code += f"Curves{i}_{j}.append(Circle{i}_{j}_{k})\n"
            cad_code += f"Loop{i}_{j} = add_loop(Curves{i}_{j})\n"
            #cad_code += f"Loops{i}.append(Loop{i}_{j})\n"
        # Profile
        cad_code += f"Profile{i} = add_profile(Loops{i})\n"
        # Sketch
        cad_code += f"Sketch{i} = add_sketch(sketch_plane= SketchPlane{i}, profile= Profile{i})\n"
        #cad_code += "\tsketch_position= {}, sketch_size= {})\n".format(
        #    np.array2string(ext.sketch_pos.round(4), separator=', '), np.array2string(ext.sketch_size.round(4), separator=', '))
        # Finally: Extrude
        cad_code += f"Extrude{i} = add_extrude(sketch= Sketch{i},\n"
        cad_code += "\toperation= {}, type= {}, extent_one= {}, extent_two= {})\n".format(
            ext.operation, ext.extent_type, np.array2string(np.float64(ext.extent_one).round(4), separator=', '), np.array2string(np.float64(ext.extent_two).round(4), separator=', '))
    return cad_code