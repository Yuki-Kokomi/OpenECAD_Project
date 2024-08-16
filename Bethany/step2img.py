import argparse
import sys
import os
import glob
import json
sys.path.append(".")
from lib.file_utils import ensure_dir
import random

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

from OCC.Extend.DataExchange import read_step_file_with_names_colors
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

from OCC.Core.AIS import AIS_Shape
from OCC.Core.Aspect import Aspect_TOL_SOLID, Aspect_TOL_DASH, Aspect_TOL_DOT
from OCC.Core.Prs3d import Prs3d_Drawer, Prs3d_LineAspect, Prs3d_Root

from PIL import Image

import traceback

os.environ["PYTHONOCC_OFFSCREEN_RENDERER"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=-1, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
parser.add_argument('--mode', type=str, default="default", help="mode of generation")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format("step"))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]
save_dir = args.src + "_images" if args.outputs is None else args.outputs
ensure_dir(save_dir)

from lib.timeout import timeout_decorator

from multiprocessing import Process, cpu_count

num_processes = 4 #cpu_count()

def main_process(process_id):

    HasTarget = False
    for index in range(process_id, len(out_paths), num_processes):
        HasTarget = True
        break
    if not HasTarget:
        return 

    # 初始化显示窗口
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.View.SetBgGradientColors(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB), Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB), 2, True)

    # 加载3D模型
    def load_step_file(filename):
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(filename)
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()
            shapes_labels_colors = read_step_file_with_names_colors(filename)
            return shapes_labels_colors, shape
        else:
            raise IOError("Error: cannot read file.")

    # 计算模型包围盒
    def compute_bounding_box(shape):
        box = Bnd_Box()
        brepbndlib_Add(shape, box)
        return box

    # 渲染并保存图片
    def render_and_save(display, shapes_labels_colors, view_name):
        #display.DisplayShape(shape, update=True)
        for shpt_lbl_color in shapes_labels_colors:
            label, c = shapes_labels_colors[shpt_lbl_color]

            if args.mode == "segment":
                color = Quantity_Color(c.Red() , c.Green() , c.Blue() , Quantity_TOC_RGB)
                ais_shape = AIS_Shape(shpt_lbl_color)
                ais_shape.SetColor(color)
                if color == Quantity_Color(1.0 , 0.1 , 0.1 , Quantity_TOC_RGB):
                    display.Context.SetTransparency(ais_shape, 0.0, False)
                else:
                    display.Context.SetTransparency(ais_shape, 1.0, False)
                display.Context.Display(ais_shape, False)
            elif args.mode == "transparent":
                ais_shape = AIS_Shape(shpt_lbl_color)
                color = Quantity_Color(c.Red() * 0.25, c.Green() * 0.25, c.Blue() * 0.25, Quantity_TOC_RGB)
                ais_shape.SetColor(color)
                
                display.Context.SetTransparency(ais_shape, 0.25, False)
                display.Context.Display(ais_shape, False)

                ais_shape = AIS_Shape(shpt_lbl_color)
                ais_shape.SetWidth(2.0)
                display.Context.SetTransparency(ais_shape, 1.0, False)
                display.Context.Display(ais_shape, False)
            else:
                ais_shape = AIS_Shape(shpt_lbl_color)
                color = Quantity_Color(c.Red() * 0.25, c.Green() * 0.25, c.Blue() * 0.25, Quantity_TOC_RGB)
                ais_shape.SetColor(color)
                
                display.Context.SetTransparency(ais_shape, 0.0, False)
                display.Context.Display(ais_shape, False)

                ais_shape = AIS_Shape(shpt_lbl_color)
                ais_shape.SetWidth(2.0)
                display.Context.SetTransparency(ais_shape, 1.0, False)
                display.Context.Display(ais_shape, False)

            #display.DisplayColoredShape(
            #    shpt_lbl_color,
            #    color=Quantity_Color(c.Red() * 0.25, c.Green() * 0.25, c.Blue() * 0.25, Quantity_TOC_RGB),
            #)
        display.FitAll()
        display.View.Dump(view_name)

    # 设置视角并保存图片
    def save_views_from_angles(shapes_labels_colors, proj_directions, output_dir, name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_paths = []
        for proj_direction in proj_directions:
            x, y, z = proj_direction
            display.View_Front()
            #display.View_Iso()
            display.View.SetProj(x, y, z)
            display.View.FitAll()
            if args.mode == "orthographic":
                if x * y * z > 0: d = 'o'
                elif x > 0: d = 'x'
                elif y > 0: d = 'y'
                else: d = 'z'
                view_name = os.path.join(output_dir, f'{name}_{d}.jpg')
            else:
                view_name = os.path.join(output_dir, f'{name}.jpg')
            render_and_save(display, shapes_labels_colors, view_name)
            image_paths.append(view_name)
        
        return image_paths

    def merge_images(image_paths, output_path):
        # 打开所有图片
        images = [Image.open(image_path) for image_path in image_paths]

        # 获取单张图片的宽度和高度（假设所有图片大小相同）
        width, height = images[0].size

        # 创建一个新图像，大小为 2x2 布局的总大小
        merged_image = Image.new('RGB', (2 * width, 2 * height))

        # 将每张图片粘贴到新图像的正确位置
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        for pos, img in zip(positions, images):
            merged_image.paste(img, pos)

        # 保存拼接后的图像
        merged_image.save(output_path)

        # 删除原图
        for image_path in image_paths:
            os.remove(image_path)

    for index in range(process_id, len(out_paths), num_processes):
        step_file = out_paths[index]

        @timeout_decorator
        def main__():
            name = step_file.split("/")[-1].split(".")[0]
            view_name = os.path.join(save_dir, f'{name}.jpg')
            if os.path.isfile(view_name): return
            # 加载模型
            shapes_labels_colors, shape = load_step_file(step_file)
            # 计算包围盒
            bounding_box = compute_bounding_box(shape)
            # 定义方向矢量
            if args.mode == "default":
                proj_directions = [
                    (random.random() * 0.25 + 0.75, random.random() * 0.25 + 0.75, random.random() * 0.25 + 0.75)
                ]
            elif args.mode == "orthographic":
                proj_directions = [
                    (1, 1, 1),
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1)
                ]
            else:
                proj_directions = [
                    (1, 1, 1)
                ]
            
            # 保存多角度视图
            image_paths = save_views_from_angles(shapes_labels_colors, proj_directions, save_dir, name=name)
            if args.mode == "orthographic":
                output_path = os.path.join(save_dir, f'{name}.jpg')
                merge_images(image_paths, output_path)
            # 结束显示
            display.EraseAll()
            #display.FitAll()
            #start_display()

            sub_folder = os.path.join(src_dir, name)
            sub_output_dir = os.path.join(save_dir, name)
            if os.path.exists(sub_folder) and args.mode == "segment":
                ensure_dir(sub_output_dir)
                sub_steps = sorted(glob.glob(os.path.join(sub_folder, "*.{}".format("step"))))
                for sub_step in sub_steps:
                    sub_shapes_labels_colors, sub_shape = load_step_file(sub_step)
                    bounding_box = compute_bounding_box(sub_shape)
                    sub_proj_directions = [(1, 1, 1)]
                    sub_name = sub_step.split("/")[-1].split(".")[0] + "." + sub_step.split("/")[-1].split(".")[1]
                    image_paths = save_views_from_angles(sub_shapes_labels_colors, sub_proj_directions, sub_output_dir, name=sub_name)
                    display.EraseAll()
        try:
            main__()        
            display.EraseAll()
        except Exception as e:
            print("load and create failed.")
            traceback.print_exc()
            display.EraseAll()
            continue


if __name__ == "__main__":
    processes = []
    for i in range(num_processes):
        process = Process(target=main_process, args=(i,))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    print('任务完成')    