import argparse
import sys
import os
import glob
sys.path.append(".")
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")

from PIL import Image
import os


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format("png"))))

def convert_png_to_jpg_and_delete(png_path, jpg_path):
    # 打开PNG文件
    png_image = Image.open(png_path)
    
    # 转换为RGB模式
    rgb_image = png_image.convert('RGB')
    
    # 保存为JPG文件
    rgb_image.save(jpg_path, "JPEG")
    
    # 删除原始的PNG文件
    os.remove(png_path)
    print(f"{png_path} has been deleted.")


from multiprocessing import Process, cpu_count

num_processes = cpu_count()

def main_process(process_id):

    for index in range(process_id, len(out_paths), num_processes):
        print(f"{index + 1}/{len(out_paths)}",end='\r')
        path = out_paths[index]
        name = path.split("/")[-1].split(".")[0]
        save_path = os.path.join(src_dir, name + ".jpg")
        # 调用函数进行转换并删除原始文件
        convert_png_to_jpg_and_delete(path, save_path)

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