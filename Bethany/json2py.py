import os
import glob
import json
import argparse
import sys
sys.path.append(".")
from lib.extrude import CADSequence
from lib.file_utils import ensure_dir

from lib.cad2code import get_cad_code

import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=-1, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--filter', type=str, default=None, help="filter folder")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format("json"))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]
save_dir = args.src + "_py" if args.outputs is None else args.outputs
ensure_dir(save_dir)

from multiprocessing import Process, cpu_count, Manager

num_processes = cpu_count()

def main_process(process_id):
    for index in range(process_id, len(out_paths), num_processes):
        print(f"{index + 1}/{len(out_paths)}",end='\r')
        path = out_paths[index]
        name = path.split("/")[-1].split(".")[0]

        if args.filter is not None:
            filter_dir = args.filter
            filter_path = os.path.join(filter_dir, name + ".jpg")
            if not os.path.isfile(filter_path):
                continue

        try:
            with open(path, 'r') as fp:
                src_data = json.load(fp)
            cad_seq = CADSequence.from_dict(src_data)
            cad_code = get_cad_code(cad_seq)
            #print(cad_code)
        except Exception as e:
            print(f"load and create failed. Error: {e}")
            traceback.print_exc()
            continue
        
        save_path = os.path.join(save_dir, name + ".py")
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(cad_code)
    ## python export2step.py --src ./ --filter ./

if __name__ == "__main__":
    with Manager() as manager:
        # 创建一个共享的列表

        processes = []
        for i in range(num_processes):
            process = Process(target=main_process, args=(i, ))
            processes.append(process)
            process.start()

    # 等待所有进程完成
        for process in processes:
            process.join()

    print('任务完成')    
