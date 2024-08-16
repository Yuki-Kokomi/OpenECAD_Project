import os
import glob
import json
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from lib.DataExchange import write_step_file
import argparse
import sys
sys.path.append(".")
from lib.extrude import CADSequence
from lib.visualize import vec2CADsolid, create_CAD, create_CAD_index
from lib.file_utils import ensure_dir


import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=-1, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--mode', type=str, default="default", help="mode of generation")
parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format("json"))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]
save_dir = args.src + "_step" if args.outputs is None else args.outputs
ensure_dir(save_dir)


from multiprocessing import Process, cpu_count

num_processes = cpu_count()

from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TCollection import TCollection_ExtendedString

from lib.timeout import timeout_decorator
def main_process(process_id):
    if args.mode == "segment":
        for index in range(process_id, len(out_paths), num_processes):
            path = out_paths[index]
            @timeout_decorator
            def main__():
                with open(path, 'r') as fp:
                    data = json.load(fp)
                cad_seq = CADSequence.from_dict(data)
                
                for i in range(len(cad_seq.seq)):
                    doc_name = TCollection_ExtendedString("pythonocc-doc")
                    doc = TDocStd_Document(doc_name)
                    out_shape = create_CAD_index(doc, cad_seq, index=i)
                    
                    name = path.split("/")[-1].split(".")[0]
                    sub_folder = os.path.join(save_dir, name)
                    ensure_dir(sub_folder)
                    save_path = os.path.join(sub_folder, name + f".{i:02d}.step")
                    write_step_file(doc, save_path, application_protocol="AP242DIS")
                
                doc_name = TCollection_ExtendedString("pythonocc-doc")
                doc = TDocStd_Document(doc_name)
                out_shape = create_CAD(doc, cad_seq)

                if args.filter:
                    analyzer = BRepCheck_Analyzer(out_shape)
                    if not analyzer.IsValid():
                        print("detect invalid.")
                        return
                name = path.split("/")[-1].split(".")[0]
                save_path = os.path.join(save_dir, name + f".step")
                write_step_file(doc, save_path, application_protocol="AP242DIS")
            try:
                main__()
            except Exception as e:
                print(f"load and create failed. Error: {e}")
                traceback.print_exc()
                continue 
    elif args.mode == "default":
        for index in range(process_id, len(out_paths), num_processes):
            path = out_paths[index]
            @timeout_decorator
            def main__():
                name = path.split("/")[-1].split(".")[0]
                save_path = os.path.join(save_dir, name + f".step")
                if os.path.isfile(save_path): return
                with open(path, 'r') as fp:
                    data = json.load(fp)
                cad_seq = CADSequence.from_dict(data)
                doc_name = TCollection_ExtendedString("pythonocc-doc")
                doc = TDocStd_Document(doc_name)
                cad = CADSequence(cad_seq)
                out_shape = create_CAD(doc, cad_seq)
                if args.filter:
                    analyzer = BRepCheck_Analyzer(out_shape)
                    if not analyzer.IsValid():
                        print("detect invalid.")
                        return
                
                write_step_file(doc, save_path, application_protocol="AP242DIS")
            try:
                main__()
            except Exception as e:
                print(f"load and create failed. Error: {e}")
                traceback.print_exc()
                continue
        

    ## python export2step.py --src ./ --outputs ./

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