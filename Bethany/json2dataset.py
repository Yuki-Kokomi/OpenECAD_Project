import os
import glob
import json
import argparse
import sys
sys.path.append(".")
from lib.extrude import CADSequence

from lib.cad2code import get_cad_code
from count_tokens.count import count_tokens_in_string

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=-1, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--filter', type=str, default=None, help="filter folder")
parser.add_argument('--ignore', type=bool, default=False, help="ignore too long code")
parser.add_argument('--mode', type=str, default="default", help="mode of generation")
parser.add_argument('--token', type=int, default=2048, help="limit of tokens count")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save filename")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format("json"))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]


from multiprocessing import Process, cpu_count, Manager

#tmp_folder="./_tmp/"
#ensure_dir(tmp_folder)

num_processes = cpu_count()

def main_process(process_id, result_list):
    json_data = []
    for index in range(process_id, len(out_paths), num_processes):
        print(f"{index + 1}/{len(out_paths)}",end='\r')
        path = out_paths[index]
        name = path.split("/")[-1].split(".")[0]

        data = {}
        data["id"] = f"{name}"
        data["image"] = f"{name}.jpg"
        data["conversations"] = []

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
            
            conversation_human = {"from": "human"}
            if args.mode == "transparent":
                conversation_human["value"] = f"<image>\nThis image is a transparent view of a 3D model from a certain angle. Please try to use OpenECAD-style API to render this model."
            elif args.mode == "orthographic":
                conversation_human["value"] = f"<image>\nThis image contains 4 views of a 3D model from a certain angle and three orthographic views. Please try to use OpenECAD-style API to render this model."
            else:
                conversation_human["value"] = f"<image>\nThis image is a view of a 3D model from a certain angle. Please try to use OpenECAD-style API to render this model."
            data["conversations"].append(conversation_human)
            conversation_gpt = {"from": "gpt"}
            conversation_gpt["value"] = f"Of course, here are the codes:\n```python\n{cad_code}```"
            num_tokens = count_tokens_in_string(cad_code)
            if num_tokens > args.token:
                continue
            data["conversations"].append(conversation_gpt)
            json_data.append(data)
        except Exception as e:
            print(f"load and create failed. Error: {e}")
            continue
    
    result_list.append(json_data)
    #json_str = json.dumps(json_data, indent=4)  # `indent=4` 用于美化输出, 使其更易读
    #with open(os.path.join(tmp_folder ,f"data{process_id}.json"), "w") as json_file:
    #    json_file.write(json_str)

    ## python export2step.py --src ./ --filter ./

if __name__ == "__main__":
    with Manager() as manager:
        # 创建一个共享的列表
        result_list = manager.list()

        processes = []
        for i in range(num_processes):
            process = Process(target=main_process, args=(i, result_list,))
            processes.append(process)
            process.start()

    # 等待所有进程完成
        for process in processes:
            process.join()

        result_list = list(result_list)
        json_res = []
        for result in result_list:
            json_res += result

        print()
        print(len(json_res))

        with open(f'{args.outputs}', 'w') as f:
            json.dump(json_res, f, indent=4)

    print('任务完成')    