from tinyllava.eval.run_tiny_llava import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help="Path to the model.")
parser.add_argument('--src', type=str, required=True, help="Path of Input Pictures and Reference Codes.")
parser.add_argument('--out', type=str, required=True, help="Any output path you like.")

# 解析命令行参数
args = parser.parse_args()
    
# 将命令行输入的值传递给相应的变量
model_path = args.model_path
src_base = args.src
out_base = args.out

input_types = ["Default"]#["Default", "Transparent", "Orthographic"]
conv_mode = "gemma" # or llama, gemma, phi

## You need to change the "max_new_tokens" if the model can't deal with long tokens.
## possible values: 1024, 1152, 1536, 2048, 3072
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "conv_mode": conv_mode,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 2048
})()

# Model
disable_torch_init()

if args.model_path is not None:
    model, tokenizer, image_processor, context_len = load_pretrained_model(args.model_path)
else:
    assert args.model is not None, 'model_path or model must be provided'
    model = args.model
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    tokenizer = model.tokenizer
    image_processor = model.vision_tower._image_processor


text_processor = TextPreprocess(tokenizer, args.conv_mode)
data_args = model.config
image_processor = ImagePreprocess(image_processor, data_args)
model.cuda()

import os

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

# Set timeout (unit: s)
timeout = 300

def timeout_decorator(func):
    def wrapper(*args, **kwargs):
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
        except TimeoutException:
            print("Function timed out!")
            raise TimeoutException
            result = None
        finally:
            signal.alarm(0)
        return result
    return wrapper

@timeout_decorator
def process_image(qs, path):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs


    msg = Message()
    msg.add_message(qs)

    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    prompt = result['prompt']
    input_ids = input_ids.unsqueeze(0).cuda()
            

    image_files = [path]
    images = load_images(image_files)[0]
    images_tensor = image_processor(images)
    images_tensor = images_tensor.unsqueeze(0).half().cuda()

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs

import re

def extract_python_code(input_str):
    # 匹配以```python开头，```结束的内容
    match = re.search(r'```python(.*?)```', input_str, re.DOTALL)
    
    if match:
        # 如果找到匹配的内容，返回```python和```之间的内容
        return match.group(1)
    else:
        # 如果没有```python```包裹的内容，返回```python后面的所有内容
        # 找到```python的位置并返回从该位置到字符串末尾的所有内容
        match = re.search(r'```python(.*)', input_str, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return input_str

import os
import glob
import traceback
errors = []
for index in range(len(input_types)):
    cur_type = input_types[index]
    src = src_base + cur_type
    out = out_base + cur_type
    ensure_dir(out)
    out_paths = sorted(glob.glob(os.path.join(src, "*.{}".format("jpg"))))
    if cur_type == "Orthographic" :
        qs = "This image is 4 views of a 3D model from certain angles. Please try to use Python-style APIs to render this model."
    else:    
        qs = "This image is a view of a 3D model from a certain angle. Please try to use Python-style APIs to render this model."

    for i in range(len(out_paths)):
        path = out_paths[i]
        print(f"{cur_type}: {i + 1}/{len(out_paths)}", end='\r')
        name = path.split("/")[-1].split(".")[0]
        save_path = os.path.join(out, f'{name}.py')
        if os.path.isfile(save_path): continue
        try:
            outputs = process_image(qs, path)
            outputs = extract_python_code(outputs)
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(outputs)
            file.close()
        except:
            errors.append(f"{cur_type}: {name}")
            print(f"gen error: {name}")
            traceback.print_exc()
    print()

print("Can't Generate these inputs:")
print(errors)
