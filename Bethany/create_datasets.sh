#!/bin/bash

ENV_NAME=Bethany

CONDA_INIT_SCRIPT="$(conda info --base)/etc/profile.d/conda.sh"

if [ -f "$CONDA_INIT_SCRIPT" ]; then
    # 初始化 conda
    source "$CONDA_INIT_SCRIPT"
else
    echo "Error: Cannot find conda initialization script."
    exit 1
fi

conda activate "$ENV_NAME"


conda activate Bethany

JSON_SOURCE=./examples/

OUT_DIR=./examples/_out

# python json2step.py --src $JSON_SOURCE -o $OUT_DIR/steps_segment/ --mode segment
# python step2img.py --src $OUT_DIR/steps_segment/ -o $OUT_DIR/images_segment/ --mode segment
# python img2seg.py --src $OUT_DIR/images_segment/ -o $OUT_DIR/images_mask --srcc $OUT_DIR/images_transparent/

python json2step.py --src $JSON_SOURCE -o $OUT_DIR/steps_default/ --mode default
python step2img.py --src $OUT_DIR/steps_default/ -o $OUT_DIR/images_default/ --mode default
python step2img.py --src $OUT_DIR/steps_default/ -o $OUT_DIR/images_transparent/ --mode transparent
python step2img.py --src $OUT_DIR/steps_default/ -o $OUT_DIR/images_orthographic/ --mode orthographic

python json2dataset.py --src $JSON_SOURCE -o $OUT_DIR/code_default_directout_3k.json --ignore True --mode default --token 3072
python json2dataset.py --src $JSON_SOURCE -o $OUT_DIR/code_transparent_directout_3k.json --ignore True --mode transparent --token 3072
python json2dataset.py --src $JSON_SOURCE -o $OUT_DIR/code_orthographic_directout_3k.json --ignore True --mode orthographic --token 3072
python json2py.py --src $JSON_SOURCE -o $OUT_DIR/codes_python/
python py2step.py --src $OUT_DIR/codes_python/ -o $OUT_DIR/steps_from_py/
python step2img.py --src $OUT_DIR/steps_from_py/ -o $OUT_DIR/images_from_py/

conda deactivate
