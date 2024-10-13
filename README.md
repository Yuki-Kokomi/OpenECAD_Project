# OpenECAD: An Efficient Visual Language Model for Editable 3D-CAD Design
The project contains the source code used in the paper "OpenECAD: An Efficient Visual Language Model for Editable 3D-CAD Design."

This project primarily utilizes modified DeepCAD libraries to convert the DeepCAD dataset into the OpenECAD dataset. The dataset is in LLaVA format, and the TinyLLaVA Factory is used to fine-tune the pre-trained Vision Language Model, resulting in the OpenECAD model weights. Finally, the OpenECAD model weights are run using the TinyLLaVA Factory. CUDA is need in this project.

If you want to start from the training set we generated, please begin from Step 2.

If you want to start from the model weights we trained, please begin from Step 3.

The dataset and model weights can be accessed at https://huggingface.co/collections/Yuki-Kokomi/openecadv2-66b2d793cb885dff986cf967.

To generate the OpenECAD dataset and train using other CAD datasets, please start from scratch (this project uses DeepCAD as an example).

## Step 1. Use the Bethany tool to convert the DeepCAD dataset into the OpenECAD dataset.
First, download all the JSON files from the DeepCAD dataset and place them in the same directory. (Download link: http://www.cs.columbia.edu/cg/deepcad/data.tar)

Then, create an environment in Linux using Conda:

```shell
conda create -n Bethany python=3.10 -y
conda activate Bethany
pip install -r requirements.txt
conda install -c conda-forge pythonocc-core=7.5.1
```

Next, modify the `JSON_SOURCE` and `OUT_DIR` in create_datasets.sh to the actual paths.

Alternatively, you can manually execute the commands to generate the desired dataset:

```shell
python json2step.py --src $JSON_SOURCE -o $OUT_DIR/steps_default/ --mode default
# Converting DeepCAD JSON Files to STEP Files

python step2img.py --src $OUT_DIR/steps_default/ -o $OUT_DIR/images_default/ --mode default
python step2img.py --src $OUT_DIR/steps_default/ -o $OUT_DIR/images_transparent/ --mode transparent
python step2img.py --src $OUT_DIR/steps_default/ -o $OUT_DIR/images_orthographic/ --mode orthographic

python json2dataset.py --src $JSON_SOURCE -o $OUT_DIR/code_default_directout_3k.json --ignore True --mode default --token 3072
python json2dataset.py --src $JSON_SOURCE -o $OUT_DIR/code_transparent_directout_3k.json --ignore True --mode transparent --token 3072
python json2dataset.py --src $JSON_SOURCE -o $OUT_DIR/code_orthographic_directout_3k.json --ignore True --mode orthographic --token 3072

# Generate datasets for the three types of views (see the paper for details) and limit the number of tokens.
```

If you want to create a mixed dataset with the three types of views, you can write a script to accomplish this after generating the datasets mentioned above.

If you encounter issues running in a non-GUI environment, you can refer to: https://github.com/tpaviot/pythonocc-core/issues/708

## Step 2. Use the TinyLLaVA training framework to fine-tune the pre-trained Small Vision Language Model to obtain the OpenECAD model.

First, pull a specific version of the TinyLLaVA repository (the latest version may work, but its compatibility is not guaranteed):

```shell
git submodule update --init --recursive
```

Next, apply our patch file to the TinyLLaVA project:

```shell
cd TinyLLaVA_Factory
git apply ../OpenECAD_TinyLLaVA.patch
```

Then, set up the environment according to the TinyLLaVA README file:

```shell
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation # for training

# If you encounter errors in this part, please try referring to https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/15863
# Downgrade setuptools to below version 70.0.0
```

Then, refer to the link in the TinyLLaVA README to download the TinyLLaVA pre-trained weights.

Finally, modify the dataset JSON file, image folder, pre-trained weights, and output paths in `OpenECADv2_{X}B_finetune.sh`. You can then start training, keeping in mind the correspondence between the script and the pre-trained weights.

After training is complete, you can use `merge_lora_weights.py` to merge the Lora weights with the pre-trained weights. If you encounter issues merging Gemma-based models, refer to https://github.com/TinyLLaVA/TinyLLaVA_Factory/issues/88.

## Step 3. Use the OpenECAD model weights to generate OpenECAD code.

If the environment is not installed, please first follow the steps mentioned in Step 2 to install the TinyLLaVA environment and apply our patch.

You can use the model with the WebUI provided by TinyLLaVA:

```shell
python tinyllava/serve/app.py --model-path <Your_Model_Path>
```

If you want to batch process, you can refer to `eval.py`.


## Step 4. Convert the OpenECAD code to STEP files.

If you have already installed the Bethany environment, you can use Bethany directly. Otherwise, you can follow the steps below to install it:

```shell
git submodule update --init --recursive
cd OpenECADv2toSTEP
conda create -n OpenECADv2toSTEP python=3.10 -y
conda activate OpenECADv2toSTEP
pip install -r requirements.txt
conda install -c conda-forge pythonocc-core=7.5.1
```

Afterwards, you can convert the OpenECAD Python files to STEP files. Before using them, make sure to remove any non-code parts generated by the model.

```shell
python py2step.py --src <Python_Files_Path> -o <Output_Path>
```

We provide test cases at https://huggingface.co/datasets/Yuki-Kokomi/OpenECADv2-Datasets/blob/main/OpenECADv2_EvaluationExamples.zip, which include input images, corresponding reference code, the code results generated by us, and a Python script used to compare the reference code with the generated code and assign scores. To get the score:

```shell
python check_scores.py --gen <Gen_Python_Codes_Folder> --ans <Reference_Code_Folder> --output <Output_CSV_Path>
```
