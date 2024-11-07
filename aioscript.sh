#!/bin/bash

# Define the conda initialization script
CONDA_INIT_SCRIPT="$(conda info --base)/etc/profile.d/conda.sh"

# Initialize conda if the script is found
if [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
else
    echo "Error: Cannot find conda initialization script."
    exit 1
fi

# Step 1: Clone the Project
clone_project() {
    echo "Cloning the project..."
    git submodule update --init --recursive
}

# Step 2: Install TinyLlava Environment
install_tinyllava() {
    echo "Installing TinyLlava environment..."
    cd TinyLLaVA_Factory
    git apply ../OpenECAD_TinyLLaVA.patch

    conda create -n tinyllava_factory python=3.10 -y
    conda activate tinyllava_factory
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
    pip install "numpy<2.0.0"
    ##pip install flash-attn --no-build-isolation  # for training
    conda deactivate
    cd ..

    cd Bethany
    conda create -n Bethany python=3.10 -y
    conda activate Bethany
    pip install -r requirements.txt
    conda install -c conda-forge pythonocc-core=7.5.1
    conda deactivate
    cd ..
}

# Step 3: Download Trained Weight
download_weights() {
    echo "Downloading trained weights..."
    # sudo apt install git-lfs
    git lfs install
    git clone https://huggingface.co/Yuki-Kokomi/OpenECADv2-SigLIP-2.4B
}

# Step 4: Download Test Images
download_test_images() {
    echo "Downloading test images..."
    # sudo apt install unzip
    wget https://huggingface.co/datasets/Yuki-Kokomi/OpenECADv2-Datasets/resolve/main/OpenECADv2_EvaluationExamples.zip
    unzip OpenECADv2_EvaluationExamples.zip
    rm -r "OpenECADv2_EvaluationExamples/Sample Output Codes"
}

# Step 5: Run and Get Results
run_and_get_results() {
    echo "Running and getting results (this may take a long time)..."
    conda activate tinyllava_factory
    # export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    python run_model.py --model_path ./OpenECADv2-SigLIP-2.4B \
        --src "./OpenECADv2_EvaluationExamples/Input Pictures and Reference Codes" \
        --out "./OpenECADv2_EvaluationExamples/Output/Codes"
    conda deactivate
}

# Step 6: Transfer Py to Step
transfer_py_to_step() {
    echo "Transferring py to step..."
    cd Bethany
    conda activate Bethany
    python py2step.py --src ../OpenECADv2_EvaluationExamples/Output/Codes/Default \
        -o ../OpenECADv2_EvaluationExamples/Output/Steps/Default
    python step2img.py --src ../OpenECADv2_EvaluationExamples/Output/Steps/Default \
        -o ../OpenECADv2_EvaluationExamples/Output/Images/Default --mode default --num -1
    conda deactivate
    cd ..
}

# Function to prompt user for step choice
choose_step() {
    echo "You need to install conda unzip cuda-11-8 git-lfs first."
    echo "You should use huggingface-cli login first."
    echo "Choose the step to run:"
    echo "1. Clone Project"
    echo "2. Install Environment"
    echo "3. Download Trained Weight"
    echo "4. Download Test Images"
    echo "5. Run and Get Results"
    echo "6. Transfer Py to Step"
    echo "7. Exit"

    read -p "Enter step number: " step_choice

    case $step_choice in
        1)
            clone_project
            ;;
        2)
            install_tinyllava
            ;;
        3)
            download_weights
            ;;
        4)
            download_test_images
            ;;
        5)
            run_and_get_results
            ;;
        6)
            transfer_py_to_step
            ;;
        7)
            echo "Exiting script."
            exit 0
            ;;
        *)
            echo "Invalid option. Please choose a valid step number."
            choose_step
            ;;
    esac
}

# Start the script by asking the user which step to begin with
choose_step
