#!/bin/bash
#SBATCH --job-name=wikigap_pipeline
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00                               
#SBATCH --error=logs/slurm-%j.log
#SBATCH --output=logs/slurm-%j.log

# --------------------------------------------------------------------------- #
# SLURM script to run WikiGap preprocessing and InfoGap pipeline
# --------------------------------------------------------------------------- #

echo "=========================================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================================================="

# --------------------------------------------------------------------------- #
# Setup environment
# --------------------------------------------------------------------------- #

# check if logs directory exists, else create it
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Print current environment for debugging
echo "PATH: $PATH"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"

# Activate conda environment
MYENV=mentalenv
eval "$(conda shell.bash hook)"
conda activate $MYENV || echo "Failed to activate environment with method 1"

# Verify Python and environment
which python || echo "Python not found after activation"
python --version || echo "Python command not working"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TARGET_LANG="fr"
N_SAMPLE=500
MODEL="gpt-5-mini"
INFOGAP_DIR="/home/Lupo/infogap"
INPUT_DATA_DIR="/data/wikimt/input"

echo ""
echo "=========================================================================="
echo "Configuration:"
echo "  Target Language: $TARGET_LANG"
echo "  Sample Size: $N_SAMPLE"
echo "  Model: $MODEL"
echo "=========================================================================="

# --------------------------------------------------------------------------- #
# Step 1: Run preprocessing script
# --------------------------------------------------------------------------- #

echo ""
echo "=========================================================================="
echo "STEP 1: Running preprocessing script"
echo "=========================================================================="
echo ""

python preprocess_wikigap_sample.py \
    --tgt-lang $TARGET_LANG \
    --n-sample $N_SAMPLE \
    --infogap-dir $INFOGAP_DIR \
    --input-data-dir $INPUT_DATA_DIR


PREPROCESS_EXIT_CODE=$?

if [ $PREPROCESS_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Preprocessing failed with exit code $PREPROCESS_EXIT_CODE"
    exit $PREPROCESS_EXIT_CODE
fi

echo ""
echo "✓ Preprocessing completed successfully"

# --------------------------------------------------------------------------- #
# Step 2: Run InfoGap pipeline
# --------------------------------------------------------------------------- #

echo ""
echo "=========================================================================="
echo "STEP 2: Running InfoGap pipeline"
echo "=========================================================================="
echo ""

python main_complete_analysis.py run-multiple-topics --model=$MODEL

PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: InfoGap pipeline failed with exit code $PIPELINE_EXIT_CODE"
    exit $PIPELINE_EXIT_CODE
fi

echo ""
echo "✓ InfoGap pipeline completed successfully"

# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #

echo ""
echo "=========================================================================="
echo "Job completed at: $(date)"
echo "Total duration: $SECONDS seconds"
echo "=========================================================================="
