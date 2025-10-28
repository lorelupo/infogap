#!/bin/bash

# Example usage:
# RUN_STEPS=step2 bash wikimt_slurm_pipeline.slurm
# RUN_STEPS=step2 slurm wikimt_slurm_pipeline.slurm

#SBATCH --job-name=wikigap_pipeline
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
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
N_SAMPLE=510
MODEL="gpt-5-mini"
INFOGAP_DIR="/home/Lupo/infogap"
INPUT_DATA_DIR="/data/wikimt"
TREATMENT_GROUP_FILE="treatment_group_rebalanced_after_sent_div_4B_d30.csv"
CONTROL_GROUP_FILE="control_group_rebalanced_after_sent_div_4B_d30.csv"

# Which steps to run: "all", "step2", or "step1"
# - "all": Run preprocessing (step 1) and pipeline (step 2)
# - "step1": Run only preprocessing
# - "step2": Run only pipeline (assumes preprocessing already done)
RUN_STEPS="${RUN_STEPS:-all}"

echo ""
echo "=========================================================================="
echo "Configuration:"
echo "  Target Language: $TARGET_LANG"
echo "  Sample Size: $N_SAMPLE"
echo "  Model: $MODEL"
echo "  Run Steps: $RUN_STEPS"
echo "  InfoGap Directory: $INFOGAP_DIR"
echo "  Input Data Directory: $INPUT_DATA_DIR"
echo "  Treatment Group File: $TREATMENT_GROUP_FILE"
echo "  Control Group File: $CONTROL_GROUP_FILE"
echo "=========================================================================="

# --------------------------------------------------------------------------- #
# Step 1: Run preprocessing script
# --------------------------------------------------------------------------- #

if [ "$RUN_STEPS" = "all" ] || [ "$RUN_STEPS" = "step1" ]; then
    echo ""
    echo "=========================================================================="
    echo "STEP 1: Running preprocessing script"
    echo "=========================================================================="
    echo ""

    python wikimt_sample_preprocessing.py \
        --tgt-lang $TARGET_LANG \
        --n-sample $N_SAMPLE \
        --infogap-dir $INFOGAP_DIR \
        --input-data-dir $INPUT_DATA_DIR \
        --treatment-group-file $TREATMENT_GROUP_FILE \
        --control-group-file $CONTROL_GROUP_FILE

    PREPROCESS_EXIT_CODE=$?

    if [ $PREPROCESS_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Preprocessing failed with exit code $PREPROCESS_EXIT_CODE"
        exit $PREPROCESS_EXIT_CODE
    fi

    echo ""
    echo "✓ Preprocessing completed successfully"
else
    echo ""
    echo "=========================================================================="
    echo "STEP 1: Skipping preprocessing (RUN_STEPS=$RUN_STEPS)"
    echo "=========================================================================="
fi

# --------------------------------------------------------------------------- #
# Step 1.5: Download LaBSE model if not cached
# --------------------------------------------------------------------------- #

if [ "$RUN_STEPS" = "all" ] || [ "$RUN_STEPS" = "step15" ]; then
    echo ""
    echo "=========================================================================="
    echo "STEP 1.5: Ensuring LaBSE model is available"
    echo "=========================================================================="
    echo ""

    python << 'PYTHON_EOF'
import os
import sys
from sentence_transformers import SentenceTransformer
import sentence_transformers

# Import after adding path
sys.path.insert(0, os.getcwd())
from packages.constants import HF_CACHE_DIR

print(f"sentence-transformers version: {sentence_transformers.__version__}")

# Set environment variable for HuggingFace cache
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR

# Get version
st_version = sentence_transformers.__version__
major_version = int(st_version.split('.')[0])

print(f"Downloading/loading LaBSE model...")
print(f"  Cache directory: {HF_CACHE_DIR}")

# Try multiple model identifiers for v5.x compatibility
if major_version >= 5:
    # For v5.x, try the exact HuggingFace repository name
    # The model might be under 'sentence-transformers/LaBSE' or just 'LaBSE'
    model_ids_to_try = [
        'sentence-transformers/LaBSE',
        'setu4993/LaBSE',  # Alternative repository that might have proper config
    ]
    
    model = None
    for model_id in model_ids_to_try:
        try:
            print(f"  Trying model ID: {model_id}")
            model = SentenceTransformer(model_id, trust_remote_code=False)
            
            # Check if it has correct pooling
            has_cls_pooling = False
            if hasattr(model, '_modules'):
                for name, module in model._modules.items():
                    if 'Pooling' in type(module).__name__:
                        config = module.get_config_dict()
                        if config.get('pooling_mode_cls_token'):
                            has_cls_pooling = True
                            break
            
            if has_cls_pooling:
                print(f"  ✓ Found model with correct CLS pooling: {model_id}")
                break
            else:
                print(f"  ⚠ Model {model_id} uses mean pooling, trying next...")
                model = None
        except Exception as e:
            print(f"  ✗ Failed to load {model_id}: {e}")
            continue
    
    if model is None:
        print("\n❌ ERROR: Could not find LaBSE model with correct CLS pooling")
        print("All attempted model IDs failed or had incorrect pooling configuration")
        print("\nPossible solutions:")
        print("  1. Downgrade sentence-transformers: pip install 'sentence-transformers<5.0'")
        print("  2. Manually download the model from https://huggingface.co/sentence-transformers/LaBSE")
        print("  3. Use the v2.x compatible model repository")
        sys.exit(1)
else:
    # v2.x
    print(f"  Using v2.x API")
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)

print(f"✓ Model loaded successfully")

# Verify pooling configuration
verified = False
if hasattr(model, '_modules'):
    for name, module in model._modules.items():
        if 'Pooling' in type(module).__name__:
            config = module.get_config_dict()
            print(f"\nPooling configuration:")
            print(f"  CLS token: {config.get('pooling_mode_cls_token', False)}")
            print(f"  Mean tokens: {config.get('pooling_mode_mean_tokens', False)}")
            
            if config.get('pooling_mode_cls_token'):
                print("✓ Model uses correct CLS token pooling")
                verified = True
            else:
                print("❌ ERROR: Model is using mean pooling instead of CLS token pooling!")
                print("This will produce incorrect embeddings.")
                sys.exit(1)

if not verified:
    print("⚠ Warning: Could not verify pooling configuration")
    sys.exit(1)
                
print("✓ LaBSE model ready with correct configuration")
PYTHON_EOF

    MODEL_DOWNLOAD_EXIT_CODE=$?

    if [ $MODEL_DOWNLOAD_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Model download/check failed with exit code $MODEL_DOWNLOAD_EXIT_CODE"
        exit $MODEL_DOWNLOAD_EXIT_CODE
    fi
else
    echo ""
    echo "=========================================================================="
    echo "STEP 1.5: Skipping LaBSE model check (RUN_STEPS=$RUN_STEPS)"
    echo "=========================================================================="
fi

# --------------------------------------------------------------------------- #
# Step 2: Run InfoGap pipeline
# --------------------------------------------------------------------------- #

if [ "$RUN_STEPS" = "all" ] || [ "$RUN_STEPS" = "step2" ]; then
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
else
    echo ""
    echo "=========================================================================="
    echo "STEP 2: Skipping InfoGap pipeline (RUN_STEPS=$RUN_STEPS)"
    echo "=========================================================================="
fi

# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #

echo ""
echo "=========================================================================="
echo "Job completed at: $(date)"
echo "Total duration: $SECONDS seconds"
echo "=========================================================================="
