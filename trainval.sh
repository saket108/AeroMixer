#!/bin/bash

PHASE=$1  # train, eval (image-only)
CFG_FILE="config_files/images/aeromixer_images.yaml"
TEST_WEIGHT=${2:-'checkpoints/model_final.pth'}

if [ ! -f "$CFG_FILE" ]; then
    echo "Missing config file: $CFG_FILE"
    exit 1
fi

if [ -z "$PHASE" ]; then
    echo "Usage:"
    echo "  bash trainval.sh train"
    echo "  bash trainval.sh eval [checkpoint_path]"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate aeromixer

if [ "$PHASE" == 'train' ]
then
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=2024 train_net.py \
        --config-file ${CFG_FILE} \
        --transfer \
        --no-head \
        --use-tfboard
elif [ "$PHASE" == 'eval' ]
then
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=2405 test_net.py \
        --config-file ${CFG_FILE} \
        MODEL.WEIGHT ${TEST_WEIGHT}
else
    echo "Unsupported PHASE '$PHASE'. Use 'train' or 'eval'."
    exit 1
fi

echo "${PHASE} finished!"
