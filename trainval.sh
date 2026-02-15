#!/bin/bash

PHASE=$1  # train, eval
DATASET=${2:-images}  # images, videos

if [ "$DATASET" = "images" ]; then
    CFG_FILE="config_files/images/aeromixer_images.yaml"
elif [ "$DATASET" = "videos" ] || [ "$DATASET" = "video" ]; then
    CFG_FILE="config_files/videos/aeromixer_videos.yaml"
else
    echo "Unsupported DATASET '$DATASET'. Use 'images' or 'videos'."
    exit 1
fi

TEST_WEIGHT=${3:-'checkpoints/model_final.pth'}

eval "$(conda shell.bash hook)"
conda activate aeromixer

if [ $PHASE == 'train' ]
then
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=2024 train_net.py \
        --config-file ${CFG_FILE} \
        --transfer \
        --no-head \
        --use-tfboard
elif [ $PHASE == 'eval' ]
then
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=2405 test_net.py \
        --config-file ${CFG_FILE} \
        MODEL.WEIGHT ${TEST_WEIGHT}
fi

echo "${PHASE} finished!"
