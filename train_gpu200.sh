EXPID=$(date +"%Y%m%d_%H%M%S")_train_without_GCN
HOST='127.0.1.1'
PORT='1'

NUM_GPU=3
CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
--config 'configs/train.yaml' \
--output_dir 'results' \
--checkpoint '/nas/data_2/wanhongz/code/MultiModal-DeepFake/ALBEF_4M.pth' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:10011 \
--token_momentum \
--world_size $NUM_GPU \
--model_save_epoch 100
