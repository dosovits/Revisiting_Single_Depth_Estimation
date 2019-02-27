# MODEL_NAME="01_resnet_ft_on_mp3d"
#MODEL_NAME="02_resnet_ft_on_mp3d_lr_4x_smaller"
#MODEL_NAME="03_c02_suncg_mf"
MODEL_NAME="04_c02_suncg_me"
MODEL_PATH="trained_models/${MODEL_NAME}"
mkdir $MODEL_PATH

# NODE_BASE_URL=/data_ssd/data/data_dmytro python -utt train.py --architecture resnet --lr 0.00002 --epochs 10 --training-data-csv ./data/mp3d/train1.csv --model-name $MODEL_NAME --init-weights ./pretrained_model/model_resnet 2>&1 | tee ${MODEL_PATH}/train.log

#NODE_BASE_URL=/data_ssd/data/data_dmytro python -utt train.py --architecture resnet --lr 0.000005 --epochs 2 --training-data-csv ./data/mp3d/train1.csv --model-name $MODEL_NAME --init-weights ./pretrained_model/model_resnet 2>&1 | tee ${MODEL_PATH}/train.log

#NODE_BASE_URL=/data_ssd/data/data_dmytro python -utt train.py --architecture resnet --lr 0.000005 --epochs 3 --training-data-csv ./data/suncg/furnished/train1.csv --model-name $MODEL_NAME --init-weights ./pretrained_model/model_resnet 2>&1 | tee ${MODEL_PATH}/train.log

NODE_BASE_URL=/data_ssd/data/data_dmytro python -utt train.py --architecture resnet --lr 0.000005 --epochs 3 --training-data-csv ./data/suncg/empty/train1.csv --model-name $MODEL_NAME --init-weights ./pretrained_model/model_resnet 2>&1 | tee ${MODEL_PATH}/train.log
