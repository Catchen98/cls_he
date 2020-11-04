#!/usr/bin/env bash

# fold=1
# CUDA_VISIBLE_DEVICES=6,7 python main.py train model_se101_${fold} --model se_resnext101_32x4d --fold ${fold} \
#         --root /share/Dataset/iMet/\
#         --n-epochs 40 --batch-size 32 --workers 8 \
#         --multi 1

# fold=0
# CUDA_VISIBLE_DEVICES=6,7 python main2.py train model_se101_f2_${fold} --model se_resnext101_32x4d --fold ${fold} \
#         --root /share/Dataset/iMet/\
#         --n-epochs 40 --batch-size 32 --workers 8 \
#         --multi 1

#iMet20

# fold=0
# CUDA_VISIBLE_DEVICES=6,7 python main.py train ckpts/model_resnext101_${fold} --model resnext101_32x8d --fold ${fold} \
#         --root /share/Dataset/iMet20/\
#         --n-epochs 40 --batch-size 32 --workers 8 \
#         --multi 1


# fold=0
# CUDA_VISIBLE_DEVICES=4,5 python main.py train ckpts/model_resnet18_${fold} --model resnet18 --fold ${fold} \
#         --root /share/Dataset/iMet20/\
#         --n-epochs 40 --batch-size 32 --workers 8 \
#         --multi 1

# fold=0
# CUDA_VISIBLE_DEVICES=4,5 python main_premodel.py train ckpts/model_senext101_${fold} --model se_resnext101_32x4d --fold ${fold} \
#         --root /share/Dataset/iMet20/\
#         --n-epochs 40 --batch-size 32 --workers 8 \
#         --multi 1

# 2020.04.24 by db

#fold=0
#CUDA_VISIBLE_DEVICES=6 python main_premodel.py train ckpts/model_resnest101_${fold} --model resnest101 --fold ${fold} \
#         --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#         --root /data/Dataset/iMet20/ \
#         --n-epochs 40 --batch-size 64 --workers 8 \
#         --multi 1

# 2020.04.25 by db

#fold=0
#CUDA_VISIBLE_DEVICES=6 python main_premodel.py train ckpts/model_res2net101_v1b_${fold} --model res2net101_v1b --fold ${fold} \
#         --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/res2net101_v1b_26w_4s-0812c246.pth' \
#         --root /data/Dataset/iMet20/ \
#         --n-epochs 40 --batch-size 64 --workers 8 \
#         --multi 1

#fold=0
#CUDA_VISIBLE_DEVICES=0,1 python main_premodel.py train ckpts/model_resnest200_v1b_${fold} --model resnest200 --fold ${fold} \
#         --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#         --root /data/Dataset/iMet20/ \
#         --n-epochs 40 --batch-size 64 --workers 8 \
#         --multi 1

# 2020.04.26 by db

#fold=0
#CUDA_VISIBLE_DEVICES=7 python main_premodel.py train ckpts/model_resnest101_autoaug_${fold} --model resnest101 --fold ${fold} \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug autoaug \
#        --root /data/Dataset/iMet20/ \
#        --n-epochs 40 --batch-size 64 --workers 8 \
#        --multi 1

#fold=0
#CUDA_VISIBLE_DEVICES=7 python main_premodel.py train ckpts/model_resnest101_autoaug_gem3_${fold} --model resnest101 --fold ${fold} \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug autoaug --pool_type gem_3 \
#        --root /data/Dataset/iMet20/ \
#        --n-epochs 40 --batch-size 64 --workers 8 \
#        --multi 1

#fold=0
#CUDA_VISIBLE_DEVICES=6 python main_premodel.py train ckpts/model_resnest101_autoaug_cutmix_${fold} --model resnest101 --fold ${fold} \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug autoaug --cutmix True \
#        --root /data/Dataset/iMet20/ \
#        --n-epochs 40 --batch-size 64 --workers 8 \
#        --multi 1

#fold=0
#CUDA_VISIBLE_DEVICES=7 python main_premodel.py train ckpts/model_resnest101_autoaug_focal_${fold} --model resnest101 --fold ${fold} \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug autoaug --focal_loss True \
#        --root /data/Dataset/iMet20/ \
#        --n-epochs 40 --batch-size 64 --workers 8 \
#        --multi 1

#fold=0
#CUDA_VISIBLE_DEVICES=5 python main_premodel.py train ckpts/model_resnest101_autoaug_lbs_${fold} --model resnest101 --fold ${fold} \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug autoaug --label_smoothing True \
#        --root /data/Dataset/iMet20/ \
#        --n-epochs 40 --batch-size 64 --workers 8 \
#        --multi 1


# for kaggle plant
#CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_resnest101_cutmix --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --cutmix True \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

#CUDA_VISIBLE_DEVICES=4 python train.py train pl_ckpts/model_resnest200_cutmix --model resnest200 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

#CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_resnest200_autuaug_cutmix_re --model resnest200 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

#CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_resnest200_autuaug_cutmix_re_lbs --model resnest200 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug \
#        --label_smoothing True \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_efficientnet-b5_autuaug_cutmix_re --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_autuaug_cutmix_re_albu --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug,albu \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

#CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_autuaug_cutmix_re_fixed --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

#CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_efficientnet-b5_autuaug_cutmix_re_fixed_lbs --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug \
#        --label_smoothing True \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

#CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_autuaug_cutmix_re_fixed_cj --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug autoaug,cj \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 \

# 0503 by dongb

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train.py train pl_ckpts/model_efficientnet-b5_albu --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_albu_mixup --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train.py train pl_ckpts/model_efficientnet-b5_albu_mixup_fixed --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --mixup True \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_efficientnet-b5_albu_mixup_lbs_fixed --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --mixup True \
#        --label_smoothing True \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# best now 0.976
# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_albu_re --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_albu_mixup_re --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --mixup True \
#        --aug re \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_efficientnet-b5_albu_cutmix --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_efficientnet-b5_albu_bl --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_albu_re_cutmix --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --cutmix True \
#        --aug re \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train.py train pl_ckpts/model_efficientnet-b5_albu_re_cj --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re,cj \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=5 python train.py train pl_ckpts/model_efficientnet-b7_albu_re --model efficientnet-b7 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_albu_re_rt90 --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re,rt90 \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=6 python train.py train pl_ckpts/model_efficientnet-b5_albu_re_rt90 --model efficientnet-b5 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re,rt90 \
#        --dataset_type cv2 \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4

# add by yao

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train.py train pl_ckpts/model-se_resnext101_32x4d-albu_re --model se_resnext101_32x4d \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=1 python train.py train pl_ckpts/model-resnet18-albu_re --model resnet18 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train.py train pl_ckpts/model-resnet18-albu_re-specific_mixup --model resnet18 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 --specific_mixup 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=2 python train.py train pl_ckpts/model-se_resnext101_32x4d-albu_re-specific_mixup --model se_resnext101_32x4d \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 --specific_mixup 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=1 python train.py train pl_ckpts/model-resnest101-albu_re --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train_ml.py train pl_ckpts/ml-model-resnet18-albu_re --model resnet18 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=1 python train_ml.py train pl_ckpts/ml-model-se_resnext101_32x4d-albu_re --model se_resnext101_32x4d \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest200-75117900.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=2 python train_ml.py train pl_ckpts/ml-model-resnest101-albu_re --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train.py train pl_ckpts/model-resnest101-albu_re-autoaug --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re,autoaug \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=1 python train.py train pl_ckpts/model-resnest101-albu_re-ricap --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  --ricap

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=2 python train.py train pl_ckpts/model-resnest101-albu_re-autoaug-ricap --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re,autoaug \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  --ricap


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=1 python train_pseudo.py train pl_ckpts/model-resnest101-albu_re-pseudo_99 --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  \
#        --pseudo_file 'pseudo_99.csv'


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=2 python train_pseudo.py train pl_ckpts/model-resnest101-albu_re-pseudo_98 --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  \
#        --pseudo_file 'pseudo_98.csv'

# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=3 python train_pseudo.py train pl_ckpts/model-resnest101-albu_re-pseudo_95 --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  \
#        --pseudo_file 'pseudo_95.csv'


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0 python train_pseudo.py train pl_ckpts/model-resnest101-albu_re-pseudo_95_pseudo_95 --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  \
#        --pseudo_file 'pseudo_95_pseudo_95.csv'


# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=1 python train_pseudo.py train pl_ckpts/model-resnest101-albu_re-pseudo_90 --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type cv2 \
#        --root /share/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4  \
#        --pseudo_file 'pseudo_90.csv'

# pil
# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=2 python train.py train pl_ckpts/pil-model-resnest101-albu_re --model resnest101 \
#        --model_path '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnest101-22405ba7.pth' \
#        --aug re \
#        --dataset_type pillow \
#        --root /data/Dataset/plant-pathology-2020-fgvc7 \
#        --n-epochs 30 --batch-size 8 --workers 4 
#/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN \
# export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=7 python train_kfold.py train pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_freeze --model resnet50 \
       --aug re,autoaug \
       --dataset_type pillow \
       --root /data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1 \
       --n-epochs 30 --batch-size 192 --workers 12 \
       --arc_bnbias_flag True\
       --freeze_backbone_flag True\
       # --model_path /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_bnbias+/best-model.pt \
       # --focal_loss True\


