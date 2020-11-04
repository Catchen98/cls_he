#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnet18 \
#     --ckpt_file pl_ckpts/model-resnet18-albu_re-specific_mixup/best-model.pt

# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model se_resnext101_32x4d \
#     --ckpt_file pl_ckpts/model-se_resnext101_32x4d-albu_re/best-model.pt


# CUDA_VISIBLE_DEVICES=2 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/model-resnest101-albu_re/best-model.pt

# CUDA_VISIBLE_DEVICES=2 python predict_ml.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/ml-model-resnest101-albu_re/best-model.pt


# CUDA_VISIBLE_DEVICES=2 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/model-resnest101-albu_re-autoaug-ricap/best-model.pt
    # --ckpt_file pl_ckpts/model-resnest101-albu_re-ricap/best-model.pt
    # --ckpt_file pl_ckpts/model-resnest101-albu_re-autoaug/best-model.pt

# CUDA_VISIBLE_DEVICES=2 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_90/best-model.pt    

    # --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_95_pseudo_95/best-model.pt    


    # --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_99/best-model.pt    

    # --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_95/best-model.pt    

# CUDA_VISIBLE_DEVICES=2 python predict_pil.py \
#     --root /data/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/pil-model-resnest101-albu_re/best-model.pt

# pseudo label
# python pseudo_label.py --save_file pseudo_99.csv --prob_thr 0.99
# python pseudo_label.py --save_file pseudo_98.csv --prob_thr 0.98
# python pseudo_label.py --save_file pseudo_95.csv --prob_thr 0.95
# python pseudo_label.py --save_file pseudo_95_pseudo_95.csv --prob_thr 0.95
# python pseudo_label.py --save_file pseudo_90.csv --prob_thr 0.90

# CUDA_VISIBLE_DEVICES=4 python predict.py \
#     --root /data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN \
#     --model resnest101 \
#     --ckpt_file /data/hejy/hejy_dp/cls_he/pl_ckpts/pil-model-resnest101-albu_re_autoaug/best-model.pt 
# /data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset/synthetic_test.csv
# CUDA_VISIBLE_DEVICES=7 python predict_eval.py \
#     --root  /data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_split_val.csv \
#     --model resnet50 \
#     --batch_size 48\
#     --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase2/kfold_Chart_pil-model-resnet50-albu_re_autoaug_224/fold0_best-model.pt

#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_expand/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_SpecScatterlineonly/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_SpecAreaonly/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_zgaug/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_testarea/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_noPretrain/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_spec/best-model.pt
#     #  --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_448_traintest_concate_+area_2019vBox/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_+area/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_+area_2019vBox/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_5000vInterval+area_2019vBox/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_5000vInterval+area_2019/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_5000vInterval+area/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_verticalIntervalExpand/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_15/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_2019/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate_verticalIntervalExpand/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase4/Chart_pil-model-resnet50-albu_re_autoaug_224_traintest_concate/best-model.pt
#     #--ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase2/kfold_Chart_pil-model-resnet50-albu_re_autoaug_224_focalloss/fold0_best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts/Chart_pil-model-efficientnet-b7-albu_re_autoaug/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase2/kfold_Chart_pil-model-resnet50-albu_re_autoaug_224/fold0_best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts/Chart_pil-model-resnet101-albu_re_autoaug/best-model.pt
#     # --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts/Chart_pil-model-resnet101-albu_re_autoaug/best-model.pt

CUDA_VISIBLE_DEVICES=7 python predict_eval.py \
    --root  /data/hejy/hejy_dp/datasets/Chart_test/PMC_2020_4sub.csv\
    --model resnet50 \
    --batch_size 48\
    --ckpt_file /data/hejy/hejy_dp/cls_he_15/pl_ckpts_phrase2/kfold_Chart_pil-model-resnet50-albu_re_autoaug_224/fold0_best-model.pt\
    --test_flag True

# #eval syn
# CUDA_VISIBLE_DEVICES=7 python predict_eval.py \
#     --root  /data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset_v2/Synthetic_2020_test.csv\
#     --model efficientnet-b7 \
#     --batch_size 48\
#     --ckpt_file /data/hejy/hejy_dp/cls_he/pl_ckpts/KFold_Chart_pil-model-efficientnet-b7-albu_re_autoaug_244/fold1_best-model.pt\

#submit syn
# CUDA_VISIBLE_DEVICES=7 python predict_eval.py \
#     --root  /data/hejy/hejy_dp/datasets/Chart_test/Syn_2020_4sub.csv\
#     --model efficientnet-b7 \
#     --batch_size 128\
#     --ckpt_file /data/hejy/hejy_dp/cls_he/pl_ckpts/KFold_Chart_pil-model-efficientnet-b7-albu_re_autoaug_244/fold1_best-model.pt\
#     --test_flag True \
