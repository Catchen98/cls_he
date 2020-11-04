import pandas as pd
import os.path as osp
import numpy as np
from tqdm import tqdm
type_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'normal-cecum', 'normal-pylorus', 'normal-z-line','oesophagitis-a', 'oesophagitis-b-d', 'polyp', 'retroflex-rectum', 'retroflex-stomach', \
                'short-segment-barretts', 'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']

input_resultPath_list=[
    "2Assemble/result_from_efficientnet-b7.csv",\
    '2Assemble/result_from_resnet50.csv',
    ]
file_list = []
for file_path in input_resultPath_list:
    file_ = pd.read_csv(file_path)
    file_list.append(file_)

img_num, N_CLASSES = file_list[0].shape
N_CLASSES -= 1

name_list = []
pred_type_list = []
pred_type_conf_list = []
for i in tqdm(range(img_num)):
    preds_confs = np.zeros((1,N_CLASSES))
    img_name = osp.basename(file_list[0].loc[i][0])
    for j in range(len(input_resultPath_list)):
        # print(i)
        line = file_list[j].loc[i]
        assert osp.basename(line[0]) == img_name
        confs = line.values[1:24][np.newaxis,:]
        preds_confs = np.concatenate((preds_confs,confs),0)
    preds_confs = np.delete(preds_confs, 0, axis = 0).mean(axis=0)
    chart_type = type_list[np.argmax(preds_confs)]
    chart_type_conf = preds_confs.max()
    name_list.append(img_name)
    pred_type_list.append(chart_type)
    pred_type_conf_list.append(chart_type_conf)
csv_file = 'endotect20_DeepBlueAI_detection_verAssemble.csv'
csv_dict = {}
csv_dict['imagename'] = name_list 
csv_dict['classification decision'] = pred_type_list
csv_dict['confidence value'] = pred_type_conf_list
dataframe = pd.DataFrame(csv_dict)
dataframe.to_csv(csv_file,  header=False, index=False, sep=',')
print('submission files => {} finished'.format(csv_file))