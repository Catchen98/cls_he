import json
import pandas as pd
import glob
import random
from tqdm import tqdm
import os
from collections import Counter
from numpy import random

image_id_list = []
PMC_2020_flag = False
PMC_2020_split_flag = False
PMC_2019_flag = False

Synthetic_flag = False
Synthetic_test_2019_flag = False
Synthetic_train_2019_flag = False

PMC_synthetic_selfGen_flag = False

# concate_csv_flag = not PMC_synthetic_selfGen_flag
concate_csv_flag = False

Endo_2020_split_flag = False
Endo_2020_flag = False
Endo_2020_inference_flag = True
# split_flag = True
if PMC_2020_split_flag or Endo_2020_split_flag:
    split_ratio_val_ = 0.2

if concate_csv_flag:
    # csv_file_path_1 = '/data/hejy/hejy_dp/datasets/clx_PMC/PMC_2020_selfGen_train_v2.csv'
    csv_file_path_2 = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_split_train.csv'
    # csv_file_path_3 = '/data/hejy/hejy_dp/datasets/clx_PMC_speconly/PMC_2020_selfGenTestSpecAreaOnly_train.csv'
    csv_file_path_3 = '/data/hejy/hejy_dp/datasets/clx_PMC_speconly/PMC_2020_selfGenTestSpecScatterlineOnly_train.csv'
    # csv_file_path_3= '/data/hejy/hejy_dp/datasets/clx_PMC_test/PMC_2020_selfGenTestarea_train.csv'

    # csv_file_path_3= '/data/hejy/hejy_dp/datasets/clx_PMC_spec/PMC_2020_selfGenSpec_train.csv'
    # csv_file_path_3 = '/data/hejy/hejy_dp/datasets/clx_PMC/PMC_2020_selfGen_train_+area.csv'
    # csv_file_path_3 = '/data/hejy/hejy_dp/datasets/clx_PMC/PMC_2020_selfGen_train_5000vInterval+area.csv'
    # csv_file_path_4 = '/data/hejy/hejy_dp/datasets/Chart_2019/PMC_2019_train.csv'
    # csv_file_path_4 = '/data/hejy/hejy_dp/datasets/Chart_2019/PMC_2019_vBox_train.csv'

    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_train_v2.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon+area_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_5000vInterval+area_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_5000vInterval+area_2019vBox_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_+area_2019vBox_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_spec_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_testarea_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_specAreaOnly_train.csv'
    csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_specScatterlineOnly_train.csv'

    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_splitSelfgenCon_5000vInterval+area_2019_train.csv'
    # csv_concate_file_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/PMC_2020_2019_train.csv'
    # for csv_file_path_in in [csv_file_path_1, csv_file_path_2]:
    #     csv_file_in = pd.read_csv(csv_file_path_in)
    #     csv_file_in.to_csv(csv_concate_file_path, index=False, mode='a+')
    csv_2concate_list = [csv_file_path_2, csv_file_path_3]
    csv_out = pd.concat([pd.read_csv(csv_file_path_in) for csv_file_path_in in csv_2concate_list])
    # csv_out = pd.concat([pd.read_csv(csv_file_path_in) for csv_file_path_in in [csv_file_path_2, csv_file_path_3, csv_file_path_4]])
    csv_out.to_csv(csv_concate_file_path, index=False)
    for csv_file in csv_2concate_list:
        print('=>',csv_file)
    print('concate done')
    print('=>',csv_concate_file_path)

elif Synthetic_flag:
    # json_dir_path = "/data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset/JSONs"
    # csv_dir_path = "/data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset/"
    json_dir_path = "/data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset_v2/ICPR/JSONs"
    csv_dir_path = "/data/hejy/hejy_dp/datasets/Adobe_Synthetic_Dataset_v2/"
    area_list = []; donut_list = []; hbox_list = []; hGroup_list = [];hStack_list = [];line_list = [];pie_list = [];polar_list = [];scatter_list = [];vbox_list=[];vGroup_list=[];vStack_list=[]
    type_list = ['area', 'donut','hbox', 'hGroup','hStack', 'line', 'pie', 'polar', 'scatter', 'vbox', 'vGroup', 'vStack']
    type_map_dict={'Area':'area', 'Donut':'donut','Horizontal box':'hbox', "Grouped horizontal bar":'hGroup', "Stacked horizontal bar":'hStack', "Line":'line',\
         "Pie":'pie', "Polar":'polar', "Scatter":'scatter',  "Vertical box":'vbox', "Grouped vertical bar":'vGroup', "Stacked vertical bar":'vStack'}
    type_list_list = [area_list, donut_list, hbox_list, hGroup_list, hStack_list, line_list, pie_list, polar_list, scatter_list, vbox_list, vGroup_list, vStack_list]
    # import ipdb;ipdb.set_trace()
    file_list = glob.glob(json_dir_path + '/*/*.json')
    random.shuffle(file_list)
    for i, json_file_path in tqdm(enumerate(file_list)):
        json_file = json.load(open(json_file_path,'rb'))
        chart_type = json_file['task1']['output']['chart_type']
        # if ' ' in chart_type:
        #     chart_type = '_'.join(chart_type.split(' '))
        if not chart_type:
            print("chart_type of %s is None" %json_file_path)
        for _ in type_list_list:
            _.append(0)
        type_list_list[type_list.index(type_map_dict[chart_type])][i] = 1
        image_id = csv_dir_path+"ICPR/Charts/" + type_map_dict[chart_type] + '/'+json_file_path.split('/')[-1].rstrip('.json')+".png"
        image_id_list.append(image_id)
    file_name = "Synthetic_2020_train.csv"

elif Synthetic_test_2019_flag:
    json_dir_path = "/data/hejy/hejy_dp/datasets/Chart_2019/Synthetic_2019/test_tasks/test_release/task1/gt_json"
    csv_dir_path = "/data/hejy/hejy_dp/datasets/Chart_2019"
    area_list = []; donut_list = []; hbox_list = []; hGroup_list = [];hStack_list = [];line_list = [];pie_list = [];polar_list = [];scatter_list = [];vbox_list=[];vGroup_list=[];vStack_list=[]
    type_list = ['area', 'donut','hbox', 'hGroup','hStack', 'line', 'pie', 'polar', 'scatter', 'vbox', 'vGroup', 'vStack']
    type_map_dict={'Area':'area', 'Donut':'donut','Horizontal box':'hbox', "Grouped horizontal bar":'hGroup', "Stacked horizontal bar":'hStack', "Line":'line',\
         "Pie":'pie', "Polar":'polar', "Scatter":'scatter',  "Vertical box":'vbox', "Grouped vertical bar":'vGroup', "Stacked vertical bar":'vStack'}
    type_list_list = [area_list, donut_list, hbox_list, hGroup_list, hStack_list, line_list, pie_list, polar_list, scatter_list, vbox_list, vGroup_list, vStack_list]
    type_overall_list = []

    file_list = glob.glob(json_dir_path + '/*.json')
    random.shuffle(file_list)
    for i, json_file_path in tqdm(enumerate(file_list)):
        json_file = json.load(open(json_file_path,'rb'))
        chart_type = json_file['task1']['output']['chart_type']
        if not chart_type:
            print("chart_type of %s is None" %json_file_path)
        for _ in type_list_list:
            _.append(0)
        type_list_list[type_list.index(type_map_dict[chart_type])][i] = 1
        image_id = os.path.join(json_file_path.split('gt_json')[0],'png',json_file_path.split('/')[-1].rstrip('.json')+'.png')
        # import ipdb;ipdb.set_trace()
        image_id_list.append(image_id)
        type_overall_list.append(chart_type)
    for key, value in Counter(type_overall_list).items():
        print("%s: %d"%(key, value))
    file_name = "Synthetic_2019_test.csv"
elif Synthetic_train_2019_flag:
    json_dir_path = "/data/hejy/hejy_dp/datasets/Chart_2019/Synthetic_2019/train_json_gt/json_gt"
    csv_dir_path = "/data/hejy/hejy_dp/datasets/Chart_2019"
    area_list = []; donut_list = []; hbox_list = []; hGroup_list = [];hStack_list = [];line_list = [];pie_list = [];polar_list = [];scatter_list = [];vbox_list=[];vGroup_list=[];vStack_list=[]
    type_list = ['area', 'donut','hbox', 'hGroup','hStack', 'line', 'pie', 'polar', 'scatter', 'vbox', 'vGroup', 'vStack']
    type_map_dict={'Area':'area', 'Donut':'donut','Horizontal box':'hbox', "Grouped horizontal bar":'hGroup', "Stacked horizontal bar":'hStack', "Line":'line',\
         "Pie":'pie', "Polar":'polar', "Scatter":'scatter',  "Vertical box":'vbox', "Grouped vertical bar":'vGroup', "Stacked vertical bar":'vStack'}
    type_list_list = [area_list, donut_list, hbox_list, hGroup_list, hStack_list, line_list, pie_list, polar_list, scatter_list, vbox_list, vGroup_list, vStack_list]
    type_overall_list = []

    file_list = glob.glob(json_dir_path + '/*.json')
    random.shuffle(file_list)
    for i, json_file_path in tqdm(enumerate(file_list)):
        json_file = json.load(open(json_file_path,'rb'))
        chart_type = json_file['task1']['output']['chart_type']
        if not chart_type:
            print("chart_type of %s is None" %json_file_path)
        for _ in type_list_list:
            _.append(0)
        type_list_list[type_list.index(type_map_dict[chart_type])][i] = 1
        image_id = os.path.join(json_file_path.split('train_json_gt')[0],'images_train',json_file_path.split('/')[-1].rstrip('.json')+'.png')
        # import ipdb;ipdb.set_trace()
        image_id_list.append(image_id)
        type_overall_list.append(chart_type)
    for key, value in Counter(type_overall_list).items():
        print("%s: %d"%(key, value))
    file_name = "Synthetic_2019_train.csv"
elif PMC_synthetic_selfGen_flag:
    # csv_dir_path = '/data/hejy/hejy_dp/datasets/clx_PMC/'
    # csv_dir_path = '/data/hejy/hejy_dp/datasets/clx_PMC_spec/'
    # csv_dir_path = '/data/hejy/hejy_dp/datasets/clx_PMC_test/'
    csv_dir_path = '/data/hejy/hejy_dp/datasets/clx_PMC_speconly/'
    area_list = []; heatmap_list = [];horizontal_bar_list = [];horizontal_interval_list = [];line_list = [];manhattan_list = [];map_list = []
    pie_list = [];scatter_list = [];scatter_line_list = [];surface_list = [];venn_list = [];vertical_bar_list = [];vertical_box_list = []
    vertical_interval_list = []
    
    type_list = ['area', 'heatmap','horizontal_bar', 'horizontal_interval', 'line', 'manhattan','map','pie',\
                'scatter', 'scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']
    type_list_list = [area_list, heatmap_list, horizontal_bar_list, horizontal_interval_list, line_list, manhattan_list, map_list, \
                     pie_list, scatter_list,scatter_line_list, surface_list, venn_list, vertical_bar_list,vertical_box_list, vertical_interval_list]
    for type_dir in glob.glob(csv_dir_path + '/*'):
        # import ipdb;ipdb.set_trace()
        chart_type = type_dir.split('/')[-1]
        file_list = glob.glob(type_dir + '/*')
        len_list = len(type_list_list[0])
        for i, img_path in tqdm(enumerate(file_list)):
            for _ in type_list_list:
                _.append(0)
            type_list_list[type_list.index(chart_type)][i+len_list] = 1
            image_id = img_path
            image_id_list.append(image_id) 
    num = len(image_id_list)
    id_list = [i for i in range(num)]
    random.shuffle(id_list)
    image_id_list_s = [image_id_list[i] for i in id_list]
    type_list_list_s = []
    for one_type_list in type_list_list:
        # import ipdb;ipdb.set_trace()
        one_type_list_s = [one_type_list[i] for i in id_list]
        type_list_list_s.append(one_type_list_s)
    
    image_id_list = image_id_list_s
    type_list_list = type_list_list_s
    # file_name = "PMC_2020_selfGen_train_5000vInterval+area.csv"
    # file_name = "PMC_2020_selfGenSpec_train.csv"
    # file_name = "PMC_2020_selfGenTestarea_train.csv"
    # file_name = "PMC_2020_selfGenTestSpecAreaOnly_train.csv"
    file_name = "PMC_2020_selfGenTestSpecScatterlineOnly_train.csv"

    
elif PMC_2020_flag:
    json_dir_path = "/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/annotations"
    csv_dir_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/'
    area_list = []; heatmap_list = [];horizontal_bar_list = [];horizontal_interval_list = [];line_list = [];manhattan_list = [];map_list = []
    pie_list = [];scatter_list = [];scatter_line_list = [];surface_list = [];venn_list = [];vertical_bar_list = [];vertical_box_list = []
    vertical_interval_list = []
    
    type_list = ['area', 'heatmap','horizontal_bar', 'horizontal_interval', 'line', 'manhattan','map','pie',\
                'scatter', 'scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']
    type_list_list = [area_list, heatmap_list, horizontal_bar_list, horizontal_interval_list, line_list, manhattan_list, map_list, \
                     pie_list, scatter_list,scatter_line_list, surface_list, venn_list, vertical_bar_list,vertical_box_list, vertical_interval_list]
    
    file_list = glob.glob(json_dir_path + '/*/*.json')
    random.shuffle(file_list)
    for i, json_file_path in tqdm(enumerate(file_list)):
        json_file = json.load(open(json_file_path,'rb'))
        chart_type = json_file['task1']['output']['chart_type']
        if ' ' in chart_type:
            chart_type = '_'.join(chart_type.split(' '))
        if not chart_type:
            print("chart_type of %s is None" %json_file_path)
        for _ in type_list_list:
            _.append(0)
        type_list_list[type_list.index(chart_type)][i] = 1
        image_id = csv_dir_path + 'images/'+ chart_type + '/'+json_file_path.split('/')[-1].rstrip('.json')+'.jpg'
        image_id_list.append(image_id)    
    
    file_name = "PMC_2020_train.csv"
elif PMC_2020_split_flag:
    json_dir_path = "/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/annotations"
    csv_dir_path = '/data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.1/'
    area_list_train = []; heatmap_list_train = [];horizontal_bar_list_train = [];horizontal_interval_list_train = [];line_list_train = [];manhattan_list_train = [];map_list_train = []
    pie_list_train = [];scatter_list_train = [];scatter_line_list_train = [];surface_list_train = [];venn_list_train = [];vertical_bar_list_train = [];vertical_box_list_train = []
    vertical_interval_list_train = []
    
    area_list_val = []; heatmap_list_val = [];horizontal_bar_list_val = [];horizontal_interval_list_val = [];line_list_val = [];manhattan_list_val = [];map_list_val = []
    pie_list_val = [];scatter_list_val = [];scatter_line_list_val = [];surface_list_val = [];venn_list_val = [];vertical_bar_list_val = [];vertical_box_list_val = []
    vertical_interval_list_val = []
    type_list = ['area', 'heatmap','horizontal_bar', 'horizontal_interval', 'line', 'manhattan','map','pie',\
                'scatter', 'scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']
    type_list_list_train = [area_list_train, heatmap_list_train, horizontal_bar_list_train, horizontal_interval_list_train, line_list_train, manhattan_list_train, map_list_train, \
                     pie_list_train, scatter_list_train,scatter_line_list_train, surface_list_train, venn_list_train, vertical_bar_list_train,vertical_box_list_train, vertical_interval_list_train]
    type_list_list_val = [area_list_val, heatmap_list_val, horizontal_bar_list_val, horizontal_interval_list_val, line_list_val, manhattan_list_val, map_list_val, \
                     pie_list_val, scatter_list_val,scatter_line_list_val, surface_list_val, venn_list_val, vertical_bar_list_val,vertical_box_list_val, vertical_interval_list_val]                    
    
    image_id_list_train = []
    image_id_list_val = []
    for type_json_dir in glob.glob(json_dir_path + '/*'):
        # import ipdb;ipdb.set_trace()
        file_list = glob.glob(type_json_dir + '/*.json')
        random.shuffle(file_list)
        val_num = int(len(file_list) * split_ratio_val)
        len_train_list = len(type_list_list_train[0])
        len_val_list = len(type_list_list_val[0])

        for i, json_file_path in tqdm(enumerate(file_list)):
            json_file = json.load(open(json_file_path,'rb'))
            chart_type = json_file['task1']['output']['chart_type']
            if ' ' in chart_type:
                chart_type = '_'.join(chart_type.split(' '))
            if not chart_type:
                print("chart_type of %s is None" %json_file_path)
            if i >= val_num:
                for _ in type_list_list_train:
                    _.append(0)
                type_list_list_train[type_list.index(chart_type)][i+len_train_list-val_num] = 1
                image_id = csv_dir_path + 'images/'+ chart_type + '/'+json_file_path.split('/')[-1].rstrip('.json')+'.jpg'
                image_id_list_train.append(image_id)   
            else:
                for _ in type_list_list_val:
                    _.append(0)
                type_list_list_val[type_list.index(chart_type)][i+len_val_list] = 1
                image_id = csv_dir_path + 'images/'+ chart_type + '/'+json_file_path.split('/')[-1].rstrip('.json')+'.jpg'
                image_id_list_val.append(image_id)  
        
    file_name = "PMC_2020_train.csv"
    
    num = len(image_id_list_train)
    id_list = [i for i in range(num)]
    random.shuffle(id_list)
    image_id_list_train_s = [image_id_list_train[i] for i in id_list]
    type_list_list_train_s = []
    for one_type_list in type_list_list_train:
        # import ipdb;ipdb.set_trace()
        one_type_list_s = [one_type_list[i] for i in id_list]
        type_list_list_train_s.append(one_type_list_s)
    num = len(image_id_list_val)
    id_list = [i for i in range(num)]
    random.shuffle(id_list)
    image_id_list_val_s = [image_id_list_val[i] for i in id_list]
    type_list_list_val_s = []
    for one_type_list in type_list_list_val:
        # import ipdb;ipdb.set_trace()
        one_type_list_s = [one_type_list[i] for i in id_list]
        type_list_list_val_s.append(one_type_list_s)


    type_dict_val = {}
    type_dict_val['image_id'] = image_id_list_val_s
    for key, value in zip(type_list, type_list_list_val_s):
        type_dict_val[key] = value
    dataframe = pd.DataFrame(type_dict_val)
    dataframe.to_csv(csv_dir_path+'/'+file_name.replace('train','split_val'), index=False,sep=',' )
    type_dict_train = {}
    type_dict_train['image_id'] = image_id_list_train_s
    for key, value in zip(type_list, type_list_list_train_s):
        type_dict_train[key] = value
    dataframe = pd.DataFrame(type_dict_train)
    dataframe.to_csv(csv_dir_path+'/'+file_name.replace('train','split_train'), index=False,sep=',' )

elif Endo_2020_split_flag:
    json_dir_path = "/data/hejy/datasets/Endotect2020/labeled-images/"
    csv_dir_path = '/data/hejy/datasets/Endotect2020/'
    barretts_list_train = []; bbps_0_1_list_train = []; bbps_2_3_list_train = [];dyed_lifted_polyps_list_train = [];dyed_resection_margins_list_train = [];hemorrhoids_list_train = [];ileum_list_train = []
    impacted_stool_list_train = [];normal_cecum_list_train = [];normal_pylorus_list_train = [];normal_z_line = [];oesophagitis_a_list_train = [];oesophagitis_b_d_list_train = [];polyp_list_train = []
    retroflex_rectum_list_train = [];retroflex_stomach_list_train = [];short_segment_barretts_list_train=[]; ulcerative_colitis_0_1_list_train=[]; ulcerative_colitis_1_2_list_train=[]; ulcerative_colitis_2_3_list_train=[]
    ulcerative_colitis_grade_1_list_train=[]; ulcerative_colitis_grade_2_list_train=[];ulcerative_colitis_grade_3_list_train=[]

    barretts_list_val = []; bbps_0_1_list_val = []; bbps_2_3_list_val = [];dyed_lifted_polyps_list_val = [];dyed_resection_margins_list_val = [];hemorrhoids_list_val = [];ileum_list_val = []
    impacted_stool_list_val = [];normal_cecum_list_val = [];normal_pylorus_list_val = [];normal_z_line = [];oesophagitis_a_list_val = [];oesophagitis_b_d_list_val = [];polyp_list_val = []
    retroflex_rectum_list_val = [];retroflex_stomach_list_val = [];short_segment_barretts_list_val=[]; ulcerative_colitis_0_1_list_val=[]; ulcerative_colitis_1_2_list_val=[]; ulcerative_colitis_2_3_list_val=[]
    ulcerative_colitis_grade_1_list_val=[]; ulcerative_colitis_grade_2_list_val=[];ulcerative_colitis_grade_3_list_val=[]
    
    type_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'normal-cecum', 'normal-pylorus', 'normal-z-line','oesophagitis-a', 'oesophagitis-b-d', 'polyp', 'retroflex-rectum', 'retroflex-stomach', \
                'short-segment-barretts', 'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']
    type_map_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'cecum', 'pylorus', 'z-line','esophagitis-a', 'esophagitis-b-d', 'polyps', 'retroflex-rectum', 'retroflex-stomach', \
                'barretts-short-segment', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']
    
    type_list_list_train = [barretts_list_train, bbps_0_1_list_train,  bbps_2_3_list_train, dyed_lifted_polyps_list_train, dyed_resection_margins_list_train, hemorrhoids_list_train, ileum_list_train,
    impacted_stool_list_train, normal_cecum_list_train, normal_pylorus_list_train, normal_z_line, oesophagitis_a_list_train, oesophagitis_b_d_list_train, polyp_list_train,
    retroflex_rectum_list_train, retroflex_stomach_list_train, short_segment_barretts_list_train,  ulcerative_colitis_0_1_list_train,  ulcerative_colitis_1_2_list_train,  ulcerative_colitis_2_3_list_train,
    ulcerative_colitis_grade_1_list_train,  ulcerative_colitis_grade_2_list_train, ulcerative_colitis_grade_3_list_train,]

    type_list_list_val = [barretts_list_val, bbps_0_1_list_val,  bbps_2_3_list_val, dyed_lifted_polyps_list_val, dyed_resection_margins_list_val, hemorrhoids_list_val, ileum_list_val,
    impacted_stool_list_val, normal_cecum_list_val, normal_pylorus_list_val, normal_z_line, oesophagitis_a_list_val, oesophagitis_b_d_list_val, polyp_list_val,
    retroflex_rectum_list_val, retroflex_stomach_list_val, short_segment_barretts_list_val,  ulcerative_colitis_0_1_list_val,  ulcerative_colitis_1_2_list_val,  ulcerative_colitis_2_3_list_val,
    ulcerative_colitis_grade_1_list_val,  ulcerative_colitis_grade_2_list_val, ulcerative_colitis_grade_3_list_val,]
    
    image_id_list_train = []
    image_id_list_val = []
    type_overall_list_val = []
    import ipdb;ipdb.set_trace()
    cc=0
    for type_json_dir in glob.glob(json_dir_path + '*/*/*'):
        # import ipdb;ipdb.set_trace()
        file_list = glob.glob(type_json_dir + '/*.*')
        chart_type = type_json_dir.split('/')[-1]
        random.shuffle(file_list)
        # if chart_type == type_map_list[5] or chart_type == type_map_list[6]:
        #     split_ratio_val = split_ratio_val_ + 0.1
        # else:
        #     split_ratio_val = split_ratio_val_
        val_num = int(len(file_list) * split_ratio_val_)
        if val_num <= 1:
            val_num += 1
        cc += 1
        print(chart_type,'\n', val_num, '  ',cc)
        len_train_list = len(type_list_list_train[0])
        len_val_list = len(type_list_list_val[0])

        for i, json_file_path in tqdm(enumerate(file_list)):
            if not chart_type:
                print("chart_type of %s is None" %json_file_path)
            if i >= val_num:
                for _ in type_list_list_train:
                    _.append(0)
                type_list_list_train[type_map_list.index(chart_type)][i+len_train_list-val_num] = 1
                image_id_list_train.append(json_file_path)   
            else:
                for _ in type_list_list_val:
                    _.append(0)
                type_list_list_val[type_map_list.index(chart_type)][i+len_val_list] = 1
                image_id_list_val.append(json_file_path) 
                type_overall_list_val.append(chart_type) 
        
    file_name = "Endo_2020_train.csv"
    
    for key, value in Counter(type_overall_list_val).items():
        print("%s: %d"%(key, value)) 

    num = len(image_id_list_train)
    id_list = [i for i in range(num)]
    random.shuffle(id_list)
    image_id_list_train_s = [image_id_list_train[i] for i in id_list]
    type_list_list_train_s = []
    for one_type_list in type_list_list_train:
        # import ipdb;ipdb.set_trace()
        one_type_list_s = [one_type_list[i] for i in id_list]
        type_list_list_train_s.append(one_type_list_s)
    num = len(image_id_list_val)
    id_list = [i for i in range(num)]
    random.shuffle(id_list)
    image_id_list_val_s = [image_id_list_val[i] for i in id_list]
    type_list_list_val_s = []
    for one_type_list in type_list_list_val:
        # import ipdb;ipdb.set_trace()
        one_type_list_s = [one_type_list[i] for i in id_list]
        type_list_list_val_s.append(one_type_list_s)

    type_dict_val = {}
    type_dict_val['image_id'] = image_id_list_val_s
    for key, value in zip(type_list, type_list_list_val_s):
        type_dict_val[key] = value
    dataframe = pd.DataFrame(type_dict_val)
    dataframe.to_csv(csv_dir_path+'/'+file_name.replace('train','split_val'), index=False,sep=',' )
    type_dict_train = {}
    type_dict_train['image_id'] = image_id_list_train_s
    for key, value in zip(type_list, type_list_list_train_s):
        type_dict_train[key] = value
    dataframe = pd.DataFrame(type_dict_train)
    dataframe.to_csv(csv_dir_path+'/'+file_name.replace('train','split_train'), index=False,sep=',' )

elif Endo_2020_flag:
    json_dir_path = "/data/hejy/datasets/Endotect2020/labeled-images/"
    csv_dir_path = '/data/hejy/datasets/Endotect2020/'
    barretts_list = []; bbps_0_1_list = []; bbps_2_3_list = [];dyed_lifted_polyps_list = [];dyed_resection_margins_list = [];hemorrhoids_list = [];ileum_list = []
    impacted_stool_list = [];normal_cecum_list = [];normal_pylorus_list = [];normal_z_line = [];oesophagitis_a_list = [];oesophagitis_b_d_list = [];polyp_list = []
    retroflex_rectum_list = [];retroflex_stomach_list = [];short_segment_barretts_list=[]; ulcerative_colitis_0_1_list=[]; ulcerative_colitis_1_2_list=[]; ulcerative_colitis_2_3_list=[]
    ulcerative_colitis_grade_1_list=[]; ulcerative_colitis_grade_2_list=[];ulcerative_colitis_grade_3_list=[]
    
    type_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'normal-cecum', 'normal-pylorus', 'normal-z-line','oesophagitis-a', 'oesophagitis-b-d', 'polyp', 'retroflex-rectum', 'retroflex-stomach', \
                'short-segment-barretts', 'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']
    type_map_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'cecum', 'pylorus', 'z-line','esophagitis-a', 'esophagitis-b-d', 'polyps', 'retroflex-rectum', 'retroflex-stomach', \
                'barretts-short-segment', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']
    
    type_list_list = [barretts_list, bbps_0_1_list,  bbps_2_3_list, dyed_lifted_polyps_list, dyed_resection_margins_list, hemorrhoids_list, ileum_list,
    impacted_stool_list, normal_cecum_list, normal_pylorus_list, normal_z_line, oesophagitis_a_list, oesophagitis_b_d_list, polyp_list,
    retroflex_rectum_list, retroflex_stomach_list, short_segment_barretts_list,  ulcerative_colitis_0_1_list,  ulcerative_colitis_1_2_list,  ulcerative_colitis_2_3_list,
    ulcerative_colitis_grade_1_list,  ulcerative_colitis_grade_2_list, ulcerative_colitis_grade_3_list,]
    type_overall_list=[]
    type_map_dict= {}
    for i, cl in enumerate(type_list):
        type_map_dict[cl] = type_list_list[i]
    
    file_list = glob.glob(json_dir_path + '*/*/*/*.*')
    random.shuffle(file_list)
    cnt = 0
    for i, img_file_path in tqdm(enumerate(file_list)):
        for _ in type_list_list:
            _.append(0)
        
        get_flag = False
        for cla in type_list:
            # cla_new = cla.lstrip('normal-') if 'normal' in cla else cla
            print(type_map_list[type_list.index(cla)])
            if type_map_list[type_list.index(cla)] == img_file_path.split('/')[-2]:
                type_list_list[type_list.index(cla.lower())][cnt] = 1
                image_id_list.append(img_file_path)  
                type_overall_list.append(cla.lower())
                get_flag = True
        if not get_flag:
            import ipdb;ipdb.set_trace()
            print('image lost')
        
        cnt += 1
        # import ipdb;ipdb.set_trace()
    print(cnt)
    for key, value in Counter(type_overall_list).items():
        print("%s: %d"%(key, value)) 
    file_name = "Endo_2020_train.csv"  

elif Endo_2020_inference_flag:
    json_dir_path = "/data/hejy/datasets/Endotect2020/test/EndoTect_2020_Test_Dataset/"
    csv_dir_path = '/data/hejy/datasets/Endotect2020/test/'
    barretts_list = []; bbps_0_1_list = []; bbps_2_3_list = [];dyed_lifted_polyps_list = [];dyed_resection_margins_list = [];hemorrhoids_list = [];ileum_list = []
    impacted_stool_list = [];normal_cecum_list = [];normal_pylorus_list = [];normal_z_line = [];oesophagitis_a_list = [];oesophagitis_b_d_list = [];polyp_list = []
    retroflex_rectum_list = [];retroflex_stomach_list = [];short_segment_barretts_list=[]; ulcerative_colitis_0_1_list=[]; ulcerative_colitis_1_2_list=[]; ulcerative_colitis_2_3_list=[]
    ulcerative_colitis_grade_1_list=[]; ulcerative_colitis_grade_2_list=[];ulcerative_colitis_grade_3_list=[]
    
    type_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'normal-cecum', 'normal-pylorus', 'normal-z-line','oesophagitis-a', 'oesophagitis-b-d', 'polyp', 'retroflex-rectum', 'retroflex-stomach', \
                'short-segment-barretts', 'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']
    type_map_list = ['barretts', 'bbps-0-1', 'bbps-2-3', 'dyed-lifted-polyps', 'dyed-resection-margins','hemorrhoids', 'ileum', 'impacted-stool',\
                 'cecum', 'pylorus', 'z-line','esophagitis-a', 'esophagitis-b-d', 'polyps', 'retroflex-rectum', 'retroflex-stomach', \
                'barretts-short-segment', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2-3', 'ulcerative-colitis-grade-1', \
                'ulcerative-colitis-grade-2','ulcerative-colitis-grade-3']
    
    type_list_list = [barretts_list, bbps_0_1_list,  bbps_2_3_list, dyed_lifted_polyps_list, dyed_resection_margins_list, hemorrhoids_list, ileum_list,
    impacted_stool_list, normal_cecum_list, normal_pylorus_list, normal_z_line, oesophagitis_a_list, oesophagitis_b_d_list, polyp_list,
    retroflex_rectum_list, retroflex_stomach_list, short_segment_barretts_list,  ulcerative_colitis_0_1_list,  ulcerative_colitis_1_2_list,  ulcerative_colitis_2_3_list,
    ulcerative_colitis_grade_1_list,  ulcerative_colitis_grade_2_list, ulcerative_colitis_grade_3_list,]
    type_overall_list=[]
    type_map_dict= {}
    for i, cl in enumerate(type_list):
        type_map_dict[cl] = type_list_list[i]
    
    file_list = glob.glob(json_dir_path + '*.*')

    for i, img_file_path in tqdm(enumerate(file_list)):
        for _ in type_list_list:
            _.append(0)
        image_id_list.append(img_file_path) 
    file_name = "Endo_2020_4sub_task1.csv"  

elif PMC_2019_flag:
    json_dir_path = "/data/hejy/hejy_dp/datasets/Chart_2019/PMC_2019/annotations/task1/converted_JSON"
    csv_dir_path = '/data/hejy/hejy_dp/datasets/Chart_2019/'
    area_list = []; heatmap_list = [];horizontal_bar_list = [];horizontal_interval_list = [];line_list = [];manhattan_list = [];map_list = []
    pie_list = [];scatter_list = [];scatter_line_list = [];surface_list = [];venn_list = [];vertical_bar_list = [];vertical_box_list = []
    vertical_interval_list = [];horizontal_box_list = []
    
    type_list = ['area', 'heatmap','horizontal_bar', 'horizontal_interval', 'line', 'manhattan','map','pie',\
                'scatter', 'scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']#,'horizontal_box']
    type_list_list = [area_list, heatmap_list, horizontal_bar_list, horizontal_interval_list, line_list, manhattan_list, map_list, \
                     pie_list, scatter_list,scatter_line_list, surface_list, venn_list, vertical_bar_list,vertical_box_list, vertical_interval_list]#,horizontal_box_list]
    type_overall_list = []

    file_list = glob.glob(json_dir_path + '/*.json')
    random.shuffle(file_list)
    cnt = 0
    for i, json_file_path in tqdm(enumerate(file_list)):
        json_file = json.load(open(json_file_path,'rb'))
        chart_type = json_file['task1']['output']['chart_type']
        if ' ' in chart_type:
            chart_type = '_'.join(chart_type.split(' '))
        if not chart_type:
            print("chart_type of %s is None" %json_file_path)
        # if chart_type.lower() == "horizontal_box":
        if chart_type.lower() != "vertical_box":
            continue
        for _ in type_list_list:
            _.append(0)
        try:
            type_list_list[type_list.index(chart_type.lower())][cnt] = 1
        except:
            import ipdb;ipdb.set_trace()
        image_id = os.path.join(json_file_path.split('annotations')[0],'images/task1',json_file_path.split('/')[-1].rstrip('.json')+'.jpg')
        image_id_list.append(image_id)  
        type_overall_list.append(chart_type.lower())
        cnt += 1
        # import ipdb;ipdb.set_trace()
    for key, value in Counter(type_overall_list).items():
        print("%s: %d"%(key, value)) 
    # file_name = "PMC_2019_train.csv"  
    file_name = "PMC_2019_vBox_train.csv"  
 

if not PMC_2020_split_flag and not concate_csv_flag:
    type_dict = {}
    type_dict['image_id'] = image_id_list
    import ipdb;ipdb.set_trace()
    for key, value in zip(type_list, type_list_list):
        type_dict[key] = value
    dataframe = pd.DataFrame(type_dict)
    dataframe.to_csv(csv_dir_path+file_name, index=False,sep=',' )
    print('{} has been generated!'.format(csv_dir_path+file_name))

