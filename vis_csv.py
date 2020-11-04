import os
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import glob
import sys
import numpy as np
import random
# reload(sys)
# sys.setdefaultencoding('utf-8')


type_list = ['area', 'heatmap','horizontal_bar', 'horizontal_interval', 'line', 'manhattan','map','pie',\
        'scatter', 'scatter-line','surface','venn','vertical_bar','vertical_box','vertical_interval']

false_cnt = 0

def vis_csv(csv_file_path):
    global false_cnt
    with open(csv_file_path) as f:
        lines = f.readlines()
        # random.shuffle(lines)
    output = []
    for line in tqdm(lines[1:]):
        line_parts = line.strip().split(',')
        img_name = line_parts[0]
        try:
            cat_score = np.asarray([float(_) for _ in line_parts[1:]])
        except:
            import ipdb;ipdb.set_trace()
        cls_type = type_list[np.argmax(cat_score)]
        var_info = [cls_type, img_name]
        vis(var_info)
    print(false_cnt)
    # with open(txt_file_path, 'w') as f:
    #     f.write('\n'.join(output))   


def vis(var_info):
    global false_cnt
    cls_type = var_info[0]
    img_name = var_info[1]

    img_save_path = '/data/hejy/hejy_dp/cls_he/plot'  
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path) 

    font_size = 20
    gap_size = 2
    max_text_num = 8
    bg_plate_height = font_size
    bg_plate_width = font_size * max_text_num
    box_plate_dist = 5 
    bg_plate_color = (100, 100, 100)
    text_color = (255, 255, 255)
    gt_text_color = (0, 255, 0 )

    frame_output_name = os.path.join(img_save_path, 'plot_'+os.path.basename(img_name))
    img = cv2.imread(img_name)
    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    attr_vis_cnt = 0

    vis_str = cls_type
    gt_str = img_name.split('/')[-2]
    if vis_str == gt_str:
        return
    false_cnt += 1
    draw = ImageDraw.Draw(image_pil)
    bg_plate_x1 = box_plate_dist
    bg_plate_y1 = (bg_plate_height + gap_size * 2) * attr_vis_cnt
    bg_plate_x2 = bg_plate_x1 + bg_plate_width
    bg_plate_y2 = bg_plate_y1 + bg_plate_height
    draw.rectangle((bg_plate_x1, bg_plate_y1, bg_plate_x2, bg_plate_y2), fill=bg_plate_color)
    font = ImageFont.truetype('/data/hejy/Scripts/FangZhengFangSongJianTi-1.ttf', font_size, encoding='utf-8')
    # font = ImageFont.load_default()
    text_plate_x1 = bg_plate_x1+3
    text_plate_y1 = bg_plate_y1
    # draw.text((text_plate_x1, text_plate_y1), vis_str, text_color)
    draw.text((text_plate_x1, text_plate_y1), 'pr:'+vis_str, text_color, font)
    
    attr_vis_cnt += 1
    bg_plate_y1 = (bg_plate_height + gap_size * 2) * attr_vis_cnt
    bg_plate_y2 = bg_plate_y1 + bg_plate_height
    text_plate_y1 = bg_plate_y1
    draw.rectangle((bg_plate_x1, bg_plate_y1, bg_plate_x2, bg_plate_y2), fill=bg_plate_color)
    draw.text((text_plate_x1, text_plate_y1), 'gt:'+gt_str, gt_text_color, font)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_output_name, img)
        
if __name__ == "__main__":
    csv_file_path = "/data/hejy/hejy_dp/cls_he/submission.csv"
    vis_csv(csv_file_path)
