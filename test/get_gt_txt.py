#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

VOC_2012_test_path = '../preparation/VOCdevkit/VOC2007'
image_ids = open(os.path.join(VOC_2012_test_path, 'ImageSets/Main/test.txt')).read().strip().split()
print(image_ids)

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in tqdm(image_ids):
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse(os.path.join(VOC_2012_test_path, "Annotations/"+image_id+".xml")).getroot()
        for obj in root.findall('object'):
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    continue
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")
