#----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from SSD_body.inference.ssd_test import SSD
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from SSD_body.utils import BBoxUtility,letterbox_image,ssd_correct_boxes
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class mAP_SSD(SSD):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])
        crop_img,x_offset,y_offset = letterbox_image(image, (self.model_image_size[0],self.model_image_size[1]))
        photo = np.array(crop_img,dtype = np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.reshape(photo,[1,self.model_image_size[0],self.model_image_size[1],3]))
        preds = self.ssd_model.predict(photo)

        # 将预测结果进行解码
        results = self.bbox_util.detection_out(preds, confidence_threshold=self.confidence)
        
        if len(results[0])<=0:
            f.close()
            return

        # 筛选出其中得分高于confidence的框
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)


        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)-1]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

if __name__=='__main__':
    ssd = mAP_SSD()
    VOC_2012_test_path = '../preparation/VOCdevkit/VOC2007'
    image_ids = open(os.path.join(VOC_2012_test_path, 'ImageSets/Main/test.txt')).read().strip().split()

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOC_2012_test_path, "JPEGImages/"+image_id+".jpg")
        image = Image.open(image_path)
        # 开启后在之后计算mAP可以可视化
        # image.save("./input/images-optional/"+image_id+".jpg")
        ssd.detect_image(image_id,image)


    print("Conversion completed!")
