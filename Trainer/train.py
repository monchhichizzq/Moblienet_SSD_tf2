import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from SSD_body.nets.ssd_training import MultiboxLoss, Generator
from SSD_body.nets.ssd import SSD300
from SSD_body.utils import BBoxUtility
# from SSD_body.utils import ModelCheckpoint
from SSD_body.anchors import get_mobilenet_anchors, get_vgg16_anchors
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算
# ----------------------------------------------------#
if __name__ == "__main__":
    backbone = 'vgg16'
    log_dir = "logs/"
    annotation_path = '../preparation/data_txt/kitti_obj_trainval.txt'

    NUM_CLASSES = 9
    input_shape = (300, 300, 3)
    if backbone is 'mobilenet':
        priors = get_mobilenet_anchors()
    elif backbone is 'vgg16':
        priors = get_vgg16_anchors()

    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    print('Train dataset: {0}, Test dataset:{1}'.format(num_train, num_val))

    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    # ------------------------------------------------------#
    model = SSD300(input_shape, backbone='vgg16', num_classes=NUM_CLASSES)
    if backbone is 'mobilenet':
        model.load_weights('../models/original/essay_mobilenet_ssd_weights.h5', by_name=True, skip_mismatch=True)
    if backbone is 'vgg16':
        model.load_weights('../models/original/ssd_weights.h5', by_name=True, skip_mismatch=True)


    # 训练参数设置

    # logging = TensorBoard(log_dir=log_dir)

    checkpoint =  tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    freeze_layer = 21
    for i in range(freeze_layer):
        model.layers[i].trainable = False
    # -------------------------------------#
    #   TF2的小bug，冻结后显存要求更大了
    # -------------------------------------#
    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        BATCH_SIZE = 32
        Lr = 5e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]), NUM_CLASSES)

        model.compile(optimizer=Adam(lr=Lr), loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit(gen.generate(True),
                  steps_per_epoch=num_train // BATCH_SIZE,
                  validation_data=gen.generate(False),
                  validation_steps=num_val // BATCH_SIZE,
                  epochs=Freeze_Epoch,
                  initial_epoch=Init_Epoch,
                  verbose=2,
                  # callbacks=[logging, checkpoint, reduce_lr, early_stopping])
                  callbacks=[checkpoint, reduce_lr, early_stopping])

    for i in range(freeze_layer):
        model.layers[i].trainable = True
    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        BATCH_SIZE = 32
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 300
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]), NUM_CLASSES)

        model.compile(optimizer=Adam(lr=Lr), loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit(gen.generate(True),
                  steps_per_epoch=num_train // BATCH_SIZE,
                  validation_data=gen.generate(False),
                  validation_steps=num_val // BATCH_SIZE,
                  epochs=Epoch,
                  initial_epoch=Freeze_Epoch,
                  verbose=2,
                  callbacks=[checkpoint, reduce_lr, early_stopping])
                  # callbacks=[logging, checkpoint, reduce_lr, early_stopping])
