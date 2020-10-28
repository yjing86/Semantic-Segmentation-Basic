import os
import cv2
import math
import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

CLS_num = {0: [0, 0, 0], 1: [0, 200, 0], 2: [150, 250, 0], 3: [150, 200, 150], 4: [200, 0, 200], 5: [150, 0, 250],
           6: [150, 150, 250], 7: [250, 200, 0], 8: [200, 200, 0], 9: [200, 0, 0], 10: [250, 0, 150],
           11: [200, 150, 150], 12: [250, 150, 150], 13: [0, 0, 200], 14: [0, 150, 200], 15: [0, 200, 250]}

def vis_label(label):
    last = []
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            last.append(CLS_num[label[i][j]])

    last = np.array(last).reshape((label.shape[0],label.shape[1], 3))
    # print(last)
    return last

def get_large(result, unit_size=576, step=288, width=3600, height=3400):
    '''
    本函数将小图预测的结果拼成大图

    :param: result 网络预测结果
    :param: unit_size step
    :param: width height 拼到一起的图片尺寸

    :return: 返回 N ch的大图mask
    '''
    # height在前面
    N_height = math.ceil((height-step)/step)+1
    N_width = math.ceil((width-step)/step)+1
    # print(N_height,N_width)

    side = (unit_size-step)//2   # 重叠部分的宽度

    for i in range(N_height):
        for j in range(N_width-1):
        # 遍历第i行，存在large_A 中
            tmp = result[N_width*i+j]
            tmp = tmp [side:side+step, side:side+step]
            if j == 0 :
                large_A = tmp   # large_A
            else:
                large_A = np.concatenate((large_A,tmp),axis=1)  # 同行拼接

        # 拼接每一行最后一列, 特殊情况
        tmp = result[N_width*(i+1)-1]
        tmp = tmp [side:side+step, -(width-(N_width-1)*step+side):-side]
        large_A = np.concatenate((large_A,tmp),axis=1)

        if i == 0 :
            large = large_A           # 首行，large_A赋值于large
        elif i < N_height-1:
            large = np.concatenate((large,large_A),axis=0)  # 同列拼接
        else:
        # 拼接最后一行, 特殊情况
            large_A = large_A[-(height-(N_height-1)*step):, :]
            large = np.concatenate((large,large_A),axis=0)  # 同列拼接

    return np.argmax(large,axis = 2)

def get_cls_index(result):
    # 将图片转化为clsindex
    CLS = {'[0, 0, 0]' : 0,
           '[0, 200, 0]' : 1,
           '[150, 250, 0]' : 2,
           '[150, 200, 150]' : 3,
           '[200, 0, 200]' : 4,
           '[150, 0, 250]' : 5,
           '[150, 150, 250]' : 6,
           '[250, 200, 0]' : 7,
           '[200, 200, 0]' : 8,
           '[200, 0, 0]' : 9,
           '[250, 0, 150]' : 10,
           '[200, 150, 150]' : 11,
           '[250, 150, 150]' : 12,
           '[0, 0, 200]' : 13,
           '[0, 150, 200]' : 14,
           '[0, 200, 250]' : 15}
    result_cls = np.zeros((result.shape[0],result.shape[1]),dtype=np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result_cls[i][j] = CLS['{}'.format(list(result[i][j]))]   # 通过字典形式快速索引类别编号
    return result_cls

def to_onehot(npy):
    # 将label转化为onehot
    h,w = npy.shape
    onehot_axis = (np.arange(16)==npy.reshape(h*w,1)).astype(np.int)
    #onehot = onehot_axis.reshape(h,w,15)
    return onehot_axis

def calculate(y_pred, y_true, cls_num = 16, cm=False):
    # 输入两个numpy,形为一位数组，比较他们之间的关系
    # cls_num 类别数，默认为16。cm为是否返回混淆矩阵。
    if cm:
        CM = sklearn.metrics.confusion_matrix(y_pred, y_true, labels=np.arange(cls_num))
    CP = sklearn.metrics.classification_report(y_pred, y_true, labels=np.arange(cls_num), output_dict= True)
    try:
        P = CP['micro avg']['precision']
        R = CP['micro avg']['recall']
        ACC = F1_micro = CP['micro avg']['f1-score']
    except:
        P = sklearn.metrics.precision_score(y_pred, y_true, labels=np.arange(cls_num), average='micro')
        R = sklearn.metrics.recall_score(y_pred, y_true, labels=np.arange(cls_num), average='micro')
        ACC = sklearn.metrics.accuracy_score(y_pred, y_true)
        F1_micro = 2*P*R/(R+P)
    F1 = CP['macro avg']['f1-score']
    F1_w = CP['weighted avg']['f1-score']

    Kappa = sklearn.metrics.cohen_kappa_score(y_pred, y_true, labels=np.arange(cls_num))
    Jaccard = sklearn.metrics.jaccard_score(y_pred, y_true, labels=np.arange(cls_num), average='macro')
    Jaccard_w = sklearn.metrics.jaccard_score(y_pred, y_true, labels=np.arange(cls_num), average='weighted')
    Jaccard_micro = sklearn.metrics.jaccard_score(y_pred, y_true, labels=np.arange(cls_num), average='micro')

    if cm:
        return {'CM':CM, 'CP':CP, 'P':P, 'R':R, 'ACC':ACC, 'K':Kappa, 'F1':F1,'F1-W':F1_w, 'F1-micro':F1_micro,
                'J':Jaccard, 'J-W':Jaccard_w, 'J-micro':Jaccard_micro}
    else:
        return {'CP':CP, 'P':P, 'R':R, 'ACC':ACC, 'K':Kappa, 'F1':F1,'F1-W':F1_w, 'F1-micro':F1_micro,
                'J':Jaccard, 'J-W':Jaccard_w, 'J-micro':Jaccard_micro}

def plot_confusion_matrix(cm, labels_name, title=' ', intFlag=False):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 归一化，其中[:, np.newaxis]将一维结果二维化
    plt.figure(figsize=(14,12), dpi=240)
    #plt.grid()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.binary)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()

    num_local = np.array(range(len(labels_name)))
    # labels_name = ['unknown', 'paddy field', 'irrigated land', 'dry land', 'garden land',
    #                'arbor forest', 'shrub land', 'natural meadow', 'artificial meadow',
    #                'industrial land', 'urban residential', 'rural residential', 'traffic land',
    #                'river','lake','pond',]
    plt.xticks(num_local, labels_name, fontsize=11, rotation=45)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=11)    # 将标签印在y轴坐标上
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)

    # plt.gca().set_xticks(num_local, minor=True)
    # plt.gca().set_yticks(num_local, minor=True)
    # plt.gca().xaxis.set_ticks_position('True')
    # plt.gca().yaxis.set_ticks_position('none')
    #plt.grid(True, which='minor', linestyle='-')
    #plt.gcf().subplots_adjust(bottom=-0.15)

    # 在图上标注数量
    for x_val in range(len(labels_name)):
        for y_val in range(len(labels_name)):
            if (intFlag):
                c = cm[x_val][y_val]
                plt.text(y_val, x_val, "%d" % (c,),
                         color='red', fontsize=9, va='center', ha='center')
            else:
                c = cm_normalized[x_val][y_val]
                if (c > 0.01):
                    #这里是绘制数字，可以对数字大小和颜色进行修改
                    plt.text(y_val, x_val, "%0.2f" % (c,),
                             color='red', fontsize=9, va='center', ha='center')
                else:
                    plt.text(y_val, x_val, "%d" % (0,),
                             color='red', fontsize=9, va='center', ha='center')
