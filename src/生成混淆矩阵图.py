import numpy as np
from segmentation.test import *

# CM_path = input("输入混淆矩阵numpy的保存地址")
# SAVENAME = input("输入网络结构名")
CM_path = "../results/01-baseline-3CH-UNet-resnet101_checkpoint/confusion_matrix_01-baseline-3CH-UNet-resnet101.npy"
CM = np.load(CM_path)

SAVENAME = CM_path.split('/')[2]
SAVEPATH = os.path.join("../results/", SAVENAME)

labels_name = ['unknown', 'paddy field', 'irrigated land', 'dry land', 'garden land',
               'arbor forest', 'shrub land', 'natural meadow', 'artificial meadow',
               'industrial land', 'urban residential', 'rural residential', 'traffic land',
               'river','lake','pond',]

plot_confusion_matrix(CM, labels_name, title=' ')

plt.savefig(os.path.join(SAVEPATH, 'confusion_matrix_{}.png'.format(SAVENAME)), format='png')
print("Done!")
