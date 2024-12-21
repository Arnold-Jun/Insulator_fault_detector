import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm



def load_data_VOC2007(path):
    data = []
    # -------------------------------------------------------------#
    #   对于每一个xml都寻找box
    # -------------------------------------------------------------#
    for xml_file in tqdm(glob.glob('{}/*xml'.format(path))):
        tree = ET.parse(xml_file)

        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        # -------------------------------------------------------------#
        #   对于每一个目标都获得它的宽高
        # -------------------------------------------------------------#
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


def evaluate(box, cluster):

    row = box.shape[0]


    distance = np.empty((row, len(cluster)))

    cluster = cluster

    for i in range(row):
        distance[i] = 1 - cas_iou(box[i], cluster)

    near = np.argmin(distance, axis=1)
    fitness =avg_iou(box, cluster)

    return fitness, near


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]
    np.seterr(divide="ignore", invalid="ignore")
    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 - intersection)
    # assert not (area1 + area2 - intersection) <= 0

    return iou

def avg_iou(box, cluster):
    np.seterr(all='igno re')
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])



def decode(genes):
    genes_decoded = []
    for j in range(0,9):
            width = round((int(genes[0 + 18 * j:9 + j * 18], 2)))
            height = round((int(genes[9 + j * 18:18 + j * 18], 2)))
            genes_decoded.append([width / 512,height / 512])

    return np.array(genes_decoded)


def kmeans(box, cluster, k):
    # -------------------------------------------------------------#
    #   取出一共有多少框
    # -------------------------------------------------------------#
    row = box.shape[0]  # (40138, 2)

    # -------------------------------------------------------------#
    #   每个框各个点的位置
    # -------------------------------------------------------------#
    distance = np.empty((row, k))  # (40138, 9)

    last_clu = np.zeros((row,))

    np.random.seed()

    iter = 0
    while True:

        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        near = np.argmin(distance, axis=1)  # [1 1 3 ... 0 5 2] (40138,)

        if (last_clu == near).all():
            break

        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near
        iter += 1

        if iter >= 150:
            break
    fitness = avg_iou(box, cluster)

    return fitness, near, cluster
