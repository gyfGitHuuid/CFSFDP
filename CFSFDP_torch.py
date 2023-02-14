# -*- coding: utf-8 -*-
# Clustering by fast search and find of density peaks
from collections import Counter

import torch


# 确定每点的最终分类
# densitySortArr密度从小到大的索引排序
# closestNodeIdArr是比自己密度大，距离自己最近点的id
# n代表一共有多少条数据


def extract_cluster(densitySortArr, closestNodeIdArr, classNum, gamma):
    n = densitySortArr.shape[0]
    # 初始化每一个点的类别
    labels = torch.full((n,), -1, dtype=torch.float64).to(torch.device('cuda'))
    corePoints = torch.argsort(-gamma)[: classNum]  # 选择
    # 将选择的聚类中心赋予类别
    # labels[corePoints] = range(corePoints.shape[0])
    for i in range(corePoints.shape[0]):
        labels[corePoints] = i
    # 将ndarrar数组转为list集合
    densitySortList = densitySortArr.tolist()
    # 将集合元素反转，即密度从大到小的排序索引
    densitySortList.reverse()
    # 循环赋值每一个元素的label
    for nodeId in densitySortList:
        if (labels[nodeId] == -1):
            # 如果nodeId节点没有类别
            # 首先获得closestNodeIdArr[nodeId] 比自己密度大，且距离自己最近的点的索引
            # 将比自己密度大，且距离自己最近的点的类别复制给nodeId
            labels[nodeId] = labels[int(closestNodeIdArr[nodeId].item())]
    return corePoints, labels


def CFSFDP(data, dc):
    n, m = data.shape
    # 制作任意两点之间的距离矩阵
    # print(squareform(pdist(data.cpu().numpy(), 'euclidean')))
    # print(torch.norm(data[:, None] - data, dim=2, p=2))
    torch.cuda.empty_cache()
    temp = data[:, None] - data
    disMat = torch.norm(temp, dim=2, p=2)
    # print(disMat)
    # 计算每一个点的密度（在dc的圆中包含几个点）
    densityArr = torch.where(disMat < dc, 1, 0).sum(dim=1)
    # 将数据点按照密度大小进行排序（从小到大）
    densitySortArr = torch.argsort(densityArr, stable=True)
    # 初始化，比自己密度大的且最近的距离
    closestDisOverSelfDensity = torch.zeros((n,), dtype=torch.float64).to(torch.device('cuda'))
    # 初始化，比自己密度大的且最近的距离对应的节点id
    closestNodeIdArr = torch.zeros((n,), dtype=torch.float64).to(torch.device('cuda'))
    # 从密度最小的点开始遍历
    for index, nodeId in enumerate(densitySortArr):
        #  点密度大于当前点的点集合
        nodeIdArr = densitySortArr[index + 1:]
        # print(nodeIdArr.shape[0])
        # 如果不是密度最大的点
        if nodeIdArr.shape[0] != 0:
            # 计算比自己密度大的点距离nodeId的距离集合
            largerDistArr = disMat[nodeId][nodeIdArr]
            # print(index, "----->", largerDistArr.shape, largerDistArr)
            # 寻找到比自己密度大，且最小的距离节点
            closestDisOverSelfDensity[nodeId] = torch.min(largerDistArr)
            # print(closestDisOverSelfDensity)
            # 寻找到最小值的索引，索引实在largerdist里面的索引（确保是比nodeId）节点大
            # 如果存在多个最近的节点，取第一个
            # 注意，这里是largerDistArr里面的索引
            min_distance_index = torch.nonzero(largerDistArr == closestDisOverSelfDensity[nodeId])[0][0]
            # 获得整个数据中的索引值
            closestNodeIdArr[nodeId] = nodeIdArr[min_distance_index]
        else:
            # 如果是密度最大的点，距离设置为最大，且其对应的ID设置为本身
            closestDisOverSelfDensity[nodeId] = torch.max(closestDisOverSelfDensity)
            closestNodeIdArr[nodeId] = nodeId
    #  由于密度和最短距离两个属性的数量级可能不一样，分别对两者做归一化使结果更平滑
    if torch.max(densityArr) == torch.min(densityArr):
        normal_den = densityArr - torch.min(densityArr)
    else:
        normal_den = (densityArr - torch.min(densityArr)) / (torch.max(densityArr) - torch.min(densityArr))
    normal_dis = (closestDisOverSelfDensity - torch.min(closestDisOverSelfDensity)) / (
            torch.max(closestDisOverSelfDensity) - torch.min(closestDisOverSelfDensity))
    gamma = normal_den * normal_dis

    return densityArr, densitySortArr, closestDisOverSelfDensity, closestNodeIdArr, gamma
