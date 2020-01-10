# -*- coding: utf-8 -*-
# Clustering by fast search and find of density peaks
import numpy as np
import numpy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pandas as pd
import csv
# 确定每点的最终分类
# densitySortArr密度从小到大的索引排序
# closestNodeIdArr是比自己密度大，距离自己最近点的id
# n代表一共有多少条数据
def extract_cluster(densitySortArr,closestNodeIdArr, classNum,gamma):
    n=densitySortArr.shape[0]
    # 初始化每一个点的类别
    labels=np.full((n,),-1)
    corePoints =  np.argsort(-gamma)[: classNum]  # 选择
    # 将选择的聚类中心赋予类别
    labels[corePoints]=range(len(corePoints))
    # 将ndarrar数组转为list集合
    densitySortList=densitySortArr.tolist()
    # 将集合元素反转，即密度从大到小的排序索引
    densitySortList.reverse()
    # 循环赋值每一个元素的label
    for nodeId in densitySortList:
        if(labels[nodeId]==-1):
            # 如果nodeId节点没有类别
            # 首先获得closestNodeIdArr[nodeId] 比自己密度大，且距离自己最近的点的索引
            # 将比自己密度大，且距离自己最近的点的类别复制给nodeId
            labels[nodeId]=labels[closestNodeIdArr[nodeId]]
    return corePoints,labels

def CFSFDP(data,dc):
    n,m=data.shape
    # 制作任意两点之间的距离矩阵
    disMat = squareform(pdist(data,metric='euclidean'))
    # 计算每一个点的密度（在dc的圆中包含几个点）
    densityArr = np.where(disMat < dc, 1, 0).sum(axis=1)
    # 将数据点按照密度大小进行排序（从小到大）
    densitySortArr=np.argsort(densityArr)
    # 初始化，比自己密度大的且最近的距离
    closestDisOverSelfDensity = np.zeros((n,))
    # 初始化，比自己密度大的且最近的距离对应的节点id
    closestNodeIdArr = np.zeros((n,), dtype=np.int32)
    # 从密度最小的点开始遍历
    for index,nodeId in enumerate(densitySortArr):
        #  点密度大于当前点的点集合
        nodeIdArr = densitySortArr[index+1:]
        # 如果不是密度最大的点
        if nodeIdArr.size != 0:
            # 计算比自己密度大的点距离nodeId的距离集合
            largerDistArr = disMat[nodeId][nodeIdArr]
            # 寻找到比自己密度大，且最小的距离节点
            closestDisOverSelfDensity[nodeId] = np.min(largerDistArr)
            # 寻找到最小值的索引，索引实在largerdist里面的索引（确保是比nodeId）节点大
            # 如果存在多个最近的节点，取第一个
            # 注意，这里是largerDistArr里面的索引
            min_distance_index = np.argwhere(largerDistArr == closestDisOverSelfDensity[nodeId])[0][0]
            # 获得整个数据中的索引值
            closestNodeIdArr[nodeId] = nodeIdArr[min_distance_index]
        else:
            # 如果是密度最大的点，距离设置为最大，且其对应的ID设置为本身
            closestDisOverSelfDensity[nodeId] = np.max(closestDisOverSelfDensity)
            closestNodeIdArr[nodeId] = nodeId
    #  由于密度和最短距离两个属性的数量级可能不一样，分别对两者做归一化使结果更平滑
    normal_den = (densityArr - np.min(densityArr)) / (np.max(densityArr) - np.min(densityArr))
    normal_dis = (closestDisOverSelfDensity - np.min(closestDisOverSelfDensity)) / (
                np.max(closestDisOverSelfDensity) - np.min(closestDisOverSelfDensity))
    gamma = normal_den * normal_dis


    return densityArr,densitySortArr,closestDisOverSelfDensity,closestNodeIdArr,gamma




if __name__ == '__main__':
    data = np.loadtxt("C:/Users/Desktop/Data/T3.csv", delimiter=",")
    # 执行聚类算法
    densityArr,densitySortArr,closestDisOverSelfDensity,closestNodeIdArr,gamma= CFSFDP(data,2)
    # 根据决策图提取类别
    #自动确定聚类中心数目
    G = []
    G = numpy.matrix.tolist(gamma)
    G.sort(reverse = True)
    K = []
    for i in range(len(G)-1):
        k =G[i] - G[i+1]
        K.append(k)
    ksum = 0
    for i in range(len(K)):
        ksum = ksum+K[i]
    R = ksum/len(K)
    Result = 1
    for i in range(len(K)):
        if K[i] > R:
            Result = Result + 1
    # 聚类中心数目传值
    classNum = Result
    #导出聚类中心坐标
    corePoints,labels = extract_cluster(densitySortArr, closestNodeIdArr, classNum, gamma)
    X = data[corePoints,0]
    Y = data[corePoints,1]
    CC = []
    for i in range(len(X)):
        CC.append([X[i],Y[i]])
    name1 = ['lat', 'lon']
    S = pd.DataFrame(columns=name1, data=CC)
    S.to_csv('C:/Users/Desktop/CC.csv', encoding='utf-8')
    #********
    #分类结果导出
    M = []
    R = []
    L = []
    for n in corePoints:
        M.append(n)
    with open("C:/Users/Desktop/Data/T3.csv", 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    for i in labels:
        L.append((i))
    for x in range(0,len(rows)):
        R.append([rows[x][0],rows[x][1],L[x]])
    name = ['lat', 'lon', 'code' ]
    SortR = pd.DataFrame(columns=name, data=R)
    SortR.to_csv('C:/Users/Desktop/Rt.csv', encoding='utf-8')