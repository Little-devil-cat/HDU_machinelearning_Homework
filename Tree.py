from math import log
import operator
import copy


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵


def splitDataSet(dataSet, axis, value):
    result = []
    for data in dataSet:
        if data[axis] == value:  # 如果该行数据axis列的值为value
            r = data[:axis] + data[axis + 1:]  # 就把这一行除axis列以外的元素放入r
            result.append(r)  # 意思就是，当前是以axis列的特征作为决策树分支标准的，
    return result



def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益（率）
    bestFeature = -1  # 最优特征的索引值
    A = []

    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        ibaseEntroy = 0.0  # 数据集dataSet关于特征i的值的熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
            ibaseEntroy -= prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy  # 信息增益
        C = [0, 0]
        C[0] = i
        C[1] = infoGain
        A.append(C)
        infoGainRatio = (baseEntropy - newEntropy) / ibaseEntroy  # 按照i特征划分的信息增益比
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGainRatio  # 更新信息增益，找到最大的信息增益率
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


def chooseBestFeatureToSplit_rate(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征个数，除去最后一列的分类标签
    ent = calcShannonEnt(dataSet)  # 当前数据集的信息熵
    best_gain = 0.0  # 最佳信息熵
    best_feature_id = -1  # 最佳信息熵的特征编号
    infogain = 0.0  # 最佳信息增益率
    for i in range(numFeatures):
        unique_values = set([line[i] for line in dataSet])  # 获取数据集中当前特征的所有值
        new_ent = 0.0
        s = 0.0
        for value in unique_values:  # 遍历此特征所有的值
            sub_dataset = splitDataSet(dataSet, i, value)  # 按照这个值，获得子数据集
            prob = len(sub_dataset) / len(dataSet)  # 子数据集的长度 就是 此特征i值为value时的数据个数
            new_ent += prob * calcShannonEnt(sub_dataset)  # 例：𝑔(𝑍|𝐴)
            a = prob * log(prob, 2)
            if a == 0:
                a = 0.00000000000000001
            s -= a  # 例：𝑆𝑝𝑙𝑖𝑡𝐼𝑛𝑓𝑜𝑟_𝐴 (𝑍)
        gain = ent - new_ent  # 计算信息增益
        infogain = gain / s  # 计算信息增益率
        if infogain > best_gain:  # 如果此信息增益率更大
            best_gain = infogain  # 更新
            best_feature_id = i
    return best_feature_id


# 频率最高的类别函数 判断递归何时结束
def majorityCnt(classList):                 #参数classList在下面创建树的代码里，是每一行的最后一个特征
    classCount = {}
    for vote in classList:                   #将特征里的元素添加到新建的字典里作为键值，并统计该键值出现次数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]           #返回出现次数最多的键值


# 创建决策树
def createTree(dataset, labels):
    Lab = copy.deepcopy(labels)
    class_list = [data[-1] for data in dataset]  # 获取当前数据集每行的分类标签，贷款是/否
    if class_list.count(class_list[0]) == len(class_list):
        # 如果当前数据集中，属于同一类，则退出递归
        return class_list[0]  # 返回此类的标签

    # 获取最好的分类特征索引
    best_feature_id = chooseBestFeatureToSplit_rate(dataset)  # 编号为2
    # 获取该特征的名字
    best_feature_label = Lab[best_feature_id]  # 特征名为：车否

    # 这里直接使用字典变量来存储树信息，这对于回执树形图很重要
    my_tree = {best_feature_label: {}}  # 当前数据集选取最好的特征存储在bestFeat中
    del (Lab[best_feature_id])  # 删除已经在选取的特征  此时  labels = ['是否有车']

    # 取出最优列的值
    feature_values_set = set([data[best_feature_id] for data in dataset])  # ['是', '否']
    for value in feature_values_set:  # 对于此特征的每一个取值
        temp = splitDataSet(dataset, best_feature_id, value)  # 切分数据集，进行分支
        my_tree[best_feature_label][value] = createTree(temp, Lab)  # 对于每一个子节点递归建树
    return my_tree


def show_result(node, class_label, step=0):
    if type(node) == bool:
        node = str(node)
        if type(node) == str:
            print('-'*step + class_label + "：" + node)
            return
    for k1, v1 in node.items():
        for k2, v2 in v1.items():
            print('-'*step + k1 + ':' + str(k2))
            show_result(v2, class_label, step+1)


def get_class(node, data, step=0): # 递归函数
    if type(node) == bool: # 如果当前结点类型为str，而不是dict，说明找到结果了
        return node # 返回结果
    for k1, v1 in node.items(): # 找到特征名称
        for k2, v2 in v1.items(): # 找到决策树中此特征的取值
            if data[k1] == k2: # 如果当前数据中此特征的值等于k2
                res = get_class(v2, data, step+1) # 递归
                if res != None: # 如果找到的结果不是None，则一定是“是”或“否”
                    return res  # 直接返回


def predict_acc(test_res_list, res_list):
    #################################
    # test_res_list:测试集的正确答案
    # res_list:得到的答案
    #################################
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    for i in range(len(test_res_list)):
        if not test_res_list[i]:
            if not res_list[i] and res_list[i] is not None:
                TN += 1
            else:
                FN += 1
        if test_res_list[i]:
            if res_list[i] and res_list[i] is not None:
                TP += 1
            else:
                FP += 1
    acc = (TN + TP)/(TN + TP + FN + FP)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2*P*R)/(P+R)
    return acc, F1


def Vote_Tree(res_list, test_res_list):
    res = []
    T = 0
    F = 0
    compare = []
    for i in range(len(res_list[0])):
        for j in range(len(res_list)):
            compare.append(res_list[j][i])
            if j == len(res_list)-1:
                for k in compare:
                    if k:
                        T += 1
                    elif not k:
                        F += 1
                if T > F:
                    res.append(True)
                elif T < F:
                    res.append(False)
                elif T == F:
                    res.append(None)
                compare = []
                T = 0
                F = 0
    acc, F1 = predict_acc(test_res_list[0], res)
    print('多棵决策树投票完成，最终结果为：'+'\n')
    print(res)
    print('acc:'+str(acc)+'\n'+'F1:'+str(F1))



# 预测主函数
def predict(test_dataset, my_tree, labels):
    res_list = []
    test_res_list = []
    print("使用决策树预测：")
    for i in range(len(test_dataset)): # 遍历测试集
        data = {} # 将测试集的每一行数据，做个特征到特征值的映射
        for j in range(len(test_dataset[i])-1):
            data[labels[j]] = test_dataset[i][j]
        test_res_list.append(test_dataset[i][20])
        res = get_class(my_tree, data) # 根据决策树，找到结果
        res_list.append(res)
        # print(test_dataset[i], res) # 打印结果
    acc, F1 = predict_acc(test_res_list, res_list)
    print('acc:'+str(acc)+'\n'+'F1:'+str(F1))
    return res_list, test_res_list, acc, F1



