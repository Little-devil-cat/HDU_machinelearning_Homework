import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import Tree
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.utils import shuffle
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))


def Pokemon_Dataset():
    df = pd.read_csv('./data/pokemon_alopez247.csv')
    print('正在读取数据....')
    print('正在处理数据....')
    # print(df)
    # df.info()

    df.drop('Name', axis=1, inplace=True)  # 删除Name列，删除分类无关列
    df.drop('Number', axis=1, inplace=True)  # 删除Number列

    # Type_1 = df['Type_1'].drop_duplicates().values.tolist()  # 用于构建Type_mapping
    Type_mapping = {
        'Grass':1,
        'Fire':2,
        'Water':3,
        'Bug':4,
        'Normal':5,
        'Poison':6,
        'Electric':7,
        'Ground':8,
        'Fairy':9,
        'Fighting':10,
        'Psychic':11,
        'Rock':12,
        'Ghost':13,
        'Ice':14,
        'Dragon':15,
        'Dark':16,
        'Steel':17,
        'Flying':18,
    }  # 为了保持两列特征数值化后的一致性，手动构建字典
    df['Type_1'] = df['Type_1'].map(Type_mapping)  # 字符特征数值化，采用标签编码，独热码会导致维度灾难
    df['Type_2'] = df['Type_2'].map(Type_mapping)
    df.loc[df['Type_2'].isnull(), 'Type_2'] = 0  # nan项填0
    df.loc[df['Type_1'].isnull(), 'Type_1'] = 0

    # Egg = df['Egg_Group_1'].drop_duplicates().values.tolist()
    Egg_mapping = {
        'Grass':1,
        'Monster':2,
        'Water_1':3,
        'Bug':4,
        'Water_2':5,
        'Water_3':6,
        'Field':7,
        'Undiscovered':8,
        'Fairy':9,
        'Human-Like':10,
        'Mineral':11,
        'Amorphous':12,
        'Ditto':13,
        'Dragon':14,
        'Flying':15,
    }
    df['Egg_Group_1'] = df['Egg_Group_1'].map(Egg_mapping)  # 字符特征数值化，采用标签编码，独热码会导致维度灾难
    df['Egg_Group_2'] = df['Egg_Group_2'].map(Egg_mapping)
    df.loc[df['Egg_Group_2'].isnull(), 'Egg_Group_2'] = 0  # nan项填0
    df.loc[df['Egg_Group_1'].isnull(), 'Egg_Group_1'] = 0

    df.loc[df['Pr_Male'].isnull(), 'Pr_Male'] = 0

    class_le = LabelEncoder()  # 调用方法虽然方便，但是可解释性变差了
    # df['isLegendary'] = class_le.fit_transform(df['isLegendary'])
    df['Color'] = class_le.fit_transform(df['Color'])
    df['hasGender'] = class_le.fit_transform(df['hasGender'])
    df['hasMegaEvolution'] = class_le.fit_transform(df['hasMegaEvolution'])
    df['Body_Style'] = class_le.fit_transform(df['Body_Style'])


    df.info()
    return df


def Data_Scaling(df):  # 数据标准化和训练测试切分
    Zscore_df1 = df.iloc[:, 0:10]
    Zscore_df2 = df.iloc[:, 11:21]
    # label_df = df.iloc[:, 10]
    Zscore_df = pd.concat([Zscore_df1, Zscore_df2], axis=1)
    zscore = StandardScaler()
    zscore = zscore.fit_transform(Zscore_df)  # 数据标准化操作
    df_zscore = pd.DataFrame(zscore,index=Zscore_df.index,columns=Zscore_df.columns)
    df.update(df_zscore, join='left', overwrite=True, filter_func=None, errors='ignore')  # 按索引更新源Dataframe

    columns = list(df)
    columns.append(columns.pop(10))
    df = df[columns]

    isLegendary = df.loc[df['isLegendary'] == 1]
    notLegendary = df.loc[df['isLegendary'] != 1]
    isLegendary = shuffle(isLegendary)  # 随机打乱
    notLegendary = shuffle(notLegendary)

    notLegendary_split_result = np.array_split(notLegendary, 15)
    isLegendary_split_result = np.array_split(isLegendary, 15)

    test_data = pd.concat([notLegendary_split_result[14], isLegendary_split_result[14]], axis=0)
    del notLegendary_split_result[14]
    del isLegendary_split_result[14]

    isLegendary_train_data = isLegendary_split_result[0]
    for i in range(1, 14):
        isLegendary_train_data = pd.concat([isLegendary_train_data, isLegendary_split_result[i]], axis=0)

    train_data = []
    for i in range(14):
        train_data.append(pd.concat([notLegendary_split_result[i], isLegendary_train_data], axis=0))


    print('数据处理完毕....')
    return df, test_data, train_data


def Train_Tree(train_data, labels, Test_dataSet, used, train_num):
    res_list = []
    test_res_list = []
    acc_list = []
    F1_list = []
    for i in range(train_num):
        dataset = np.array(train_data[i])
        dataset = dataset.tolist()
        Tree1 = Tree.createTree(dataset, labels)
        # Tree.show_result(Tree1, class_label)
        res_transfer, test_res_transfer, acc, F1 = Tree.predict(Test_dataSet, Tree1, labels)
        res_list.append(res_transfer)
        test_res_list.append(test_res_transfer)
        acc_list.append(acc)
        F1_list.append(F1)
        used = i
    Tree.Vote_Tree(res_list, test_res_list)
    return used, acc_list, F1_list

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    PokemoData_df = Pokemon_Dataset()
    PokemoData_df, test_data, train_data = Data_Scaling(PokemoData_df)
    Labels = test_data.columns.values.tolist()
    Labels.pop()
        # ['Type_1', 'Type_2', 'Total', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Generation', 'Color', 'hasGender', 'Pr_Male', 'Egg_Group_1', 'Egg_Group_2', 'hasMegaEvolution', 'Height_m', 'Weight_kg', 'Catch_Rate', 'Body_Style']
    class_label = 'isLegendary'

    test_dataSet = np.array(test_data)
    test_dataSet = test_dataSet.tolist()

    used = 0
    Tree_train_num = 13  # 为减少F == T的情况，尽量使用奇数

    used, acc_list, F1_list = Train_Tree(train_data, Labels, test_dataSet, used, Tree_train_num)

    plt.plot(acc_list)
    plt.xlabel('Different Tree')
    plt.ylabel('acc')
    plt.grid(axis='x')
    plt.show()

    plt.plot(F1_list)
    plt.xlabel('Different Tree')
    plt.ylabel('F1')
    plt.grid(axis='x')
    plt.show()










    # array_DATA = np.array(PokemoData_df)
    # list_DATA = array_DATA.tolist()
    # bestfeature, A = Tree.chooseBestFeatureToSplit(list_DATA)
    # data = pd.DataFrame(A)
    # data = data.iloc[:, 1]
    # data = np.array(data)
    # list_data = data.tolist()

    # plt.plot(list_data)
    # plt.xlabel('label')
    # plt.ylabel('信息增益')
    # plt.grid(axis='x')
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
