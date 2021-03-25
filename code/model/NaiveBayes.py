import files
import math
import numpy as np

# 加载训练集，格式为[array([[]])]
def loadtrain():
    data = files.readpkl("pro_data/train_stat.pkl")

    for i in range(len(data)):
        if i % 500 == 0:
            print("i:", i)
        user_id = np.array([data[i][0]] * data[i][1]).reshape(-1, 1)
        item_id = np.array(list(data[i][2].keys()), dtype=int).reshape(-1, 1)
        score = np.array(list(data[i][2].values()), dtype=int).reshape(-1, 1)
        if i == 0:
            matrix = np.concatenate((user_id, item_id, score), axis=1)
        else:
            m = np.concatenate((user_id, item_id, score), axis=1)
            matrix = np.vstack((matrix, m))

    _score = matrix[:, 2]
    train = []
    for i in range(101):
        index = np.argwhere(_score == i).reshape(-1)
        train.append(matrix[index])

    #files.writetxt('pro_data/train.txt', train)
    files.writepkl('pro_data/train.pkl', train)

# 将训练集中X与Y区分开
def loadtrain_xy():
    #loadtrain()
    train = files.readpkl("pro_data/train.pkl")
    train_X, train_Y = [], []
    for i in range(len(train)):
        train_X.append(train[i][:, :-1])
        train_Y.append(train[i][:, -1])
    files.writepkl('pro_data/train_X.pkl', train_X)
    #files.writetxt('pro_data/train_X.txt', train_X)
    files.writepkl('pro_data/train_Y.pkl', train_Y)

# 加载测试集，保存为二维矩阵
def loadtest():
    data = files.readpkl("pro_data/test_stat.pkl")

    for i in range(len(data)):
        if i % 500 == 0:
            print("i:", i)
        user_id = np.array([data[i][0]] * data[i][1]).reshape(-1, 1)
        item_id = np.array(list(data[i][2].keys()), dtype=int).reshape(-1, 1)
        if i == 0:
            matrix = np.concatenate((user_id, item_id), axis=1)
        else:
            m = np.concatenate((user_id, item_id), axis=1)
            matrix = np.vstack((matrix, m))

    #files.writetxt('pro_data/test.txt', matrix)
    files.writepkl('pro_data/test.pkl', matrix)

# prior_pr,样本为类别c的先验概率，格式为一维数组，train为[array([[]])], test为array([])
# 朴素贝叶斯分类器，对测试集进行结果分类
def NaiveBayes_Classifier(ave, std, test, prior_pr, labels=101):
    score = []
    for i in range(labels):
        comp_1 = np.log((2 * math.pi) ** 0.5 * std[i])  # 取自然对数减小计算成本
        comp_2 = np.power(test - ave[i], 2) / (2 * np.power(std[i], 2))
        guassian_score = -1 * np.sum(comp_1 + comp_2) + math.log(prior_pr[i])
        score.append(guassian_score)
    
    return score.index(max(score))

# 对测试集进行打分
def get_scores():
    train_X = files.readpkl("pro_data/train_X.pkl")
    each_label_num = np.array([len(data) for data in train_X])
    prior_pr = each_label_num / np.sum(each_label_num)  # 先验概率

    ave = np.array([np.mean(train_X[i], axis=0) for i in range(len(train_X))])    # 均值
    std = np.array([np.std(train_X[i], axis=0) for i in range(len(train_X))])     # 标准差

    test = files.readpkl("pro_data/test.pkl")   # 二维矩阵
    scores = []
    for t in test:
        scores.append(NaiveBayes_Classifier(ave, std, t, prior_pr))
    scores = np.array(scores)

    files.writepkl("scores.pkl", scores)

# 为计算RMSE，按照7：3的比例划分出的训练集与测试集
def load_rmsedata():
    train = files.readpkl("pro_data/train_X.pkl")
    each_label_num = np.array([len(data) for data in train])
    train_count = each_label_num * 0.7
    train_count = train_count.astype(int)   # 训练集每层总量
    
    rmse_train, rmse_test = [], []
    for i in range(101):
        rmse_train.append(train[i][:train_count[i]])
        rmse_test.append(train[i][train_count[i]:])
    
    files.writepkl("rmse_data/rmse_train.pkl", rmse_train)
    files.writepkl("rmse_data/rmse_test.pkl", rmse_test)
    #files.writetxt("rmse_data/rmse_test.txt", rmse_test)

# 计算rmse
def get_rmse():
    train = files.readpkl("rmse_data/rmse_train.pkl")
    each_label_num = np.array([len(data) for data in train])
    prior_pr = each_label_num / np.sum(each_label_num)  # 先验概率

    ave = np.array([np.mean(train[i], axis=0) for i in range(len(train))])    # 均值
    std = np.array([np.std(train[i], axis=0) for i in range(len(train))])     # 标准差
    test = files.readpkl("rmse_data/rmse_test.pkl")

    y_true, y_hat = [], []
    for i in range(len(test)):
        print("i: ", i, "/", len(test))
        for t in test[i]:
            y_true.append(i)
            y_hat.append(NaiveBayes_Classifier(ave, std, t, prior_pr))
    
    y_true, y_hat = np.array(y_true), np.array(y_hat)
    rmse = np.sqrt(1 / len(y_true) * np.sum((y_true - y_hat) ** 2))
    print("RMSE计算结果为: %f" % rmse)


if __name__ == "__main__":
    #loadtest()
    #loadtrain_xy()
    #load_rmsedata()
    get_rmse()
    get_scores()
