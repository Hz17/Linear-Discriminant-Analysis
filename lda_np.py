import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    #读取数据
    df_wine = ps.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_test)
    print(y_test)
    #标准化
    sc =StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    #计算平均向量
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    #    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))
    #计算类内散布矩阵
    d = 13  # number of features 13*13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))  # scatter matrix for each class
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
            class_scatter += (row - mv).dot((row - mv).T)
        S_W += class_scatter  # sum class scatter matrices
    #print(S_W)
    #计算散布矩阵
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13  # number of features
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)  # make column vector
        mean_overall = mean_overall.reshape(d, 1)  # make column vector
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    #print(S_B)
    #求解Sw-1Sb的广义特征值
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    #构建（eigen_vals, eigen_vecs）元组
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    #将该元组从大到小排序
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    #print(eigen_pairs)
    #选取eigen——pairs的前两个（判别能力最强）特征向量构建转换矩阵W
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real, eigen_pairs[2][1][:, np.newaxis].real, eigen_pairs[3][1][:, np.newaxis].real))
    print(w)
    #画结果图
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train == l, 0] * (-1), X_train_lda[y_train == l, 1] * (-1), c=c, label=l, marker=m)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()