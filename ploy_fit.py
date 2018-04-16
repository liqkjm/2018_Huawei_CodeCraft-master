# coding=utf-8

"""
多项式曲线拟合算法
"""
import matplotlib.pyplot as plt
# import numpy as np
import random
import math

'阶数为9阶 n=9=k'
order = 5


def solve(Vx, Vy, ex, Em, coefficient):
    # Em = []
    # 首先计算增广矩阵
    for i in range(ex + 1):
        Em.append([])
        for j in range(len(Vx)):
            Em[i].append(Vx[i][j])
        Em[i].append(Vy[i])

    # 高斯消元求系数
    gauss(Em, coefficient)


def rand(a, b):
    return (b - a) * random.random() + a


def swap(a, b):
    t = a
    a = b
    b = t


def gauss(ma, coefficient):
    m = ma  # 这里主要是方便最后矩阵的显示
    row_pos = 0
    col_pos = 0
    ik = 0
    n = len(m)

    # row_pos 变量标记行循环, col_pos 变量标记列循环
    # print_matrix("一开始 de 矩阵", m)

    while (row_pos < n) and (col_pos < n):
        # print "位置：row_pos = %d, col_pos = %d" % (row_pos, col_pos)
        # 选主元
        mik = - 1
        for i in range(row_pos, n):
            if abs(m[i][col_pos]) > mik:
                mik = abs(m[i][col_pos])
                ik = i

        if mik == 0.0:
            col_pos = col_pos + 1
            continue

        # 交换两行
        if ik != row_pos:
            for j in range(col_pos, n):
                swap(m[row_pos][j], m[ik][j])
                swap(m[row_pos][n], m[ik][n])  # 区域之外？

        try:
            # 消元
            m[row_pos][n] /= m[row_pos][col_pos]
        except ZeroDivisionError:
            # 除零异常 一般在无解或无穷多解的情况下出现……
            return 0

        j = n - 1
        while j >= col_pos:
            m[row_pos][j] /= m[row_pos][col_pos]
            j = j - 1

        for i in range(0, n):
            if i == row_pos:
                continue
            m[i][n] -= m[row_pos][n] * m[i][col_pos]

            j = n - 1
            while j >= col_pos:
                m[i][j] -= m[row_pos][j] * m[i][col_pos]
                j = j - 1

        row_pos = row_pos + 1
        col_pos = col_pos + 1

        for i in range(0, len(m)):
            coefficient[i] = m[i][len(m)]


def predict(pre_x, coefficient):
    ya = []
    for i in range(0, len(pre_x)):
        yy = 0.0
        for j in range(0, order + 1):
            dy = 1.0
            for k in range(0, j):
                dy *= pre_x[i]  # x^k
            # ak*x^k
            dy *= coefficient[j]  # matAA[j]即为系数a, dy = a * x ^ k
            yy += dy    # yy = a0 + a1 * x ^1 + a2 * x ^ 2 + ... + an * x ^ n
        ya.append(yy)
    return ya


def ploy_fit(x, y1, pre_x):
    # 生成样例曲线上的各个点 100个, x的个数决定了样本量

    Em = []
    coefficient = [0.0] * (order + 1)  # 系数 a
    nd = []

    # data = x, y
    for i in range(len(x)):
        nd.append([x[i], y1[i]])

    # 进行曲线拟合 AX = Y
    matA = []  # 多项式矩阵 X
    for i in range(0, order + 1):
        matA1 = []  # 每一行
        for j in range(0, order + 1):
            tx = 0.0  # 每一列
            for k in range(0, len(x)):
                dx = 1.0  # 表示初始
                for l in range(0, j + i):
                    dx = dx * x[k]  # x^k
                tx += dx
                # 运行n次(len(xa)次)后，tx为sum(x[i]^2k) 或 n (dx=1运行n次)
            matA1.append(tx)
        matA.append(matA1)

    matB = []  # Y
    for k in range(0, order + 1):
        ty = 0.0
        # 加和n个
        for i in range(0, len(x)):
            dy = 1.0
            # 对于从i=1->n 求 (x[i])^(k-1)
            for l in range(0, k):
                dy = dy * x[i]  # dy即为公式中 (x[i])^k
            ty += y1[i] * dy  # 先乘完再加和
        matB.append(ty)

    # 求 A
    solve(matA, matB, order, Em, coefficient)
   # predict y
    ya = predict(x, coefficient)
    y_pred = predict(pre_x, coefficient)
    # plt.plot(x, y, '*')
    plt.plot(x, ya, 'g')
    plt.plot(x, y1, "b")
    plt.plot(pre_x, y_pred, color='r')
    plt.show()
    return y_pred


def predict_vm(x, y_list, pre_x):

    num_dict = []
    for v in range(5):
        y = []
        for i in range(len(y_list)):
            y.append(y_list[i][v])
        temp = 0.0
        y_pred = ploy_fit(x, y, pre_x)
        for i in range(len(y_pred)):
            if y_pred[i] < 0:
                y_pred[i] = 0
            temp += y_pred[i]
        num_dict.append(int(round(temp)))

    return num_dict


if __name__ == '__main__':

    x = [1, 2, 3, 4, 5, 6, 7]
    y = [9, 1, 8, 2, 7, 3, 6]
    ya = ploy_fit(x, y, x)
    print "ya = ", ya

