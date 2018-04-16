# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """

    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example_0():

    np.random.seed(0)
    mem_cell_ct = 100

    x_dim = 1
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    """
    y_list = [-0.5, 0.2, 0.1, -0.5]  # 五步
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for i in range(len(input_val_arr)):
        print "input_val_arr = ", input_val_arr[i]
    """
    input_val_arr = [
        [1, 2],
        [2, 3],
        [3, 4]
    ]

    pre_x = [
        [3, 4]
    ]

    y_list = [0.03, 0.05, 0.07]
    x = np.arange(-1, 1, 0.01)
    xa = []
    for i in range(len(x)):
        xa.append([x[i]])
    # y = 2 * np.sin(x * 2.3) + 0.5 * x ** 3
    # y1 = y + 0.5 * (np.random.rand(len(x)) - 0.5)
    y = ((x * x - 1) ** 3 + 1) * (np.cos(x * 2) + 0.6 * np.sin(x * 1.3))
    y_list = y + (np.random.rand(len(x)) - 0.5)
    # print "input_val_arr = ", input_val_arr

    for cur_iter in range(len(x)):
        for ind in range(len(y_list)):
            lstm_net.x_list_add(xa[ind])

        y_pred = [0] * len(y_list)

        for i in range(len(y_list)):
            y_pred[i] = lstm_net.lstm_node_list[i].state.h[0]

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        # print("loss:", "%.3e" % loss)

        lstm_param.apply_diff(lr=0.1)  # 更新权重
        lstm_net.x_list_clear()

    plt.plot(xa, y_list)
    plt.plot(xa, y_pred)
    plt.show()
    print "loss = ", loss
    print "y_pred = ", y_pred

def test(data_x, day_flaovr_num):
    np.random.seed(0)
    mem_cell_ct = 100
    x_dim = 7  # 7天一个维度

    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    """
    input = []
    y_list = []
    # 处理数,将数据处理成  7天一段，逐天滚动
    for i in range(len(data_x) - 7):
        input.append(data_x[i: i+7])
        # y_list.append(day_flaovr_num[i + 7][0])

    for i in range(len(data_x) - 7):
        y_list.append(day_flaovr_num[i + 7])

    # y_list = [-0.5, 0.2, 0.1, -0.5]  # 五步
    # input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for i in range(len(input)):
        print "input_val_arr = ", input[i]

    for i in range(len(y_list)):
        print "y_list =", y_list[i]

    print "len(input) = ", len(input)
    print "len(y_list) = ", len(y_list)
    
"""

    input_val_arr = [
        [1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 8]
        [3, 4, 5, 6, 7, 8, 9],
    ]
    y_list = [1, 2, 3]
    for cur_iter in range(100):
        # print("iter", "%2s" % str(cur_iter), end=": ")
        print "str(cur_iter) = ", str(cur_iter)
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        # print("y_pred = [" +
        #       ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
        #       "]", end=", ")
        for i in range(len(y_list)):
            print "y_pred = ", lstm_net.lstm_node_list[i].state.h[0]

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        # print("loss:", "%.3e" % loss)
        print "loss = ", loss
        lstm_param.apply_diff(lr=0.1)  # 更新权重
        lstm_net.x_list_clear()


if __name__ == "__main__":
    example_0()
