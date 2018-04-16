# coding=utf-8
import math
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    # type: (object, object, object) -> object
    """

    :type fill: object
    """
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh_derivative(values):
    return 1. - values ** 2


# 定义BPNeuralNetwork类， 使用三个列表维护输入层，隐含层和输出层神经元， 列表中的元素代表对应神经元当前的输出值.
# 使用两个二维列表以邻接矩阵的形式维护输入层与隐含层， 隐含层与输出层之间的连接权值， 通过同样的形式保存矫正矩阵.
class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0  # 每层神经元的个数
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []  # 三个列表分别表示输入层，隐含层，输出层 当前的输出值
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []  # 两个权值矩阵 ij,jk
        self.output_weights = []
        self.input_correction = []  # 两个矫枉矩阵
        self.output_correction = []

    def setup(self, ni, nh, no):  # 定义setup方法初始化神经网络:
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-2.0, 2.0)
                # self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):  # 一次反馈，计算隐含层和输出层的输出，并返回输出层的输出
        # activate input layer
        # print "type(input_cells) = ", type(self.input_cells[0])
        # print "type(inputs) = ", type(inputs[0])
        # print "inputs = ", inputs
        # print "inputs[0] = ", inputs[0]

        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
            # self.input_cells[i] = inputs  # 更改
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            # self.hidden_cells[j] = sigmoid(total)
            # self.hidden_cells[j] = math.tanh(total)
            self.hidden_cells[j] = 0.01 * total

        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            # self.output_cells[k] = sigmoid(total)
            self.output_cells[k] = 0.01 * total
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):  # 一次反向传播， 更新权值的全部过程， 并返回误差
        # feed forward       #case ：输入数据
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            # output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
            output_deltas[o] = 0.01 * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            # hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
            # hidden_deltas[h] = tanh_derivative(self.hidden_cells[h]) * error
            hidden_deltas[h] = 0.01 * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * abs(label[o] - self.output_cells[o])
        return error

    def train(self, cases, labels, limit=2000, learn=0.05, correct=0.1):  # 控制迭代
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
        print "error = ", error

    def bp_predict(self, data_x, data_y, pre_x):  # 演示

        # 处理数据
        x, y = [], []
        for i in range(len(data_x)):
            x.append([data_x[i]])

        predict_x = []
        for i in range(len(pre_x)):
            predict_x.append([pre_x[i]])

        pre_y = []
        self.setup(1, 5, len(data_y[0]))
        self.train(x, data_y, 2000, 0.001, 0.1)
        for i in range(len(predict_x)):
            pre_y.append(self.predict(predict_x[i]))

        print "pre_y = ", pre_y
        num_predict = [0] * len(data_y[0])

        for i in range(len(pre_y)):
            for j in range(len(pre_y[0])):
                num_predict[j] += pre_y[i][j]

        for i in range(len(num_predict)):
            num_predict[i] = int(round(num_predict[i]))
            if num_predict[i] < 0:
                num_predict[i] = 0

        # print "self.input_weights = ", self.input_weights
        # print "self.output_weights = ", self.output_weights
        return num_predict


if __name__ == '__main__':
    data_x = [1, 2, 3, 4]
    data_y = [[2], [4], [6], [8]]
    pre_x = [7, 8, 9, 10]
    # plt.plot()
    nn = BPNeuralNetwork()
    nn.bp_predict(data_x, data_y, pre_x)
