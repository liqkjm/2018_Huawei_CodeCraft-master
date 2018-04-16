# -*- coding:UTF-8 -*-
"""
import sys
import time
import string
import math

"""
from BP import BPNeuralNetwork
import random
import ploy_fit

random.seed(0)


def large_rand(num_dict):
    for i in range(len(num_dict)):
        num_dict[i] += rand(-3, 5)
        num_dict[i] = int(round(num_dict[i]))
        if num_dict[i] < 0:
            num_dict[i] = 0
    print "num_dict = ", num_dict
    return num_dict


vm_flavorName = ['flavor1', 'flavor2', 'flavor3', 'flavor4', 'flavor5', 'flavor6', 'flavor7', 'flavor8', 'flavor9',
                 'flavor10', 'flavor11', 'flavor12', 'flavor13', 'flavor14', 'flavor15']  # 名称
vm_flavorCpu = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16]  # CPU
vm_flavorMem = [1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64]  # 内存
recordFlavor = []
flavorPredict = []  # 初始化预测空列表
numOfPredict = []
pm_flavor = []
pm_restCpu = []  # 当前剩余资源
pm_restMem = []


def rand(a, b):
    return (b - a) * random.random() + a


# 对日期进行解码,如果专业一点,按月份来算天数,每次要多个循环。
def decode(date_str):  # 对单独日期进行解码

    date = date_str.split(" ")[0].split("-")

    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    date_int = (month - 1) * 30 + day
    """
    total = 0
    m_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        m_month[1] = 29
    for i in range(month - 1):
        total += m_month[i]
    total += day
    return total
    """
    return date_int


# 对所有的日期进行解码
def Decode_Date(date_list):
    vm_date_int = []
    for i in range(len(date_list)):
        vm_date_int.append(decode(str(date_list[i])))

    # print "vm_date-int = ", vm_date_int
    return vm_date_int


# 读取参数,包括物理服务器的信息,以及虚拟机的规格信息和预测起始时间等
def Read_Par(lines):
    # type: (object) -> object
    global pm_cpu
    global pm_mem
    global kind
    global vm_cpu
    global vm_mem
    global vm_flavor

    pm = lines[0].split(" ")
    pm_cpu = int(pm[0])
    pm_mem = int(pm[1])
    pm_hd = int(pm[2])

    vm_num = int(lines[2])

    vm_flavor = []
    vm_cpu = []
    vm_mem = []
    for i in range(vm_num):
        line = lines[3 + i].split(" ")
        vm_flavor.append(line[0])
        vm_cpu.append(int(line[1]))
        vm_mem.append(int(line[2]))

    kind = lines[vm_num + 4]
    begin = decode(lines[vm_num + 6])
    end = decode(lines[vm_num + 7])
    predict_day = end - begin + 1  # 预测的总天数
    pre_x = []
    for i in range(predict_day):
        pre_x.append(begin + i)
    return pm_cpu, pm_mem, pm_hd, vm_num, vm_flavor, vm_cpu, vm_mem, kind, begin, end, predict_day, pre_x


# 读取测试数据
def Read_Data(lines):
    vm_id = []
    vm_f = []
    vm_date = []

    for i in range(len(lines)):
        line = lines[i].split("\t")
        vm_id.append(line[0])
        vm_f.append(line[1])
        vm_date.append(line[2])

    vm_date_int = Decode_Date(vm_date)
    return vm_id, vm_f, vm_date_int


# 前后几天求和
def sum_begin_to_end(day_flavor_num, sub, begin, end):
    total = []
    for i in range(len(day_flavor_num[0])):
        temp = 0
        for j in range(sub - begin, sub + end, 1):
            temp += day_flavor_num[j][i]
        total.append(temp)
    return total


# 判断是否为噪点
def is_noise(date, sub, day_flavor_num):
    """
    判断兼降噪函数,

    """
    # total = []
    # 求前后两周天的总数
    if sub > len(day_flavor_num) - 7:  # 首尾没有前/后
        total = sum_begin_to_end(day_flavor_num, sub, 7, 0)
    elif sub < 7:
        total = sum_begin_to_end(day_flavor_num, sub, 0, 7)
    else:
        total = sum_begin_to_end(day_flavor_num, sub, 7, 7)

    for i in range(len(day_flavor_num[0])):
        if day_flavor_num[sub][i] > total[i] / 14.0 * 3:  # 如果当天大于前后14天的1/2
            day_flavor_num[sub][i] = total[i] / 2.0  # 就把它等于2倍的平均数?

    return day_flavor_num[sub]
    # print " 降噪之后的输出: ", day_flavor_num[date]

    # noise = [360, 1, 119, 120, 121, 271, 272, 273, 274, 275, 276, 277, 311]  # 不行

    '''
    全国节假日安排:
    元旦:12.30 - 1.1
    春节
    劳动节:4.29 - 5.1
    国庆节:10.1 - 10.7
    双十一:11.11 !
    '''


# 以天为周期统计,其实都一样,不存在什么影响。
def count_flavor(vm_f1, vm_flavor1, vm_date_int):
    end = vm_date_int[len(vm_date_int) - 1]
    begin = vm_date_int[0]
    day_max = end - begin + 1

    day = 0  # 统计一共有多少天, 记住最后天数 + 1(包括 0 )
    day_flavor_num = []  # 二维列表, 统计每一天每规格的个数
    day_flavor = []  # 记录相同日期内的所有规格
    data_x = []

    for ii in range(day_max):
        data_x.append(begin + ii)
        day_flavor_num.append([])

    # print "data_x = ", data_x

    vm_date_int.append(0)  # 添加一个标记点,用于接下来的判断

    # print "vmn_flavor1 = ", vm_flavor1
    for r in range(len(vm_date_int) - 1):  # 遍历所有日期,把相同日期的规格先放入一个列表,
        # 再使用list的count函数,统计每个规格的个数
        if vm_date_int[r] != vm_date_int[r + 1]:  # 不同日期,天数加一
            day_flavor.append(vm_f1[r])
            for i in range(len(vm_flavor1)):  # 对列表统计要求的规格的个数
                temp = day_flavor.count(vm_flavor1[i])  # 该列表该规格的个数
                day_flavor_num[day].append(temp)  # 二维列表, 一维为天数, 二维为规格, 其值为 该天该规格的个数
            day = day + 1
            day_flavor = []  # 重新赋值为空集,开始记录下一天
        else:  # 与后面一条日期相同,对其进行统计
            day_flavor.append(vm_f1[r])
        # print "the day is not end and the day_flavor is : ", day_flavor
    # print "day_flavor_num = ", day_flavor_num
    # print "len(day_flavor_num) = ", len(day_flavor_num)
    print "day = ", day
    return day_flavor_num, day


# 以天为周期,补上了数据上没有的天数,连2月30号也给补了大概。
def count_day(vm_f, vm_flavor, date):
    end_time = date[len(date) - 1]
    begin_time = date[0]
    day_max = end_time - begin_time + 1  # 总共的天数
    day_flavor_num = []  # 二维列表, 统计每一天每种规格的个数
    day_flavor = []  # 记录相同日期内的所有规格
    data_x = []  # 即 0 -> day_max,!!!

    for i in range(day_max):  # 2018年4月5日 16:34:00 day_max 不用减一
        data_x.append(begin_time + i)  # 没毛病,即使专业点的解码,依旧是这样。只不过总天数会减少。
        day_flavor.append([])
        day_flavor_num.append([])

    # print "data_x = ", data_x
    for i in range(day_max):
        for j in range(len(date) - 1):
            if date[j] == data_x[i]:
                day_flavor[i].append(vm_f[j])

    for i in range(day_max):
        for j in vm_flavor:
            temp = day_flavor[i].count(j)
            day_flavor_num[i].append(temp)

    # print "降噪之前的day_flavor_num = ", day_flavor_num
    # plt.plot(data_x, day_flavor_num)
    # plt.show()

    for i in range(len(day_flavor_num)):
        # for j in range(len(day_flavor_num)):
        day_flavor_num[i] = is_noise(data_x[i], i, day_flavor_num)

    # print "降噪之后的day_flavor_num = ", day_flavor_num
    # plt.plot(data_x, day_flavor_num)
    # plt.show()
    return day_flavor_num, day_max, data_x


# 以周为周期
def count_week(vm_f, vm_flavor, date):
    day_flavor_num, day_max, data_x = count_day(vm_f, vm_flavor, date)
    week_max = day_max / 7  # 此处修改:去掉加一,相当于省略了不满一星期的末尾天数
    week_flavor_num = []  # 二维列表, 统计每周每规格的个数
    week_x = []  # 横坐标,所以,由第一天开始,七天为一个周期,(即使某一天当中,没有申请,也应该把该天列入)。便为其编号为每周第一天的日期除以 7(取整)
    week_day = []  # 列表元素为七天的日期
    for ii in range(week_max):  # 暂时假设跨度最大天数为100天, 初始化二维列表
        week_flavor_num.append([])
        week_day.append([])

    for i in range(week_max):
        week_x.append(data_x[0] / 7 + 7 * i)
        for j in range(7):
            week_day[i].append(data_x[0] + j + i * 7)

    count = 0
    for j in range(len(vm_flavor)):
        week = 0
        for i in range(day_max):
            count += day_flavor_num[i][j]
            if (i + 1) % 7 == 0:
                week_flavor_num[week].append(count)
                week += 1
                count = 0
    # print "week_x = ", week_x
    # print "week_day", week_day
    # print "week_flavor_num = ", week_flavor_num
    return week_max, week_x, week_day, week_flavor_num


# 预测方法1:求平均数
def average(vm_flavor, week, week_flavor_num, week_predict):  # 预测
    num_predict = []
    for ii in range(len(vm_flavor)):
        num_predict.append(0)

    '''
    此处有两种用法:(懒得解决的bug)
    针对于理论上 day = end - begin
    但其实可能没有那么多天的情况(即某些天申请的数量为零,故数据上没有,则没有统计)
    所以:
    - 1. day = len(day_flavor_num)
    - 2. day < len(day_flavor_num), 可能造成越界
    '''

    for j in range(len(week_flavor_num)):
        for k in range(len(vm_flavor)):
            num_predict[k] += int(week_flavor_num[j][k])

    for k in range(len(vm_flavor)):
        num_predict[k] = int(round(float(num_predict[k]) / len(week_flavor_num) * week_predict))

    return num_predict


# 预测方法2:最小二乘法 least square method
def compute_mb(data_x, data_y, sub):  # 计算 m 和 b
    #  横坐标,纵坐标,纵坐标下标,针对不同的规格有不同的 m b
    n = len(data_x)
    xx_sum = 0.0  # x的平方和
    x_sum = 0.0  # x求和
    y_sum = 0.0  # y求和
    xy_sum = 0.0  # xy的乘积求和

    for i in range(n):
        xx_sum += data_x[i] ** 2
        y_sum += data_y[i][sub]
        x_sum += data_x[i]
        xy_sum += data_x[i] * data_y[i][sub]

    b = (xx_sum * y_sum - x_sum * xy_sum) / (n * xx_sum - x_sum * x_sum)
    m = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum ** 2)

    return m, b


def predict_lsq(predict_x, m, b):  # 预测某种规格,在所给时间段的总申请数量,即 # 预测 y = m * x + b,
    # 本次测试用例为一周,predict_x = 1
    predict_y = 0.0
    for i in range(len(predict_x)):
        predict_y += m * predict_x[i] + b
    # print "predict_y = ", predict_y
    return predict_y


def least_sq(data_x, data_y, pre_x, vm_num):  # 预测所有规格(numpy库有leastsq函数,还得各种手撸)
    # 参数定义:数据横坐标,纵坐标,预测的横坐标,预测的规格种类数量
    predict_num = []  # 列表存储所有规格的预测数量
    for i in range(vm_num):
        m, b = compute_mb(data_x, data_y, i)
        pre_y = int(round(predict_lsq(pre_x, m, b)))
        if pre_y < 0:
            pre_y = 0
        predict_num.append(pre_y)
    # print "predict_num = ", predict_num
    return predict_num


# 预测方法3:线性回归-低度下降法(未测试)
def compute_error(b, m, data):  # 计算误差
    totalError = 0

    for i in range(len(data) - 1):
        x = data[i][0]
        y = data[i][1]
        totalError += (y - (m * x + b)) * (y - (m * x + b))

    return totalError / float(len(data))


def optimizer(data, starting_b, starting_m, learning_rate, num_iter):
    b = starting_b
    m = starting_m

    for i in range(num_iter):
        b, m = compute_gradient(b, m, data, learning_rate)
    return [b, m]


def compute_gradient(b_current, m_current, data, learning_rate):  # 梯度下降,更新m,b
    b_gradient = 0
    m_gradient = 0

    N = float(len(data) - 1)
    for i in range(len(data) - 1):
        x = data[i][0]
        y = data[i][1]
        # computing partial derivations of our error function
        # b_gradient = -(2 / N) * sum((y - (m_current * x + b_current)) ^ 2)
        # m_gradient = -(2 / N) * sum(x * (y - (m_current * x + b_current)) ^ 2)
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def Linear_regression(data_x, data_y, pre_x):  # data 传入
    """
    处理数据,使之符合函数的要求
    """
    num_predict = [0] * len(data_y[0])
    # print "num_predict = ", num_predict
    for ii in range(len(data_y[0])):
        data = []
        for i in range(len(data_x)):
            data.append([])
        for i in range(len(data_x)):
            data[i].append(data_x[i])
            data[i].append(data_y[i][ii])
        # print "data = ", data
        # 常用的学习率数值:0.01,0.03,0.1,0.3,1,3,10
        learning_rate = 0.000001
        initial_b = 0.001
        initial_m = 0.001
        num_iter = 1000

        [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)
        predict_y = predict_lsq(pre_x, m, b)
        num_predict[ii] = int(round(predict_y))
    return num_predict


# ****************************************************************
# 资源分配
def arrange(flavorPredict, numOfPredict):
    global pm_restCpu
    global pm_restMem
    global recordFlavor
    global pm_cpu
    global pm_mem
    global vm_flavorName
    if len(pm_restCpu) == 0:  # 第一台pm赋值
        pm_restCpu.append(pm_cpu)
        pm_restMem.append(pm_mem)
        recordFlavor.append([]);  # 在第一台pm记录中加入0(用来占位,无实际意义)
        index1 = 0

    for i in range(len(flavorPredict)):  # 对已预测虚拟机分配
        index2 = vm_flavorName.index(flavorPredict[i])  # 确定需要分配的vm类型,取下标
        temp1 = numOfPredict[i]  # 对应虚拟机数量
        while temp1 > 0:  # 预测的某种vm数量是0没关系,无法进入循环
            # print temp1,"++++++++++++++++"
            for index1 in range(len(pm_restCpu)):  # 选择哪台pm
                # print index1
                if (pm_restCpu[index1] < vm_flavorCpu[index2] or pm_restMem[index1] < vm_flavorMem[
                    index2]) and index1 + 1 == len(pm_restCpu):  # 若放不下且是最后一个物理机
                    recordFlavor.append([])  # 重新分一台pm
                    pm_restCpu.append(pm_cpu)  # 加物理机
                    pm_restMem.append(pm_mem)
                    index1 += 1  # 物理机数量加 1
                elif (pm_restCpu[index1] < vm_flavorCpu[index2] or pm_restMem[index1] < vm_flavorMem[
                    index2]) and index1 + 1 < len(pm_restCpu):  # 当前装不下,但并不是以后的也装不下
                    continue

                pm_restCpu[index1] -= vm_flavorCpu[index2]  #
                pm_restMem[index1] -= vm_flavorMem[index2]  # 更新剩余men
                recordFlavor[index1].append(vm_flavorName[index2])  # 记录某台pm已分配的虚拟机(数量之后统计)
                temp1 -= 1
                # print index1
                # print recordFlavor,"******"
                break  # 进行下一个vm的分配


# 首先是判断下,根据输入文件的kind字段,如果是cpu就针对cpu作优化,如果是内存,就针对内存优化
# 具体策略是:比如kind=cpu,就把预测的的虚拟机和对应的所需个数排序,具体是按照虚拟机的cpu要求来对predictflavor排序,最终目的
# 是得到一个sortedFlavorType列表,这是把预测的所需虚拟机规格按照cpu要求从大到小排列,sortedNumOfPredict是sortedFlavorType列表对应的个数
# 现在只是把虚拟机的被分配顺序改动了,物理资源的分配策略那里没有改动
def arrange_bestfit(flavorPredict, numOfPredict):
    print 'bestfit'
    global pm_restCpu
    global pm_restMem
    global recordFlavor
    global vm_cpu
    global vm_mem
    global vm_flavor
    global kind

    if len(pm_restCpu) == 0:  # 第一台pm赋值
        pm_restCpu.append(pm_cpu)
        pm_restMem.append(pm_mem)
        recordFlavor.append([])  # 在第一台pm记录中加入0(用来占位,无实际意义)
        index1 = 0

    sortedFlavorType = []  # 对预测的虚拟机按照kind类型进行排序,是一个嵌套列表
    FlavorType = []  # 对预测的虚拟机的对应虚拟机规格的kind需求量(未排序)

    # print '(' + kind + ')'
    kind = kind.strip()

    # print '(' + kind + ')'

    if kind == 'MEM':  # 如果是内存,就按照虚拟机的内存需求从小到大来对预测出来的虚拟机规格列表排序
        # print 'kind = 内存'
        for i in flavorPredict:
            index2 = vm_flavor.index(i)
            FlavorType.append(vm_mem[index2])
        sortFlavorType = zip(flavorPredict, FlavorType)
        sortedFlavorType = sorted(sortFlavorType, key=lambda x: x[1], reverse=True)
    else:  # 如果不是内存,就按照虚拟机的cpu需求从小到大来对预测出来的虚拟机规格列表排序
        # print 'kind = CPU'
        for i in flavorPredict:
            index2 = vm_flavor.index(i)
            FlavorType.append(vm_cpu[index2])
        sortFlavorType = zip(flavorPredict, FlavorType)
        # for i in sortFlavorType:
            # print i
        # print "sortFlavorType"
        # print sortFlavorType
        sortedFlavorType = sorted(sortFlavorType, key=lambda x: x[1], reverse=True)

    # print 'this'
    # print sortedFlavorType

    sortedFlavorPredict = []
    sortedNumOfPredict = []
    for i in range(len(sortedFlavorType)):
        # print i
        temp = sortedFlavorType[i][0]
        # print temp
        sortedFlavorPredict.append(temp)
        # print flavorPredict.index(temp)
        tempindex = flavorPredict.index(temp)
        sortedNumOfPredict.append(numOfPredict[tempindex])
    # print sortedFlavorPredict  # predict demand type(sorted)
    # print sortedNumOfPredict  # predict demand count(sorted)

    totalFlavor = []
    restFlavor = 0
    totalFlavorList = []
    for i in range(len(sortedFlavorPredict)):
        # print 'sortedFlavorPredict[i]'
        # print sortedFlavorPredict[i]
        if (sortedNumOfPredict[i] != 0):
            templist = []
            for j in range(sortedNumOfPredict[i]):
                totalFlavor.append(sortedFlavorPredict[i])
                restFlavor += 1
                templist.append(sortedFlavorPredict[i])
            totalFlavorList.append(templist)

    # print totalFlavor
    # print restFlavor
    # print totalFlavorList

    while len(totalFlavorList) > 0:  # 对所有要放置的虚拟机列表作判断,如果列表不为空,就继续
        # print totalFlavor[0][0]
        # print vm_flavorName.index(totalFlavorList[0][0])
        index2 = vm_flavorName.index(totalFlavorList[0][0])  # 准备放置虚拟机列表的第一个元素
        # print index2
        for index1 in range(len(pm_restCpu)):  # 对物理机做遍历
            if (pm_restCpu[index1] >= vm_flavorCpu[index2]) and (pm_restMem[index1] >= vm_flavorMem[index2]):
                pm_restCpu[index1] -= vm_flavorCpu[index2]
                pm_restMem[index1] -= vm_flavorMem[index2]
                recordFlavor[index1].append(vm_flavorName[index2])
                del (totalFlavor[0])
                del (totalFlavorList[0][0])
                if (len(totalFlavorList[0]) == 0):
                    del (totalFlavorList[0])
                if (len(totalFlavorList) == 0):
                    return
                row = len(totalFlavorList)
                col = len(totalFlavorList[row - 1])
                minFlavor = totalFlavorList[row - 1][col - 1]
                minindex = vm_flavorName.index(minFlavor)
                # print minFlavor
                # print minindex
                while len(totalFlavorList) > 0 and (pm_restCpu[index1] > vm_flavorCpu[minindex]) and (
                        pm_restMem[index1] > vm_flavorMem[minindex]):
                    # while (pm_restCpu > 0) and (pm_restMem > 0):
                    for i in range(len(totalFlavorList)):
                        index3 = vm_flavorName.index(totalFlavorList[i][0])
                        # print index3
                        if (pm_restCpu[index1] >= vm_flavorCpu[index3]) and (
                                pm_restMem[index1] >= vm_flavorMem[index3]):
                            pm_restCpu[index1] -= vm_flavorCpu[index3]
                            pm_restMem[index1] -= vm_flavorMem[index3]
                            recordFlavor[index1].append(vm_flavorName[index3])
                            del (totalFlavorList[i][0])
                            if (len(totalFlavorList[i]) == 0):
                                del (totalFlavorList[i])
                            if (len(totalFlavorList) == 0):
                                return
                            break
                    row = len(totalFlavorList)
                    # print row
                    # print len(totalFlavorList[row - 1])
                    col = len(totalFlavorList[row - 1])
                    minFlavor = totalFlavorList[row - 1][col - 1]
                    minindex = vm_flavorName.index(minFlavor)
                    # print minFlavor
                    # print minindex
            else:
                if index1 + 1 == len(pm_restCpu):
                    recordFlavor.append([])  # 重新分一台pm
                    pm_restCpu.append(pm_cpu)  # 加物理机
                    pm_restMem.append(pm_mem)
                    index1 += 1  # 物理机数量加 1
                else:
                    continue


def dynamic_programming(flavorPredict, numOfPredict):
    global pm_restCpu
    global pm_restMem
    global recordFlavor
    global pm_cpu
    global pm_mem
    global vm_flavorName
    flag = 1
    index1 = 0
    while flag == 1:

        flag = 0  # 若不改,下个循环退出

        if len(pm_restCpu) == 0:  # pm赋值
            pm_restCpu.append(pm_cpu)
            pm_restMem.append(pm_mem)
            recordFlavor.append([])  # 在第一台pm记录中加入空列表
            # 将存储的预测结果存入一维列表

            wc = [0]  # cpu大小列表
            wm = [0]  # mem大小列表
            n = 0  # 计数待分配的总数
            for i in range(len(flavorPredict)):
                if numOfPredict[i] > 0:  # 预测结果大于0
                    for j in range(numOfPredict[i]):
                        index2 = vm_flavorName.index(flavorPredict[i])
                        wc.append(vm_flavorCpu[index2])  # 添加
                        wm.append(vm_flavorMem[index2])  #
                        n += 1  # 自增计数

            x = [0] * (n + 1)  # 初始化,n+1个0
        # 以上为初始状态,只有一台pm
        pm_cpu1 = pm_cpu
        pm_mem1 = pm_mem  # 防止改变,中间变量
        mc = []  # m[i][j]剩余CPU容量j下,从第i个货物到第n个货物最大装载重量
        mm = []  # MEM
        for i in range(n + 1):
            mc.append([0] * (pm_cpu1 + 1))  # 第一行初始化为0,存储已占用CPU
            mm.append([0] * (pm_mem1 + 1))  # 初始化,存储已占用mem

        # print mm,"+++++++++++++++"
        if pm_cpu >= vm_flavorCpu[index2] and pm_mem >= vm_flavorMem[
            index2]:  # 能装下,                            这里下标0在循环中要改
            pm_restCpu_max = wc[n] - 1  # 记录mc[],mm[]两个表前几个为0,
        else:
            pm_restCpu_max = pm_cpu  # 全为0

        for j in range(pm_restCpu_max + 1):
            mc[n][j] = 0
            mm[n][j] = 0

        for j in range(wc[n], pm_cpu1 + 1):
            mc[n][j] = wc[n]  # d对于物体n
            mm[n][j] = wm[n]

        for i in range(n - 1, 0, -1):  # 倒序循环n-1---->0
            # pm_restCpu_max = min(wc[ii] - 1, c)
            if pm_cpu >= vm_flavorCpu[index2] and pm_mem >= vm_flavorMem[
                index2]:  # 能装下,                            这里下标0在循环中要改
                pm_restCpu_max = wc[i] - 1  # 记录mc[],mm[]两个表前几个为0,
            else:
                pm_restCpu_max = pm_cpu1  # 全为0

            for j in range(pm_restCpu_max + 1):
                mc[i][j] = mc[i + 1][j]
                mm[i][j] = mm[i + 1][j]
            # for j in range(w[ii], c + 1):
            #  m[ii][j] = max(m[ii + 1][j], m[ii + 1][j - w[ii]] + w[ii])
            for j in range(wc[i], pm_cpu + 1):
                # mc[i][j]=max(mc[i + 1][j], mc[i + 1][j - wc[i]] + wc[i])#自定义max(,)

                if (mc[i + 1][j] < mc[i + 1][j - wc[i]] + wc[i]) and (mm[i + 1][j - wm[i]] + wm[i] <= pm_mem):  #

                    mc[i][j] = mc[i + 1][j - wc[i]] + wc[i]
                    mm[i][j] = mm[i + 1][j - wc[i]] + wm[i]
                else:  # 放不下
                    mc[i][j] = mc[i + 1][j]
                    mm[i][j] = mm[i + 1][j]

        for i in range(1, n):  # 这里的pm_cpu都是初始值
            if mc[i][pm_cpu1] == mc[i + 1][pm_cpu1]:  # 没装
                x[i] = 0
            else:  # 装了
                x[i] = 1
                pm_cpu1 -= wc[i]

        if x[n] == mc[n][pm_cpu1]:  # 第n个装不装
            x[n] = 0
        else:
            x[n] = 1
            pm_cpu1 -= wc[i]

        # print x	,"x[]"	#打印*****************************************************
        # 解析x[]
        xnum = [0]
        for i in range(len(numOfPredict)):
            xnum.append(xnum[i] + numOfPredict[i])  # 分布区间若3,2,1则为0,3,5,6(<,<=)

        for i in range(1, n + 1):
            if x[i] == 1:
                for j in range(1, len(xnum)):
                    if xnum[j - 1] < i <= xnum[j]:
                        recordFlavor[index1].append(flavorPredict[j - 1])  # 记录某pm内容,index1为物理机序号

        for i in range(len(numOfPredict)):
            numOfPredict[i] = 0  # 清零,以便下一个循环记录
        # print x,"x[]"
        '''调试'''

        # print numOfPredict,"numOfPredict1"
        for i in range(1, n + 1):  # x[1:n+1]
            if x[i] == 0:  # 从下标1,若有0,代表没装完虚拟机
                flag = 1

                for j in range(1, len(xnum)):
                    if xnum[j - 1] < i <= xnum[j]:
                        numOfPredict[j - 1] += 1  # 记录某pm内容
                        break
        # import pdb
        # pdb.set_trace()#调试入口
        # print numOfPredict,"numOfPredict2"
        # print recordFlavor,"recordFlavor"
        # print numOfPredict,"numOfPredict"
        if flag == 1:
            pm_restCpu.append(pm_cpu)
            pm_restMem.append(pm_mem)
            recordFlavor.append([])
            index1 += 1  # 新开物理机,下标自增

            wc = [0]  # cpu大小列表
            wm = [0]  # mem大小列表
            n = 0  # 计数待分配的总数
            for i in range(len(flavorPredict)):
                if numOfPredict[i] > 0:  # 预测结果大于0
                    for j in range(numOfPredict[i]):
                        index2 = vm_flavorName.index(flavorPredict[i])
                        wc.append(vm_flavorCpu[index2])  # 添加
                        wm.append(vm_flavorMem[index2])  #
                        n += 1  # 自增计数

            x = [0] * (n + 1)  # 初始化,n+1个0
        # print flag,"flag2"


# 接口函数
def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print "input file information is none"
        return result
    global pm_cpu
    global pm_mem
    global recordFlavor

    # 读取参数
    pm_cpu, pm_mem, pm_hd, vm_num, vm_flavor, vm_cpu, \
    vm_mem, kind, begin, end, predict_day, pre_x = Read_Par(input_lines)

    print "predict_day = ", predict_day
    print "pre_x = ", pre_x
    # 预测时间的总天数
    # predict_day = end - begin + 1   # 加1
    predict_week = predict_day / 7
    # 需要预测的横坐标

    pre_week_x = []
    for i in range(predict_week):
        pre_week_x.append(begin / 7 + i)

    # print "pre_week_x = ", pre_week_x
    # 读取测试数据
    vm_id, vm_f, vm_date_int = Read_Data(ecs_lines)

    # day_flavor_num1, day1 = count_flavor(vm_f, vm_flavor, vm_date_int)
    # print "day = ", day1
    # print "day_flavor_num = ", day_flavor_num1

    day_flavor_num, day, data_x = count_day(vm_f, vm_flavor, vm_date_int)

    print "day_flavor_num = ", day_flavor_num
    # day_flavor_num1, day1, data_x1 = count_flavor(vm_f, vm_flavor, vm_date_int)

    # 线性回归
    num_dict = Linear_regression(data_x, day_flavor_num, pre_x)  # 得分:63.667
    # num_dict = Linear_regression(week_x, week_flavor_num, pre_week_x, vm_num)   # 得分:28.511

    # 神经网络
    # nn = BPNeuralNetwork()
    # num_dict = nn.bp_predict(data_x, day_flavor_num, pre_x)  # 60......
    # print "BP_num_dict = ", num_dict

    # ploy_fit
    # num_dict = ploy_fit.predict_vm(data_x, day_flavor_num, pre_x)

    # 随机大法
    num_dict = large_rand(num_dict)

    total_num = 0  # 记录预测的虚拟机的总数

    for i in range(len(num_dict)):
        flavorPredict.append(vm_flavor[i])  # 记录预测虚拟机类型
        numOfPredict.append(num_dict[i])  # 记录预测相应虚拟机类型数量
        result.append(vm_flavor[i] + " " + str(num_dict[i]))

    for i in range(len(num_dict)):
        total_num += numOfPredict[i]

    result.insert(0, total_num)

    # arrange(flavorPredict, numOfPredict)  # 调用分配函数
    arrange_bestfit(flavorPredict, numOfPredict)
    # dynamic_programming(flavorPredict, numOfPredict)
    temp = ""
    temp2 = '\n' + str(len(recordFlavor))

    if len(recordFlavor) != 0:  # 物理机>0
        result.append(temp2)
        for i in range(len(recordFlavor)):  # 物理机数量
            # print i+1,			#, 表示不换行
            temp = str(i + 1)
            # for j in range(len(recordFlavor[i])):		#记录的每台物理机分配的虚拟机
            for m in range(len(vm_flavorName)):  # 一台pm中各种vm计数
                count_vm = recordFlavor[i].count(vm_flavorName[m])
                if count_vm > 0:
                    temp = temp + " " + vm_flavorName[m] + " " + str(count_vm)
            result.append(temp)
            temp = ""
    else:  # 物理机==0
        result.append(temp2)

    return result
