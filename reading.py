#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import time
import string


def Read_Par():  # 读取参数，包括物理服务器的信息，以及虚拟机的规格信息 和 预测起始时间等
    # type: () -> object
    f = open(r"D:\\python27\untitled\input_5flavors_cpu_7days.txt", "r")
    lines = f.readlines()
    f.close()

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

    begintime = lines[vm_num + 6]
    endtime = lines[vm_num + 7]
    return pm_cpu, pm_mem, pm_hd, vm_num, vm_flavor, vm_cpu, vm_mem, kind, begintime, endtime

def Read_Data():
    # type: () -> object
    f = open(r"D:\\python27\untitled\TrainData_2015.1.1_2015.2.19.txt", "r")
    lines = f.readlines()
    f.close()
    # 统计每种虚拟机规格的个数
    # numofflavor = []
    vm_id = []
    vm_f = []
    vm_date = []

    for i in range(len(lines)):
        line = lines[i].split("\t")
        vm_id.append(line[0])
        vm_f.append(line[1])
        vm_date.append(line[2])

    return vm_id, vm_f, vm_date

def Decode_Date(vm_date):
    vm_date_int = []
    for i in range(len(vm_date)):
        #print vm_date[i]
        date1 = vm_date[i].split(" ")[0].split("-")
        #print date1
        date_int = int(date1[1])*30+int(date1[2])
        vm_date_int.append(date_int)
    return vm_date_int

def Sata_flavor(vm_f1, vm_flavor1, vm_date_int):
    # type: (object, object) -> object
    numofflavor1 = []  # 统计每种虚拟机规格的个数!!!!统计每天每种规格的个数
    day_max = 100
    day = 0             #统计一共有多少天, 记住最后天数 + 1（包括 0 ）
    day_flavor_num = [] #二维列表， 统计每一天每规格的个数
    day_flavor = []     #记录相同日期内的所有规格

    for ii in range(day_max):    #暂时假设跨度最大天数为100天, 初始化二维列表为0
        day_flavor_num.append([])
        #for jj in range(len(vm_flavor1)):
            #day_flavor_num[ii].append(0)
        #print "test: ", numofflavor1[ii]day_max

    print "day_flavor_num[0] = ", day_flavor_num[0]

    vm_date_int.append(0)       #添加一个标记点，用于接下来的判断

    print "vmn_flavor1 = ", vm_flavor1
    for r in range(len(vm_date_int)-1):       #遍历所有日期，把相同日期的规格先放入一个列表，
                                              #再使用list的count函数，统计每个规格的个数
        if vm_date_int[r] != vm_date_int[r+1]:      #不同日期，天数加一
            day_flavor.append(vm_f1[r])
            #print "the day is end and the list of day_flavor is : ",day_flavor
            for i in range(len(vm_flavor1)):       #对列表统计要求的规格的个数
                temp = day_flavor.count(vm_flavor1[i])     #该列表该规格的个数
                #print "temp = ", temp
                day_flavor_num[day].append(temp)       #二维列表， 一维为天数， 二维为规格， 其值为 该天该规格的个数
            #day_flavor_num[day].append(day)
            day = day + 1
            day_flavor = []        #重新赋值为空集，开始记录下一天
        else:                                           #与后面一条日期相同，对其进行统计
            day_flavor.append(vm_f1[r])
            #print "the day is not end and the day_flavor is : ", day_flavor

    return day_flavor_num, day

#线性回归

def compute_error(b, m, data):
    totalError = 0
    for i in range(0,len(data)):
        x = data[i,0]
        y = data[i,1]
        totalError += (y-(m*x+b))**2

    return totalError

def compute_gradient(b_current,m_current,data ,learning_rate):

    b_gradient = 0
    m_gradient = 0

    N = float(len(data))
    #Two ways to implement this
    #first way
    for i in range(0,len(data)):
        x = data[i,0]
        y = data[i,1]
    #computing partial derivations of our error function
        b_gradient = -(2/N)*sum((y-(m*x+b))^2)
        m_gradient = -(2/N)*sum(x*(y-(m*x+b))^2)
        b_gradient += -(2/N)*(y-((m_current*x)+b_current))
        m_gradient += -(2/N) * x * (y-((m_current*x)+b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b,new_m]
def optimizer(data, initial_b, initial_m, learning_rate, num_iter):
    a = 0

def linear_regression(day, v_f, day_f_num):
    #file = open("")
    #file = open(r"D:\\python27\untitled\data.csv", "r")
    #data = file.readlines()

    learning_rate = 0.001
    initial_b =0.0
    initial_m = 0.0
    num_iter = 1000
    print 'initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n' \
        .format(initial_b, initial_m, compute_error(initial_b, initial_m, data))

    # optimizing b and m
    [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)

    # print final b m error
    print 'final formula parmaters:\n b = {1}\n m={2}\n error of end = {3} \n'.\
        format(num_iter, b, m, compute_error(b, m, data))

#求平均数
def predict(num_dict, vm_flavor, begintime, endtime):      #预测
    #bt =
    print "预测的虚拟机总数：", len(num_dict)
    for i in range(len(num_dict)):
        print vm_flavor[i], int(num_dict[i])/14

def output():
    print "Reading input.txt..."
    pm_cpu, pm_mem, pm_hd, vm_num, vm_flavor, vm_cpu, vm_mem, kind, begintime, endtime = Read_Par()

    print pm_cpu, pm_mem, pm_hd
    print vm_num
    for i in range(vm_num):
        print vm_flavor[i], vm_cpu[i], vm_mem[i]
    print kind
    print "begintime = ", begintime, "endtime = ", endtime

    print "Reading TrainData..."
    vm_id, vm_f, vm_date = Read_Data()

    print "vm_date = ", vm_date
    vm_date_int = Decode_Date(vm_date)

    print "vm_date_int: ", len(vm_date_int)
    print "vm_date_int = ", vm_date_int
    day_flavor_num, day = Sata_flavor(vm_f, vm_flavor, vm_date_int)

    print day_flavor_num[0:day]
    print "The total day is :", day

    num_dict = []
    for i in range(len(vm_flavor)):
        num_dict.append(0)
    temp = 0
    for i in range(day):
        for j in range(len(vm_flavor)):
            #temp = temp + int(day_flavor_num[i][j])
            num_dict[j] = num_dict[j] + int(day_flavor_num[i][j])

    for i in range(len(vm_flavor)):
        print "The  num of ",vm_flavor[i], " is : ", num_dict[i]

    predict(num_dict, vm_flavor, begintime, endtime)
    return 0

output()
