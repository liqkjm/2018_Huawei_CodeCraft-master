# coding=utf-8
import sys
import os
import predict


def main():

    # 训练数据："TrainData_2015.1.1_2015.2.19.txt"
    # 测试数据："TestData_2015.2.20_2015.2.27.txt"
    # 练习数据：
    # "data_2015_2.txt"    ->  "data_2015_5.txt"
    # "data_2015_12.txt"
    # "data_2016_1.txt"
    file_path = ["TrainData_2015.1.1_2015.2.19.txt", "TestData_2015.2.20_2015.2.27.txt", "data_2015_1.txt", "data_2015_2.txt", "data_2015_3.txt", "data_2015_4.txt", "data_2015_5.txt",
                "data_2015_12.txt", "data_2016_1.txt"]

    for ii in range(1):
        ii = 7
        print "\n第", ii + 1, "组练习数据：",file_path[ii]
        print 'main function begin.'
        ecs_infor_array = read_lines(file_path[ii])
        input_file_array = read_lines("input_5flavors_cpu_7days.txt")

        # implementation the function predictVm
        predic_result = predict.predict_vm(ecs_infor_array, input_file_array)

        # write the result to output file
        for i in range(len(predic_result)):
            print predic_result[i]
    print 'main function end.'


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print 'file not exist: ' + file_path
        return None


if __name__ == "__main__":
    main()
