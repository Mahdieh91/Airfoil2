import time
import numpy as np
import os
import subprocess
from glob import glob

xfoil_path = 'E:\\D_PHD\\D6_Project\\pre_cp\\xfoil\\xfoil\\XFOIL.exe'

# 加载数据
data = np.loadtxt('show_comparesion.dat', delimiter=',')
loc_x = np.loadtxt('loc_x.dat')
airfoils = data[:, :199]

for i in range(len(airfoils)):
    print(f'getting airfoil data.... {i + 1}/{len(airfoils)}')

    airfoil = airfoils[i, :]
    # 0.计算翼型最大相对厚度
    airfoil_reverse = airfoil[::-1]  # 颠倒顺序
    airfoil_max_thick = round(np.max(airfoil - airfoil_reverse), 6)  # 保留6位小数

    # 1.生成单个翼型文件
    airfoil_loc = np.transpose(np.vstack((loc_x.reshape(1, -1), airfoil.reshape(1, -1))))
    np.savetxt('airfoil_temp.dat', airfoil_loc, fmt='%0.6f')
    time.sleep(0.01)
    with open('airfoil_temp.dat', 'r+') as file:
        content = file.read()
        new_content = 'airfoil' + '\n' + content
        file.seek(0)  # 将指针移动到文件开头
        file.write(new_content)
    file.close()
    time.sleep(0.01)

    # 2.生成力系数
    # 启动XFOIL进程
    xfoil_process = subprocess.Popen([xfoil_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
    xfoil_process.stdin.write("\n")
    xfoil_process.stdin.flush()
    # load后面有个空格，害死人
    xfoil_process.stdin.write("load " + 'airfoil_temp.dat' + "\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("oper\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("iter 500\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("v\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write('1e5' + "\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("pacc\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("airfoil_temp_1e5_a5" + ".dat\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("a5" + "\n")
    xfoil_process.stdin.flush()
    # print(f'coefficients getted')

    # 关闭XFOIL进程
    xfoil_process.stdin.close()
    xfoil_process.stdout.close()
    xfoil_process.stderr.close()
    time.sleep(0.01)

    start_time = time.time()
    while time.time() - start_time < 0.51:
        return_code = xfoil_process.poll()
        if return_code is not None:
            break
        time.sleep(0.1)

    if return_code is None:
        xfoil_process.terminate()
        time.sleep(0.1)

    # 3.生成压力分布
    # 启动XFOIL进程
    xfoil_process = subprocess.Popen([xfoil_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
    # 发送命令到XFOIL并获取输出
    xfoil_process.stdin.write("\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("load " + 'airfoil_temp.dat' + "\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("oper\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("iter 500\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("v\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write('1e5' + "\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("a5" + "\n")
    xfoil_process.stdin.flush()
    xfoil_process.stdin.write("cpwr\n")
    xfoil_process.stdin.flush()


    xfoil_process.stdin.write("airfoil_temp_1e5_a5" + ".cp\n")
    xfoil_process.stdin.flush()
    # print(f'cp getted')

    # 关闭XFOIL进程
    xfoil_process.stdin.close()
    xfoil_process.stdout.close()
    xfoil_process.stderr.close()
    time.sleep(0.01)

    start_time = time.time()
    while time.time() - start_time < 0.51:
        return_code = xfoil_process.poll()
        if return_code is not None:
            break
        time.sleep(0.1)

    if return_code is None:
        xfoil_process.terminate()
        time.sleep(0.1)

    # 4.读取Xfoil生成的数据
    if os.path.exists("airfoil_temp_1e5_a5.dat") and os.path.exists("airfoil_temp_1e5_a5.cp"):  # 个别情况.dat文件不存在，会报错
        # 4.1 读取力系数
        c3_name = "airfoil_temp_1e5_a5.dat"
        with open(c3_name, "r") as input_file:
            lines = input_file.readlines()

        # 如果计算不收敛,也就是该文件里没有力系数,就进行下一步迭代
        if len(lines) < 13:
            continue

        data_line = lines[12]
        numbers = data_line.strip().split()
        cl = round(float(numbers[1]), 5)
        cd = round(float(numbers[2]), 5)
        cm = round(float(numbers[4]), 5)

        # 4.2 读取压力分布
        # 读取cp文件
        c3_name = "airfoil_temp_1e5_a5.cp"
        with open(c3_name, "r") as input_file:
            lines = input_file.readlines()
        # 提取数据
        if len(lines) > 1:  # 个别情况.cp文件里只有1行，无数据，会报错
            cp = []
            data_lines = lines[3:]

            # 读取cp
            for data_line in data_lines:
                number1 = data_line.strip().split()
                first_number = round(float(number1[0]), 5)
                # 1e6 a16 第80行 第二个数和第三个数挨着了，导致被识别为一个数了
                number3 = data_line[19:].strip().split()
                third_number = round(float(number3[0]), 5)
                cp.append(third_number)
            # 将元祖转换为数组
            cp = np.array(cp)

            # 将cp数组中的元素连接成一个字符串方便写入文件
            cp_str = " ".join(map(str, cp))

            # 将airfoil数组中的元素连接成一个字符串方便写入文件
            airfoil_str = " ".join(map(str, airfoil))
            # 写入文件,'a'是追加内容
            with open('show_comparesion_data.dat', "a") as output_file:
                output_file.write(f"{airfoil_str} {airfoil_max_thick} {cl} {cd} {round(cl/cd, 6)} {cm} {cp_str}\n")

            # aa = np.loadtxt('airfoils_recon_data.dat')

    # 5.数据读取完毕后，删除相应的力系数和力矩系数文件
    data_pattern = "airfoil_temp_1e5_a5.*"
    datas = glob(data_pattern)  # 使用 glob 匹配文件名
    # 个别文件被占用无法删除文件
    for data in datas:
        retries = 0
        while retries < 10:
            try:
                os.remove(data)
                break
            except Exception as e:
                retries += 1
                time.sleep(0.5)
        else:
            print('无法删除文件，报错', data)