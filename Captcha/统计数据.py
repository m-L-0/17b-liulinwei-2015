import csv  # 导入csv库


csv_reader = csv.reader(open('./data/captcha/labels/labels.csv', encoding='utf-8'))  # 读取csv文件
data_list = []  # 定义存储图片显示的数据的列表
data_dict = {1: 0, 2: 0, 3: 0, 4: 0}  # 已验证码上数字的个数为键，定义一个字典
for row in csv_reader:
    data_list.append(row[1])
data_sum = len(data_list)  # 统计数据集总数
# 统计不同位数验证码的数量
for j in range(data_sum):
    if len(data_list[j]) == 1:
        data_dict[1] += 1
    if len(data_list[j]) == 2:
        data_dict[2] += 1
    if len(data_list[j]) == 3:
        data_dict[3] += 1
    if len(data_list[j]) == 4:
        data_dict[4] += 1
# 统计不同位数验证码的比例
data_1 = data_dict[1]/data_sum
data_2 = data_dict[2]/data_sum
data_3 = data_dict[3]/data_sum
data_4 = data_dict[4]/data_sum
print('统计总数是: %d ,1位数的验证码的数量: %d ,2位数的验证码的数量: %d ,3位数的验证码的数量: %d ,4位数的验证码的数量: %d' %(data_sum, data_dict[1], data_dict[2], data_dict[3], data_dict[4]))
print('统计不同位数验证码的比例:')
print('1位:%.3f ,2位:%.3f ,3位:%.3f ,4位:%.3f' % (data_1, data_2, data_3, data_4))
