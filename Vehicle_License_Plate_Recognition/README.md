# 作业内容：

## 使用卷积神经网络对分割好的车牌字符进行识别

### 主要操作：

1. [将分类好的图片及其标签序号存入到TFRecord文件中。](https://github.com/m-L-0/17b-liulinwei-2015/blob/master/Vehicle_License_Plate_Recognition/code/tf_write_read.ipynb)

2. [读取TFRecord文件：数据解码，reshape(恢复数据形状)。shuffle_batch。然后还有归一化处理、色彩空间变化、转换为灰色图片等操作。从tfrecord文件读取数据，做批次处理](https://github.com/m-L-0/17b-liulinwei-2015/blob/master/Vehicle_License_Plate_Recognition/code/tf_write_read.ipynb)

3. [设计卷积神经网络结构并利用卷积神经网络对字母数字分别进行训练。(模型保存(train.Saver()、正则化)](https://github.com/m-L-0/17b-liulinwei-2015/blob/master/Vehicle_License_Plate_Recognition/code/cnn_saver.ipynb)

4. [利用测试集对卷积神经网络进行检测，并得到识别正确率,召回率。](https://github.com/m-L-0/17b-liulinwei-2015/blob/master/Vehicle_License_Plate_Recognition/code/CNN.ipynb)

5. [统计每类字符的数量与比例并利用图表展示(直方图、饼状图)](https://github.com/m-L-0/17b-liulinwei-2015/blob/master/Vehicle_License_Plate_Recognition/code/Draw%20diagrams.ipynb)

