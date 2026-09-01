# 分类任务

1. 运行case1.py即可，采用的SVM（随机森林算法也可以，直接切换即可），用的不同电压区间的最值以及标准差作为特征向量，详见函数**extract_features_from_file**。

1. 查看结果直接看**classified_results.txt**。

# 分类-r1

1. 先运行这个保存模型：python case1_r1.py
1. 再运行这个获取总的报告：python case1_r1.py --mode predict --input_dir data_r1
1. 运行如下命令获取3个单独的报告，--output为指定的文件名

```bash
python case1_r1.py --mode predict --input data_r1/cabbage/4.xlsx --output ./case1_r1_cabbage.txt
python case1_r1.py --mode predict --input data_r1/strawberry/1.xlsx --output ./case1_r1_strawberry.txt
python case1_r1.py --mode predict --input data_r1/grape/3.xlsx --output ./case1_r1_grape.txt
```

# 预测任务

1. 本case采用所有输入数据的**公共列**来进行训练和测试(80%的有效电流列训练，20%测试，不包括基准电流列)，但data/Mixture的公共列不足，因此舍弃了部分结果，保留了足够多的公共列，具体见case2.py的**load_dataset**函数。

1. 运行case2.py，注意设置**skip_count = 0**来全部重新运行。

1. 如果只是查看结果，直接看predicted_results文件夹下的结果即可，mean_metrics.txt保存的是所有测试数据的平均指标。

1. **predicted_results/*/predicted_dfs**文件夹下保存的df用于绘图。
