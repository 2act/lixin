# 分类任务

1. 运行case1.py即可。

1. 查看结果直接看**classified_results.txt**。

# 预测任务

1. 本case采用所有输入数据的**公共列**来进行训练和测试(80%的有效电流列训练，20%测试，不包括基准电流列)，但data/Mixture的公共列不足，因此舍弃了部分结果，保留了足够多的公共列，具体见case2.py的**load_dataset**函数。

1. 运行case2.py，注意设置**skip_count = 0**来全部重新运行。

1. 如果只是查看结果，直接看predicted_results文件夹下的结果即可，mean_metrics.txt保存的是所有测试数据的平均指标。

1. **predicted_results/*/predicted_dfs**文件夹下保存的df用于绘图。
