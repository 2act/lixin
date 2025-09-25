import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.integrate import simpson  # 数值积分
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


import numpy as np
import pandas as pd
from scipy.integrate import simpson


def extract_features_from_file(filepath, ranges=[(-0.2, 0.2), (0.6, 1)]):

    # ranges = [(-np.inf, np.inf)]
    ranges = [(-0.5, 0), (0.6, 0.9)]

    """从一个xlsx文件提取特征向量（区间内所有曲线矩阵的均值和标准差）"""
    df = pd.read_excel(filepath)

    # 去掉表头中的空格等
    df.columns = [str(c).strip() for c in df.columns]

    # 横坐标 (Voltage)
    voltage = df.iloc[:, 0].values.ravel()   # 保证一维
    currents = df.iloc[:, 1:].values        # 其他列为电流矩阵 (n点 × m曲线)

    # print(currents.shape)

    features = []

    for vmin, vmax in ranges:
        mask = (voltage >= vmin) & (voltage <= vmax)
        v_sub = voltage[mask]
        c_sub = currents[mask, :]   # 取所有曲线在这个电压区间的点

        # print(c_sub.shape)

        # --- 电流幅值特征 ---
        mean_range = np.mean(np.max(c_sub, axis=0) - np.min(c_sub, axis=0)) * 10000

        # --- 梯度矩阵 ---
        grads = np.gradient(c_sub, v_sub, axis=0)  # dI/dV

        grads = grads[grads > 0]

        print(grads)

        # 方法1：和电流一样，用 (max - min) 的平均值表示
        grad_range = np.mean(np.max(grads, axis=0) - np.min(grads, axis=0)) * 10000

        # 方法2：也可以补充整体均值 / 标准差
        grad_mean = np.mean(grads)
        grad_std = np.std(grads)

        # 把这些都作为特征
        features.extend([grad_range])

        # if c_sub.size == 0:
        #     # 如果该区间没有点，则补 0 或 NaN（推荐 NaN，后续再处理）
        #     features.extend([np.nan, np.nan])
        # else:
        #     # 对矩阵求均值和标准差（所有点 × 所有曲线）
        #     f_mean = np.mean(c_sub)
        #     f_std = np.std(c_sub)
        #     features.extend([f_std])

    print(f"{filepath} : {np.round(features, 2)}")

    return np.array(features)


def load_dataset(base_dir):
    """读取所有文件，构建 (X,y)"""
    X, y = [], []
    class_labels = os.listdir(base_dir)

    for label in class_labels:
        folder = os.path.join(base_dir, label)
        for filepath in glob.glob(os.path.join(folder, "*.xlsx")):
            feat = extract_features_from_file(filepath)
            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)


def train_and_evaluate(X, y, method="svm", seed=0):
    if method == "svm":
        """训练分类器并评估"""
        clf = SVC(kernel='rbf', probability=True)  # 也可换成 RandomForestClassifier
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    elif method == "rf":
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=400, random_state=seed)
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 交叉验证预测
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    print("分类报告：")
    print(classification_report(y, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    base_dir = "data"  # 三个子文件夹所在目录
    method = "svm"
    X, y = load_dataset(base_dir)
    train_and_evaluate(X, y, method=method)

