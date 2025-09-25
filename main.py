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
from scipy.signal import find_peaks


import numpy as np
import pandas as pd
from scipy.integrate import simpson


def extract_features_from_file(filepath, ranges=[(-0.2, 0.2), (0.6, 1)]):

    # ranges = [(-np.inf, np.inf)]
    ranges = [(-0.5, 0), (0.75, 0.85)]

    """从一个xlsx文件提取特征向量（区间内所有曲线矩阵的均值和标准差）"""
    df = pd.read_excel(filepath)

    # 去掉表头中的空格等
    df.columns = [str(c).strip() for c in df.columns]

    # 横坐标 (Voltage)
    voltage = df.iloc[:, 0].values.ravel()   # 保证一维
    currents = df.iloc[:, 1:].values        # 其他列为电流矩阵 (n点 × m曲线)

    # print(currents.shape)

    features = []

    cnt = 0
    for vmin, vmax in ranges:
        cnt += 1
        mask = (voltage >= vmin) & (voltage <= vmax)
        v_sub = voltage[mask]
        c_sub = currents[mask, :]   # 取所有曲线在这个电压区间的点

        all_peaks = []
        for i in range(c_sub.shape[1]):
            curve = c_sub[:, i]

            # 找局部极大值索引
            peak_idx, _ = find_peaks(curve, width=3)

            if len(peak_idx) > 0:
                # print(peak_idx)
                peak_vals = curve[peak_idx]
                all_peaks.extend(peak_vals)

        if len(all_peaks) == 0:
            all_peaks = [0]  # 区间内没有极大值

        all_peaks = np.array(all_peaks) * 100000
        peak_max = np.max(all_peaks)
        peak_min = np.min(all_peaks)
        peak_mean = np.mean(all_peaks)

        # print(c_sub.shape)

        # --- 电流幅值特征 ---
        mean_range = np.mean(np.max(c_sub, axis=0) - np.min(c_sub, axis=0)) * 10000


        # 把这些都作为特征
        if cnt == 1:
            # --- 梯度矩阵 ---
            grads = np.gradient(c_sub, v_sub, axis=0)  # dI/dV

            grads = grads[grads > 0]

            # print(grads)

            # 方法1：和电流一样，用 (max - min) 的平均值表示
            grad_range = np.mean(np.max(grads, axis=0) - np.min(grads, axis=0)) * 10000

            # 方法2：也可以补充整体均值 / 标准差
            grad_mean = np.mean(grads)
            grad_std = np.std(grads)
            
            diff = np.sum(np.max(c_sub, axis=0) - np.min(c_sub, axis=0)) * 10000
            grad = np.gradient(np.sort(all_peaks))
            grad_max = np.max(grad)

            features.extend([grad_max])
        else:
            features.extend([peak_mean])


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


def train_and_evaluate(X, y, method="svm", seed=1):
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


def preprocess_outliers(X, y, method="iqr", threshold=0.2):
    """
    对于每个标签分组的数据，逐列检测离群值并替换为正常值均值。

    参数:
        X : numpy.ndarray, shape (n_samples, n_features)
        y : numpy.ndarray, shape (n_samples,)
        method : str, "iqr" 或 "zscore"
        threshold : float, IQR系数 (默认1.5) 或 z-score阈值 (默认3)

    返回:
        X_new : numpy.ndarray (处理后的矩阵)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_new = X.copy()

    for label in np.unique(y):   # 多个标签都会单独处理
        mask = (y == label)
        X_group = X[mask]  # 当前标签对应的子矩阵

        for col in range(X.shape[1]):
            col_data = X_group[:, col]

            # 1. 离群值检测
            if method == "iqr":
                Q1, Q3 = np.percentile(col_data, [25, 75])
                IQR = Q3 - Q1
                lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
                outliers = (col_data < lower) | (col_data > upper)
            elif method == "zscore":
                mean, std = np.mean(col_data), np.std(col_data)
                z = (col_data - mean) / (std + 1e-8)
                outliers = np.abs(z) > threshold
            else:
                raise ValueError("Unknown method, choose 'iqr' or 'zscore'")

            # 2. 替换离群值
            normal_vals = col_data[~outliers]
            if len(normal_vals) > 0:
                replacement = np.mean(normal_vals)
                col_data[outliers] = replacement

            # 写回原矩阵
            X_new[mask, col] = col_data

    return X_new

if __name__ == "__main__":
    base_dir = "data"  # 三个子文件夹所在目录
    method = "svm"
    X, y = load_dataset(base_dir)
    # A = np.hstack([X, np.transpose([y])])
    # print(A)
    X = preprocess_outliers(X, y)
    A = np.hstack([X, np.transpose([y])])
    print(A)
    train_and_evaluate(X, y, method=method)

