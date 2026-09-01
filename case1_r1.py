import glob
import os
import argparse
import joblib

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 必须导入以启用3D
from sklearn.svm import SVC


def basic_features(curve):
    return [
        np.mean(curve),  # 均值
        np.std(curve),  # 标准差
        np.min(curve),
        np.max(curve),  # 极值
        np.median(curve),  # 中位数
        np.percentile(curve, 25),
        np.percentile(curve, 75),  # 分位数
        np.mean(np.diff(curve)),  # 一阶差分均值
        np.std(np.diff(curve)),  # 一阶差分标准差
    ]


def signal_features(curve):
    curve = curve - np.mean(curve)
    fft_vals = np.abs(rfft(curve))
    freqs = rfftfreq(len(curve))

    peaks, _ = find_peaks(curve)
    num_peaks = len(peaks)
    mean_peak_height = np.mean(curve[peaks]) if num_peaks > 0 else 0

    return [
        np.sum(fft_vals[1:5]),  # 低频能量
        np.sum(fft_vals[5:20]),  # 高频能量
        np.argmax(fft_vals[1:]),  # 主频索引
        num_peaks,  # 峰数
        mean_peak_height,  # 峰高平均值
    ]


def geometric_features(curve):
    x = np.arange(len(curve))
    diff = np.max(curve) - np.min(curve)
    deriv = np.gradient(curve)
    curvature = np.mean(np.abs(np.gradient(deriv)))
    area = np.trapz(curve, x)
    roughness = np.mean(np.abs(np.diff(curve)))

    # return [area, curvature, roughness]
    return [np.max(curve)]


def extract_features(curve):
    f1 = basic_features(curve)
    f2 = signal_features(curve)
    f3 = geometric_features(curve)
    return np.hstack([f1, f2, f3])


def extract_features_from_file(filepath, ranges=[(-0.2, 0.2), (0.6, 1)]):
    # ranges = [(-np.inf, np.inf)]

    """从一个xlsx文件提取特征向量"""
    df = pd.read_excel(filepath)

    # 去掉表头中的空格等
    df.columns = [str(c).strip() for c in df.columns]

    # 横坐标 (Voltage)
    voltage = df.iloc[:, 0].values.ravel()  # 保证一维
    currents = df.iloc[:, 1:].values  # 其他列为电流矩阵 (n点 × m曲线)

    # print(currents.shape)

    features = []

    for i in range(currents.shape[1]):
        current = currents[:, i]
        cnt = 0
        feat = []
        for vmin, vmax in ranges:
            cnt += 1
            mask = (voltage >= vmin) & (voltage <= vmax)
            v_sub = voltage[mask]
            c_sub = current[mask]  # 取所有曲线在这个电压区间的点

            # 把这些都作为特征
            if cnt == 1:
                feat_1 = np.max(c_sub)
                feat.append(feat_1)
                # print(feat)
            elif cnt == 2:
                feat_2 = np.std(np.diff(c_sub))
                feat.append(feat_2)
            elif cnt == 3:
                feat_3 = np.std(np.diff(c_sub))
                feat.append(feat_3)
        # 子区间循环结束
        features.append(feat)

    # print(f"{filepath} : {np.round(features, 2)}")

    return np.array(features)


def load_dataset(base_dir, ranges):
    """读取所有文件，构建 (X,y)"""
    X, y = [], []
    class_labels = os.listdir(base_dir)

    for label in class_labels:
        folder = os.path.join(base_dir, label)
        for filepath in glob.glob(os.path.join(folder, "*.xlsx")):
            feats = extract_features_from_file(filepath, ranges)
            for feat in feats:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


def build_classifier(method="svm", seed=1):
    """创建分类器，参数与原程序保持一致。"""
    if method == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, C=1, gamma="scale", random_state=seed),
        )
    elif method == "rf":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=400, random_state=seed),
        )
    else:
        raise ValueError("Unknown method, choose 'svm' or 'rf'")


def train_and_evaluate(X, y, method="svm", seed=1):
    """训练分类器并评估"""
    clf = build_classifier(method=method, seed=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 交叉验证预测
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    report = classification_report(y, y_pred)
    print("分类报告：")
    print(report)

    c_matrix = confusion_matrix(y, y_pred)
    print("混淆矩阵：")
    print(c_matrix)

    return report, c_matrix


def preprocess_outliers(X, y, method="iqr", threshold=0.5):
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

    for label in np.unique(y):  # 多个标签都会单独处理
        mask = y == label
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


def save_trained_model(
    X, y, ranges, method="svm", model_file="case1_model.joblib", seed=1
):
    """使用全部训练数据训练最终模型并保存。"""
    clf = build_classifier(method=method, seed=seed)
    clf.fit(X, y)

    # ranges 一并保存，保证预测时使用与训练完全相同的特征提取区间
    joblib.dump(
        {
            "model": clf,
            "ranges": ranges,
            "method": method,
        },
        model_file,
    )

    print(f"模型已保存：{model_file}")
    return clf


def get_true_labels_from_file(filepath):
    """
    根据预测输入 Excel 的曲线表头生成真实标签。

    真实标签规则（仅用于 predict 模式下的评估）：
    1. Excel 第 1 列为 Voltage，不参与分类；
    2. 后续每一列代表一条待分类曲线；
    3. 曲线表头去除首尾空格后，如果以字符 "0" 开头，则该曲线真实标签为 "None"；
    4. 否则，该曲线真实标签为 "Mixture"。

    例如：表头 "0"、"0.1"、"05ul" -> None；
          表头 "1"、"5"、"100" -> Mixture。
    """
    df = pd.read_excel(filepath, nrows=0)
    columns = [str(c).strip() for c in df.columns]

    if len(columns) < 2:
        raise ValueError(f"{filepath} 至少需要包含 Voltage 列和 1 条待分类曲线")

    curve_headers = columns[1:]
    true_labels = [
        "None" if header.startswith("0") else "Mixture" for header in curve_headers
    ]
    return curve_headers, np.array(true_labels)


def predict_file(
    input_file, model_file="case1_model.joblib", save_file="classified_results_r1.txt"
):
    """加载训练好的模型，对一个新的xlsx文件直接进行分类。"""
    saved = joblib.load(model_file)
    clf = saved["model"]
    ranges = saved["ranges"]

    X = extract_features_from_file(input_file, ranges)
    y_pred = clf.predict(X)
    curve_headers, y_true = get_true_labels_from_file(input_file)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"{input_file} 的特征数量({len(y_pred)})与曲线表头数量({len(y_true)})不一致"
        )

    print(f"输入文件：{input_file}")
    print("分类结果：")
    for i, (header, true_label, pred_label) in enumerate(
        zip(curve_headers, y_true, y_pred), start=1
    ):
        print(f"曲线 {i} [{header}]: 真实={true_label}, 预测={pred_label}")

    labels = ["None", "Mixture"]
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    c_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print("分类报告：")
    print(report)
    print("混淆矩阵：")
    print(c_matrix)
    print("混淆矩阵标签顺序：", labels)

    with open(save_file, "w", encoding="utf-8") as f:
        f.write(f"输入文件：{input_file}\n")
        f.write("真实标签规则：曲线表头以0开头 -> None；否则 -> Mixture。\n")
        f.write("分类结果：\n")
        for i, (header, true_label, pred_label) in enumerate(
            zip(curve_headers, y_true, y_pred), start=1
        ):
            f.write(f"曲线 {i} [{header}]: 真实={true_label}, 预测={pred_label}\n")
        f.write(f"\n分类报告：\n{report}")
        f.write(f"混淆矩阵（标签顺序 {labels}）：\n{c_matrix}\n")

    print(f"file {save_file} saved.")
    return y_pred


def predict_directory(
    input_dir, model_file="case1_model.joblib", save_file="classified_results_r1.txt"
):
    """
    递归扫描 input_dir 下所有 xlsx 文件并统一分类、评估。

    每个 xlsx 的第 1 列为 Voltage，后续每列是一条待分类曲线。

    【当前测试集真实标签规则】
      - 曲线表头去除首尾空格后，以 "0" 开头 -> None
      - 其他表头                              -> Mixture

    注意：模型本身仍然是原来的多分类模型，因此预测结果可能出现
    None、Mixture 之外的其他已训练类别。此类结果必须作为真实误分类保留，
    不能在混淆矩阵中丢弃。

    本函数同时输出：
      1. 完整多分类 classification report；
      2. 完整多分类 confusion matrix（包含模型全部类别）；
      3. 当前测试集简化矩阵：预测列为 None / Mixture / Other；
      4. 总正确数、总错误数与总体准确率；
      5. 每条曲线的详细预测 CSV。
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入目录不存在：{input_dir}")

    filepaths = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.xlsx"), recursive=True)
    )
    if not filepaths:
        raise FileNotFoundError(f"在目录 {input_dir} 下没有递归找到任何 xlsx 文件")

    saved = joblib.load(model_file)
    clf = saved["model"]
    ranges = saved["ranges"]

    all_true = []
    all_pred = []
    detail_rows = []

    print(f"输入目录：{input_dir}")
    print(f"递归找到 {len(filepaths)} 个 xlsx 文件。")

    for filepath in filepaths:
        X = extract_features_from_file(filepath, ranges)
        y_pred = clf.predict(X)
        curve_headers, y_true = get_true_labels_from_file(filepath)

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"{filepath} 的特征数量({len(y_pred)})与曲线表头数量({len(y_true)})不一致"
            )

        relpath = os.path.relpath(filepath, input_dir)
        print(f"\n输入文件：{relpath}")
        print("分类结果：")

        for i, (header, true_label, pred_label) in enumerate(
            zip(curve_headers, y_true, y_pred), start=1
        ):
            pred_label = str(pred_label)
            is_correct = str(true_label) == pred_label
            print(f"曲线 {i} [{header}]: 真实={true_label}, 预测={pred_label}")
            detail_rows.append(
                {
                    "file": relpath,
                    "curve_index": i,
                    "header": header,
                    "true_label": str(true_label),
                    "predicted_label": pred_label,
                    "predicted_group": pred_label
                    if pred_label in ("None", "Mixture")
                    else "Other",
                    "correct": is_correct,
                }
            )

        all_true.extend([str(v) for v in y_true.tolist()])
        all_pred.extend([str(v) for v in y_pred.tolist()])

    all_true = np.asarray(all_true, dtype=str)
    all_pred = np.asarray(all_pred, dtype=str)

    # 完整多分类标签顺序：优先采用模型训练时的 classes_，再补上本次真实/预测中出现的标签。
    # Pipeline 会直接暴露最终分类器的 classes_；若某些环境不存在该属性，则从真实/预测标签补齐。
    model_labels = []
    if hasattr(clf, "classes_"):
        model_labels = [str(v) for v in clf.classes_]
    full_labels = list(model_labels)
    for label in list(all_true) + list(all_pred):
        if label not in full_labels:
            full_labels.append(label)

    full_report = classification_report(
        all_true, all_pred, labels=full_labels, zero_division=0
    )
    full_matrix = confusion_matrix(all_true, all_pred, labels=full_labels)

    # 当前测试集真实标签只有 None / Mixture。为了不丢失预测到其他类别的误分类，
    # 将所有其他预测类别统一折叠为 Other，形成 2 x 3 的简化矩阵。
    simplified_true_labels = ["None", "Mixture"]
    simplified_pred_labels = ["None", "Mixture", "Other"]
    grouped_pred = np.asarray(
        [p if p in ("None", "Mixture") else "Other" for p in all_pred], dtype=str
    )
    simplified_matrix = np.zeros((2, 3), dtype=int)
    for true_label, pred_group in zip(all_true, grouped_pred):
        if true_label not in simplified_true_labels:
            continue
        i = simplified_true_labels.index(true_label)
        j = simplified_pred_labels.index(pred_group)
        simplified_matrix[i, j] += 1

    correct_count = int(np.sum(all_true == all_pred))
    total_count = int(len(all_true))
    error_count = total_count - correct_count
    accuracy = correct_count / total_count if total_count else 0.0

    print("\n============================================================")
    print("全部文件汇总完整多分类报告：")
    print(full_report)
    print("全部文件汇总完整多分类混淆矩阵：")
    print(full_matrix)
    print("完整混淆矩阵标签顺序：", full_labels)

    print("\n当前测试集简化混淆矩阵：")
    print("行（真实标签）：", simplified_true_labels)
    print("列（预测分组）：", simplified_pred_labels)
    print(simplified_matrix)

    print(f"总文件数：{len(filepaths)}")
    print(f"总曲线数：{total_count}")
    print(f"正确分类数：{correct_count}")
    print(f"错误分类数：{error_count}")
    print(f"总体准确率：{accuracy:.4f} ({accuracy * 100:.2f}%)")

    output_dir = os.path.dirname(os.path.abspath(save_file)) or "."
    os.makedirs(output_dir, exist_ok=True)
    base_no_ext = os.path.splitext(save_file)[0]
    detail_csv = (
        base_no_ext + "_details_r1.csv"
        if not base_no_ext.endswith("_r1")
        else base_no_ext[:-3] + "_details_r1.csv"
    )
    summary_csv = (
        base_no_ext + "_summary_r1.csv"
        if not base_no_ext.endswith("_r1")
        else base_no_ext[:-3] + "_summary_r1.csv"
    )
    full_matrix_csv = (
        base_no_ext + "_confusion_matrix_full_r1.csv"
        if not base_no_ext.endswith("_r1")
        else base_no_ext[:-3] + "_confusion_matrix_full_r1.csv"
    )
    simple_matrix_csv = (
        base_no_ext + "_confusion_matrix_simple_r1.csv"
        if not base_no_ext.endswith("_r1")
        else base_no_ext[:-3] + "_confusion_matrix_simple_r1.csv"
    )

    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "file_count": len(filepaths),
                "curve_count": total_count,
                "correct_count": correct_count,
                "error_count": error_count,
                "accuracy": accuracy,
            }
        ]
    ).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        full_matrix,
        index=[f"True_{x}" for x in full_labels],
        columns=[f"Pred_{x}" for x in full_labels],
    ).to_csv(full_matrix_csv, encoding="utf-8-sig")

    pd.DataFrame(
        simplified_matrix,
        index=[f"True_{x}" for x in simplified_true_labels],
        columns=[f"Pred_{x}" for x in simplified_pred_labels],
    ).to_csv(simple_matrix_csv, encoding="utf-8-sig")

    with open(save_file, "w", encoding="utf-8") as f:
        f.write(f"输入目录：{input_dir}\n")
        f.write(f"递归扫描 xlsx 文件数：{len(filepaths)}\n")
        f.write(f"总曲线数：{total_count}\n")
        f.write("真实标签规则：曲线表头以0开头 -> None；否则 -> Mixture。\n")
        f.write("模型为多分类模型，因此预测成其他已训练类别时按真实误分类保留。\n\n")

        for row in detail_rows:
            f.write(
                f"{row['file']} | 曲线 {row['curve_index']} [{row['header']}] | "
                f"真实={row['true_label']} | 预测={row['predicted_label']} | "
                f"预测分组={row['predicted_group']} | 正确={row['correct']}\n"
            )

        f.write(f"\n全部文件汇总完整多分类报告：\n{full_report}")
        f.write(f"完整多分类混淆矩阵（标签顺序 {full_labels}）：\n{full_matrix}\n")
        f.write("\n当前测试集简化混淆矩阵：\n")
        f.write(f"行（真实标签）：{simplified_true_labels}\n")
        f.write(f"列（预测分组）：{simplified_pred_labels}\n")
        f.write(f"{simplified_matrix}\n")
        f.write(f"\n总文件数：{len(filepaths)}\n")
        f.write(f"总曲线数：{total_count}\n")
        f.write(f"正确分类数：{correct_count}\n")
        f.write(f"错误分类数：{error_count}\n")
        f.write(f"总体准确率：{accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    print(f"file {save_file} saved.")
    print(f"file {detail_csv} saved.")
    print(f"file {summary_csv} saved.")
    print(f"file {full_matrix_csv} saved.")
    print(f"file {simple_matrix_csv} saved.")

    return all_pred, all_true, full_report, full_matrix, simplified_matrix


def train_mode(
    base_dir="data",
    method="svm",
    model_file="case1_model.joblib",
    save_file="classified_results.txt",
):
    """保持原程序训练与评估流程，并在最后保存训练好的模型。"""
    # 指定提取特征的电压区间
    ranges = [(-0.5, 0), (0.75, 0.85)]
    ranges = [(1.2, 1.5), (-0.5, 0.5), (0.75, 0.85)]
    # ranges = [(-np.inf, np.inf)]

    X, y = load_dataset(base_dir, ranges)
    # print(X, y)
    A = np.hstack([X, np.transpose([y])])

    # 指定每一列的倍数（长度 = 特征列数）
    scales = np.array([1e4, 1e6, 1e6])  # 每列乘以不同的系数

    n_features = A.shape[1] - 1  # 特征列数
    valid_scales = scales[:n_features]  # 自动截取匹配长度

    # 对特征部分按列缩放
    A_scaled = A.copy().astype(object)  # 保留不同类型列的兼容性
    A_scaled[:, :-1] = np.round(A[:, :-1].astype(float) * valid_scales, 1)

    A = A_scaled

    for label in np.unique(A[:, -1]):
        subset = A[A[:, -1] == label]  # 取出该标签对应的行
        print(f"\nLabel {label}:")
        print(subset[:10])  # 打印前10个（若不足10个就打印全部）
    # print(A)

    report, c_matrix = train_and_evaluate(X, y, method=method)

    # # 处理离群值！
    X = preprocess_outliers(X, y)
    A = np.hstack([X, np.transpose([y])])
    # 指定每一列的倍数（长度 = 特征列数）
    scales = np.array([1e4, 1e6, 1e8])  # 每列乘以不同的系数

    n_features = A.shape[1] - 1  # 特征列数
    valid_scales = scales[:n_features]  # 自动截取匹配长度

    # 对特征部分按列缩放
    A_scaled = A.copy().astype(object)  # 保留不同类型列的兼容性
    A_scaled[:, :-1] = np.round(A[:, :-1].astype(float) * valid_scales, 1)
    A = A_scaled
    for label in np.unique(A[:, -1]):
        subset = A[A[:, -1] == label]  # 取出该标签对应的行
        print(f"\nLabel {label}:")
        print(subset[:10])  # 打印前10个（若不足10个就打印全部）
    # print(A)

    report, c_matrix = train_and_evaluate(X, y, method=method)

    # 使用原程序最终处理后的全部训练数据拟合一次，并保存模型
    save_trained_model(X, y, ranges, method=method, model_file=model_file)

    # # 保存分类结果
    with open(save_file, "w", encoding="utf-8") as f:
        f.write(f"分类报告：\n{report}")
        f.write(f"混淆矩阵：\n{c_matrix}")
        print(f"file {save_file} saved.")

        # A.shape = (N, 4)，前3列是坐标，最后1列是标签
        # 示例：
        # A = np.random.rand(100, 4)
        # A[:, -1] = np.random.randint(0, 3, size=100)

        # 1️⃣ 创建3D绘图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # 颜色列表（可根据类别数量调整）
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(A[:, -1]))))

        # 2️⃣ 遍历标签分组
        for i, label in enumerate(np.unique(A[:, -1])):
            subset = A[A[:, -1] == label]
            x, y_plot, z = subset[:, 0], subset[:, 1], subset[:, 2]
            ax.scatter(
                x, y_plot, z, color=colors[i], label=f"Label {label}", s=30, alpha=0.8
            )

        # 3️⃣ 美化图形
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(title="Classes")
        ax.set_title("3D Scatter Plot by Label")
        plt.tight_layout()

        # 4️⃣ 保存图像
        plt.savefig("3d_scatter_by_label.png", dpi=300)
        plt.show()

        # 5️⃣ 保存数据为 CSV（方便外部绘图）
        df = pd.DataFrame(A, columns=["x", "y", "z", "label"])
        df.to_csv("3d_points_with_label.csv", index=False)
        print("✅ 图像已保存为 3d_scatter_by_label.png")
        print("✅ 数据已保存为 3d_points_with_label.csv")


def main():
    parser = argparse.ArgumentParser(description="case1训练/分类程序")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train：训练并保存模型；predict：加载模型直接分类",
    )
    parser.add_argument(
        "--input", default=None, help="predict模式下要分类的单个xlsx文件"
    )
    parser.add_argument(
        "--input_dir", default=None, help="predict模式下递归扫描其中所有xlsx文件的目录"
    )
    parser.add_argument(
        "--data", default="data", help="train模式下训练数据目录，默认data"
    )
    parser.add_argument(
        "--model",
        default="case1_model.joblib",
        help="模型保存/读取文件，默认case1_model.joblib",
    )
    parser.add_argument(
        "--method", choices=["svm", "rf"], default="svm", help="分类方法，默认svm"
    )
    parser.add_argument(
        "--output",
        default="classified_results_r1.txt",
        help="结果输出文件，默认classified_results_r1.txt",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_mode(
            base_dir=args.data,
            method=args.method,
            model_file=args.model,
            save_file=args.output,
        )
    else:
        if args.input is not None and args.input_dir is not None:
            parser.error("predict模式下 --input 和 --input_dir 只能指定一个")
        if args.input_dir is not None:
            predict_directory(
                input_dir=args.input_dir,
                model_file=args.model,
                save_file=args.output,
            )
        elif args.input is not None:
            predict_file(
                input_file=args.input,
                model_file=args.model,
                save_file=args.output,
            )
        else:
            parser.error(
                "predict模式必须通过 --input 指定一个xlsx文件，或通过 --input_dir 指定目录"
            )


if __name__ == "__main__":
    main()
