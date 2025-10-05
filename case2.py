import glob
import os
import re
import itertools
from functools import reduce
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from log import log


# 1️⃣ 处理列名
def clean_col(col):
    """
    清洗列名：
    1️⃣ 提取列名中的数字部分（例如 '05ul' → '5', '0.1ul' → '0.1'）
    2️⃣ 若无数字部分，则保留原列名
    3️⃣ 若以0开头，则将列名改为 '0'
    4️⃣ 若是纯数字，则去除前导0
    """
    col = str(col)

    # 如果是以0开头，则直接改为 '0'
    if col.startswith('0') and not col.startswith('0.'):
        return '0'

    # 提取数字（支持小数）
    nums = re.findall(r'\d+\.?\d*', col)
    # print(nums)

    if len(nums) == 1:
        num = nums[0]  # 提取第一个数字
        try:
            return str(float(num)) if '.' in num else str(int(num))
        except ValueError:
            return num
    else:
        # 如果没有数字部分，则保留原样
        return col


# 2️⃣ 按照“实际数值大小”排序列（仅对能转为数字的列排序）
def sort_key(c):
    try:
        return int(c)
    except ValueError:
        # 如果不能转成数字，就排在最后
        return float('-1')


def extract_data_from_file(filepath):
    # ranges = [(-np.inf, np.inf)]

    """从一个xlsx文件提取特征向量（区间内所有曲线矩阵的均值和标准差）"""
    log.info(f"loading file: {filepath}")
    df = pd.read_excel(filepath)
    # 删除列名中含下划线的列
    df = df[[c for c in df.columns if '_' not in str(c)]]
    # 处理df的列
    df.columns = [clean_col(c) for c in df.columns]
    # log.info(df.columns)
    # 按照实际数值大小排序
    df = df[sorted(df.columns, key=sort_key)]
    # log.debug(df.columns)

    # 横坐标 (Voltage)
    # voltage = df.iloc[:, 0].values.ravel()   # 保证一维
    # currents = df.iloc[:, 1:].values        # 其他列为电流矩阵 (n点 × m曲线)

    return df


def load_dataset(base_dir):
    """读取所有文件"""
    dfs = []
    for filepath in glob.glob(os.path.join(base_dir, "*.xlsx")):
        df = extract_data_from_file(filepath)
        dfs.append(df)

    # 1️⃣ 获取公共列名交集
    common_cols = list(reduce(lambda x, y: x & y, [set(df.columns) for df in dfs]))
    log.debug(f"公共列 ({len(common_cols)} 个)：{common_cols}")

    if len(common_cols) < 10:
        # 公共列小于10列，寻找列数最多的组合
        result = find_best_common_columns_combination(dfs, n_min=3)

        print("最佳组合索引：", result["best_combo_indices"])
        print("组合大小：", result["best_combo_size"])
        print("公共列数：", result["common_col_count"])
        print("公共列名：", result["common_cols"])
        # 取出这些 df
        best_dfs = [dfs[i] for i in result["best_combo_indices"]]
        # 重新覆盖公共列
        common_cols = list(reduce(lambda x, y: x & y, [set(df.columns) for df in best_dfs]))
        log.warning(f"新的公共列 ({len(common_cols)} 个)：{common_cols}")
        # 重新覆盖dfs
        dfs = best_dfs

    # 2️⃣ 所有df仅保留公共列
    dfs = [df[common_cols] for df in dfs]

    # 按照表头具体数值大小排序
    dfs = [df[sorted(df.columns, key=sort_key)] for df in dfs]

    # log.info(dfs, len(dfs))

    return dfs


def find_best_common_columns_combination(dfs, n_min=3):
    """
    从多个 DataFrame 中，找出所有大小 >= n_min 的组合，
    并返回公共列数最多的组合及其公共列名。

    参数：
        dfs: List[pd.DataFrame]
        n_min: int, 最小组合大小（例如3表示只考虑3个及以上的组合）

    返回：
        result: dict 包含最佳组合的信息
    """
    m = len(dfs)
    best_combo = None
    best_common_cols = set()
    best_n = 0

    # 将每个 df 的列集合保存
    col_sets = [set(df.columns) for df in dfs]

    for n in range(n_min, m + 1):
        for combo in itertools.combinations(range(m), n):
            common_cols = set.intersection(*(col_sets[i] for i in combo))
            if len(common_cols) > len(best_common_cols):
                best_common_cols = common_cols
                best_combo = combo
                best_n = n

    result = {
        "best_combo_indices": best_combo,
        "best_combo_size": best_n,
        "common_col_count": len(best_common_cols),
        "common_cols": sorted(list(best_common_cols)),
    }
    return result


def build_feature_target_from_dfs(
        dfs: List[pd.DataFrame],
        train_ratio: float = 0.9,
        include_voltage: bool = True,
        include_baseline: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    将 dfs（list of DataFrame）转换为监督学习的 X, y。
    假设：每个 df 的列顺序为 [voltage, baseline, cur_1, cur_2, ..., cur_n]
    train_ratio 表示用前 train_ratio 的“电流列”作为输入，其余作为输出。

    返回：
      X: shape (n_samples, n_features)
      y: shape (n_samples, n_targets)
      meta: dict 包含 train_cols, test_cols, train_n, test_n, include_* 等信息
    """
    assert len(dfs) > 0, "dfs 列表不能为空"
    # 以第一个 df 为标准，检查列数一致性（可根据需要做更严格的校验）
    n_cols = dfs[0].shape[1]
    assert n_cols >= 3, "每个 df 至少需要 3 列（voltage, baseline, >=1 current 列）"

    # 当前电流列数量（假定每个 df 列数相同）
    n_current = n_cols - 2
    train_n = int(np.floor(n_current * train_ratio))
    train_n = max(1, train_n)  # 至少 1 列用于训练
    test_n = n_current - train_n
    if test_n < 1:
        # 强制保留至少 1 列作测试
        train_n = n_current - 1
        test_n = 1

    # 列名
    train_cols = list(dfs[0].columns[2:2 + train_n])
    test_cols = list(dfs[0].columns[2 + train_n: 2 + train_n + test_n])

    X_list = []
    y_list = []

    for df in dfs:
        # 简单校验：确保列数一致
        if df.shape[1] != n_cols:
            raise ValueError("所有 DataFrame 必须具有相同列数（第一个 df 的列数为基准）")

        vol = df.iloc[:, 0].to_numpy()  # shape (625,)
        baseline = df.iloc[:, 1].to_numpy()  # shape (625,)
        currents = df.iloc[:, 2:].to_numpy()  # shape (625, n_current)

        for i in range(currents.shape[0]):  # 对每一行（电压点）产生一个样本
            features = []
            if include_voltage:
                features.append(vol[i])
            if include_baseline:
                features.append(baseline[i])
            # 前 train_n 列作为特征
            features.extend(currents[i, :train_n].tolist())
            X_list.append(features)
            # 后 test_n 列作为多输出目标
            y_list.append(currents[i, train_n:].tolist())

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    meta = {
        'train_cols': train_cols,
        'test_cols': test_cols,
        'train_n': train_n,
        'test_n': test_n,
        'include_voltage': include_voltage,
        'include_baseline': include_baseline,
        'voltage_col': dfs[0].columns[0],
        'baseline_col': dfs[0].columns[1]
    }
    return X, y, meta


def train_and_evaluate_multioutput(
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        random_state: int = 42,
        n_estimators: int = 200,
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray]:
    """
    使用 Pipeline(imputer->scaler->RandomForest) 对 X,y 进行交叉验证评估并在全部数据上训练最终模型。
    返回：fitted_pipeline, metrics_dict, y_pred_cv（交叉验证预测，用于评估）
    """
    # Pipeline: 缺失值填充 -> 标准化 -> 随机森林回归（多输出）
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state))
    ])

    # 交叉验证预测（用于评估）
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # cross_val_predict 支持多输出回归器
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=kf, method='predict', n_jobs=-1)

    # 计算每个输出（每个被预测列）的指标
    metrics = {}
    rmse_list, mae_list, r2_list = [], [], []
    for j in range(y.shape[1]):
        rmse = mean_squared_error(y[:, j], y_pred_cv[:, j], squared=False)
        mae = mean_absolute_error(y[:, j], y_pred_cv[:, j])
        r2 = r2_score(y[:, j], y_pred_cv[:, j])
        metrics[f'output_{j}'] = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    metrics['aggregate'] = {
        'rmse_mean': float(np.mean(rmse_list)),
        'mae_mean': float(np.mean(mae_list)),
        'r2_mean': float(np.mean(r2_list))
    }

    # 在所有数据上训练最终模型
    pipeline.fit(X, y)

    return pipeline, metrics, y_pred_cv


def predict_and_attach(
        model: Pipeline,
        dfs: List[pd.DataFrame],
        meta: Dict[str, Any],
        overwrite: bool = False,
        pred_suffix: str = '_pred'
) -> List[pd.DataFrame]:
    """
    使用训练好的 model 对 dfs 中每个 df 的后 test_n 列逐行预测，
    并将预测结果以新列（列名 + pred_suffix）附加到 df 的副本中（默认不覆盖原列）。
    返回：predicted_dfs（list of DataFrame）
    """
    train_n = meta['train_n']
    predicted_dfs = []

    for df in dfs:
        df_copy = df.copy()
        vol = df_copy.iloc[:, 0].to_numpy()
        baseline = df_copy.iloc[:, 1].to_numpy()
        currents = df_copy.iloc[:, 2:].to_numpy()  # shape (625, n_current)
        rows = currents.shape[0]

        X_rows = []
        for i in range(rows):
            features = []
            if meta['include_voltage']:
                features.append(vol[i])
            if meta['include_baseline']:
                features.append(baseline[i])
            features.extend(currents[i, :train_n].tolist())
            X_rows.append(features)
        X_rows = np.array(X_rows, dtype=float)

        y_hat = model.predict(X_rows)  # shape (rows, test_n)

        # 把预测值写回 DataFrame（以新列或覆盖原列）
        for j, col in enumerate(meta['test_cols']):
            if overwrite:
                col_name = col
            else:
                col_name = f"{col}{pred_suffix}"
            df_copy[col_name] = y_hat[:, j]

        predicted_dfs.append(df_copy)

    return predicted_dfs


def evaluate_predictions_on_dfs(
        predicted_dfs: List[pd.DataFrame],
        original_dfs: List[pd.DataFrame],
        meta: Dict[str, Any],
        pred_suffix: str = '_pred'
) -> List[Dict[str, Any]]:
    """
    逐个 df 计算预测列与真实列之间的 RMSE/MAE/R2，返回每个 df 的字典。
    假定 predict_and_attach 使用默认行为（新增列名 = 原列 + pred_suffix）。
    """
    results = []
    for df_pred, df_true in zip(predicted_dfs, original_dfs):
        per_col = {}
        rmse_list = []
        mae_list = []
        r2_list = []
        for col in meta['test_cols']:
            pred_col = f"{col}{pred_suffix}"
            if pred_col not in df_pred.columns:
                raise KeyError(f"预测列 {pred_col} 不存在，请检查 predict_and_attach 的 overwrite/pred_suffix 参数")
            y_true = df_true[col].to_numpy()
            y_pred = df_pred[pred_col].to_numpy()
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            per_col[col] = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)
        per_col['aggregate'] = {
            'rmse_mean': float(np.mean(rmse_list)),
            'mae_mean': float(np.mean(mae_list)),
            'r2_mean': float(np.mean(r2_list))
        }
        results.append(per_col)
    return results


# ================================================================
# 6️⃣ 绘图函数
# ================================================================
def plot_predictions_for_dfs(
    predicted_dfs: List[pd.DataFrame],
    meta: Dict[str, Any],
    save_dir: str = "plots",
    pred_suffix: str = "_pred"
) -> None:
    """
    为每个 DataFrame 生成预测 vs 实际 的电流曲线图并保存。
    - 每个 df 生成一个子文件夹。
    - 每个被预测列单独成图。
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, df_pred in enumerate(predicted_dfs):
        voltage = df_pred[meta["voltage_col"]].to_numpy()
        df_dir = os.path.join(save_dir, f"df_{i+1}")
        os.makedirs(df_dir, exist_ok=True)

        for col in meta["test_cols"]:
            pred_col = f"{col}{pred_suffix}"
            if pred_col not in df_pred.columns:
                continue
            y_true = df_pred[col].to_numpy()
            y_pred = df_pred[pred_col].to_numpy()

            plt.figure(figsize=(8, 5))
            plt.plot(voltage, y_true, label="True", lw=2)
            plt.plot(voltage, y_pred, "--", label="Predicted", lw=2)
            plt.xlabel("Voltage")
            plt.ylabel("Current")
            plt.title(f"DF {i+1} — {col}: True vs Predicted")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(df_dir, f"{col}_comparison.png")
            plt.savefig(save_path, dpi=300)
            plt.close()


# ================================================================
# 7️⃣ 保存评估结果函数
# ================================================================
def save_experiment_results(
    metrics_list: List[Dict[str, Any]],
    predicted_dfs: List[pd.DataFrame],
    save_root: str = "experiment_results",
    save_preds: bool = True
) -> str:
    """
    保存预测评估指标和预测后的 DataFrame。
    返回保存目录路径。
    """
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_root, f"predicted_results")
    os.makedirs(save_dir, exist_ok=True)

    # 保存 metrics 汇总
    all_metrics = []
    for i, m in enumerate(metrics_list):
        agg = m["aggregate"]
        all_metrics.append({
            "DF_index": i + 1,
            "RMSE_mean": agg["rmse_mean"],
            "MAE_mean": agg["mae_mean"],
            "R2_mean": agg["r2_mean"],
        })
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(save_dir, "summary_metrics.csv"), index=False)

    # 保存每个 df 的详细列指标
    for i, m in enumerate(metrics_list):
        df_metrics = pd.DataFrame(m).T
        df_metrics.to_csv(os.path.join(save_dir, f"df_{i+1}_metrics.csv"))

    # 可选保存预测后的 DataFrame
    if save_preds:
        pred_dir = os.path.join(save_dir, "predicted_dfs")
        os.makedirs(pred_dir, exist_ok=True)
        for i, df in enumerate(predicted_dfs):
            df.to_csv(os.path.join(pred_dir, f"df_{i+1}_pred.csv"), index=False)

    return save_dir


# ================================================================
# 8️⃣ 全流程主控函数
# ================================================================
def full_experiment_pipeline(
    dfs: List[pd.DataFrame],
    train_ratio: float = 0.9,
    include_voltage: bool = True,
    include_baseline: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
    n_estimators: int = 200,
    save_root: str = "experiment_results"
):
    """
    全流程封装：
      1. 构建特征与标签
      2. 训练 + 交叉验证评估
      3. 预测并附加
      4. 独立评估
      5. 绘图与保存结果
    """
    os.makedirs(save_root, exist_ok=True)

    log.info("🧩 构建特征与标签...")
    X, y, meta = build_feature_target_from_dfs(
        dfs,
        train_ratio=train_ratio,
        include_voltage=include_voltage,
        include_baseline=include_baseline
    )

    log.info("⚙️ 训练模型 + 交叉验证...")
    model, metrics_cv, y_pred_cv = train_and_evaluate_multioutput(
        X, y, cv_folds=cv_folds, random_state=random_state, n_estimators=n_estimators
    )
    log.info("✅ 模型训练完成，交叉验证平均 RMSE =", metrics_cv["aggregate"]["rmse_mean"])

    log.info("🔮 预测所有 DataFrame ...")
    predicted_dfs = predict_and_attach(model, dfs, meta)

    log.info("📏 评估预测性能 ...")
    metrics_list = evaluate_predictions_on_dfs(predicted_dfs, dfs, meta)

    rmse_total = 0
    mae_total = 0
    r2_total = 0
    for res in metrics_list:
        rmse_total += res["aggregate"]["rmse_mean"]
        mae_total += res["aggregate"]["mae_mean"]
        r2_total += res["aggregate"]["r2_mean"]

    total_test = len(metrics_list)
    mean_metrics = f"rmse_mean: {rmse_total/total_test}\n" \
                   f"mae_mean: {mae_total / total_test}\n" \
                   f"r2_mean: {r2_total / total_test}\n"

    log.debug(mean_metrics)
    # 保存全部平均指标
    with open(f"{save_root}/mean_metrics.txt", "w", encoding="utf-8") as f:
        f.write(mean_metrics)

    log.info("🎨 绘制并保存预测曲线 ...")
    plot_predictions_for_dfs(predicted_dfs, meta, save_dir=os.path.join(save_root, "plots"))

    log.info("💾 保存实验结果 ...")
    save_dir = save_experiment_results(metrics_list, predicted_dfs, save_root=save_root)

    log.debug(f"✅ 全部实验完成，结果保存在：{save_dir}")
    return model, metrics_list, save_dir


if __name__ == "__main__":
    # 开始计时
    log.start_timer()

    base_dir = "data"  # 子文件夹所在目录

    # 给定已经运行过的文件夹数量，需要全部运行则设置为0，否则将跳过前 skip_count 个文件夹的运行
    # skip_count = 0
    skip_count = 0

    all_folders = os.listdir(base_dir)

    # 循环处理文件夹下的文件夹
    cnt = 0
    for folder_name in all_folders:
        cnt += 1
        if cnt <= skip_count:
            log.warning(f"folder {folder_name} skipped!")
            continue
        current_folder = os.path.join(base_dir, folder_name)
        dfs = load_dataset(current_folder)

        # dfs 是 10 个 625x19 的 DataFrame 列表
        model, metrics_list, save_dir = full_experiment_pipeline(
            dfs,
            train_ratio=0.8,
            include_voltage=True,
            include_baseline=True,
            cv_folds=5,
            n_estimators=300,
            save_root=f"predicted_results/{folder_name}"
        )

    # 结束计时
    log.elapsed()
