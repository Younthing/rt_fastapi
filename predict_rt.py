import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
import warnings
from dataclasses import dataclass
from typing import Optional

# 忽略可能的警告信息
warnings.filterwarnings("ignore")


def load_model(model_dir, n_tasks=1, num_layers=4, graph_feat_size=128):
    """
    加载AttentiveFP模型

    参数:
        model_dir: 模型目录
        n_tasks: 任务数量
        num_layers: 模型层数
        graph_feat_size: 图特征大小

    返回:
        已加载的模型
    """
    print(f"加载模型: {model_dir}")
    model = dc.models.AttentiveFPModel(
        model_dir=model_dir,
        n_tasks=n_tasks,
        num_layers=num_layers,
        graph_feat_size=graph_feat_size,
    )
    model.restore()
    return model


def load_data(input_file, smiles_column):
    """
    加载数据并处理缺失的SMILES

    参数:
        input_file: 输入CSV文件路径
        smiles_column: SMILES所在的列名

    返回:
        清理后的DataFrame
    """
    print(f"读取输入文件: {input_file}")
    df = pd.read_csv(input_file)

    # 记录原始数据条数
    original_count = len(df)
    print(f"原始数据: {original_count} 条记录")

    # 删除SMILES为空的行
    df = df.dropna(subset=[smiles_column])
    df = df[df[smiles_column].str.strip() != ""]

    # 记录清理后的数据条数
    cleaned_count = len(df)
    if cleaned_count < original_count:
        print(f"删除了 {original_count - cleaned_count} 条SMILES为空的记录")
    print(f"清理后数据: {cleaned_count} 条记录")

    return df


def setup_featurizer(smiles_column):
    """
    设置特征化器和加载器

    参数:
        smiles_column: SMILES所在的列名

    返回:
        CSVLoader实例
    """
    tasks = ["RT"]  # 任务名称，仅用于特征化过程
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    # 创建CSVLoader实例
    loader = dc.data.CSVLoader(
        tasks=tasks, feature_field=smiles_column, featurizer=featurizer
    )

    return loader


def featurize_molecule(smiles, loader):
    """
    特征化单个分子

    参数:
        smiles: SMILES字符串
        loader: CSVLoader实例

    返回:
        特征化后的数据和状态（成功/失败）
    """
    # 检查SMILES是否有效
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False, "无效SMILES"

    # 创建单个分子的DataFrame
    single_mol_df = pd.DataFrame(
        {
            loader.feature_field: [smiles],
            "RT": [0.0],  # 占位符值，预测模式下不会使用
        }
    )

    # 特征化处理
    try:
        X, valid_inds = loader._featurize_shard(single_mol_df)

        # 检查特征化是否成功
        if len(valid_inds) == 0 or not any(valid_inds):
            return None, False, "特征化失败"

        return X, True, None

    except Exception as e:
        return None, False, str(e)


def predict_molecule_rt(model, X, smiles, min_rt=1.0, max_rt=6.0):
    """
    预测单个分子的保留时间

    参数:
        model: 已加载的模型
        X: 特征化后的数据
        smiles: SMILES字符串
        min_rt: 保留时间下限
        max_rt: 保留时间上限

    返回:
        预测的保留时间
    """
    # 创建数据集对象
    ids = np.array([smiles])
    dataset = dc.data.NumpyDataset(X=X, ids=ids)

    # 预测
    prediction = model.predict(dataset)

    # 将预测值限制在指定范围内
    bounded_prediction = np.clip(prediction, min_rt, max_rt)[0][0]

    return bounded_prediction


def process_data(
    df,
    model,
    loader,
    smiles_column,
    output_column,
    min_rt=1.0,
    max_rt=6.0,
    batch_size=10,
):
    """
    处理数据并预测保留时间

    参数:
        df: 输入数据DataFrame
        model: 已加载的模型
        loader: CSVLoader实例
        smiles_column: SMILES所在的列名
        output_column: 输出保留时间的列名
        min_rt: 保留时间下限
        max_rt: 保留时间上限
        batch_size: 报告进度的批次大小

    返回:
        处理后的DataFrame和错误记录
    """
    total_compounds = len(df)
    predicted_rt_list = []
    error_records = []

    # 逐个处理每个SMILES
    for i, row in df.iterrows():
        try:
            smiles = row[smiles_column]

            # 特征化分子
            X, success, error_msg = featurize_molecule(smiles, loader)

            if not success:
                print(
                    f"处理失败 ({i + 1}/{total_compounds}): {smiles}, 原因: {error_msg}"
                )
                predicted_rt_list.append(None)
                error_records.append({"Index": i, "SMILES": smiles, "Error": error_msg})
                continue

            # 预测保留时间
            rt = predict_molecule_rt(model, X, smiles, min_rt, max_rt)
            predicted_rt_list.append(rt)

            # 打印进度
            if (i + 1) % batch_size == 0 or i == 0 or i == total_compounds - 1:
                print(
                    f"处理进度: {i + 1}/{total_compounds}, 当前SMILES: {smiles}, 预测RT: {rt:.4f}"
                )

        except Exception as e:
            smiles = row.get(smiles_column, "无法获取SMILES")
            print(f"处理失败 ({i + 1}/{total_compounds}): {smiles}, 错误: {str(e)}")
            predicted_rt_list.append(None)
            error_records.append({"Index": i, "SMILES": smiles, "Error": str(e)})

    # 将预测结果添加到DataFrame
    result_df = df.copy()
    result_df[output_column] = predicted_rt_list
    # 新增一列，将预测值乘以60，列名为"predict_RT(sec)"
    result_df["Predicted_RT(sec)"] = result_df[output_column] * 60

    # 统计成功率
    success_count = total_compounds - len(error_records)
    print(
        f"\n成功处理: {success_count}/{total_compounds} SMILES ({success_count / total_compounds * 100:.1f}%)"
    )

    # 如果有错误，创建错误记录DataFrame
    error_df = pd.DataFrame(error_records) if error_records else None

    return result_df, error_df


def save_results(result_df, error_df, output_file, error_file):
    """
    保存结果和错误记录

    参数:
        result_df: 结果DataFrame
        error_df: 错误记录DataFrame
        output_file: 输出文件路径
        error_file: 错误记录文件路径
    """
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")

    # 如果有错误，保存错误记录
    if error_df is not None and not error_df.empty:
        error_df.to_csv(error_file, index=False)
        print(f"处理失败: {len(error_df)} SMILES，详情已保存到 {error_file}")


def main(args):
    """
    主函数

    参数:
        args: 命令行参数
    """
    # 加载模型
    model = load_model(
        args.model_dir,
        n_tasks=args.n_tasks,
        num_layers=args.num_layers,
        graph_feat_size=args.graph_feat_size,
    )

    # 加载并清理数据
    df = load_data(args.input_file, args.smiles_column)

    # 设置特征化器
    loader = setup_featurizer(args.smiles_column)

    # 处理数据
    result_df, error_df = process_data(
        df,
        model,
        loader,
        args.smiles_column,
        args.output_column,
        min_rt=args.min_rt,
        max_rt=args.max_rt,
        batch_size=args.batch_size,
    )

    # 保存结果
    save_results(result_df, error_df, args.output_file, args.error_file)


# dataclass的好处是可以指定输入类型和验证比如指定字段为字符串等
@dataclass
class BaseModelArgs:
    """基础模型参数类，包含共同的参数设置"""

    # 文件参数
    input_file: str
    error_file: str = "results/error_smiles.csv"

    # 列名参数
    smiles_column: str = "IsomericSMILES"
    output_column: str = "Predicted_RT(min)"  # 这里改就行

    # 模型固定参数
    n_tasks: int = 1
    num_layers: int = 4
    graph_feat_size: int = 128
    batch_size: int = 10

    # 预测参数基础值
    min_rt: float = 0.0

    # 这些需要子类覆盖
    output_file: Optional[str] = None
    model_dir: Optional[str] = None
    max_rt: Optional[float] = None


@dataclass
class Args_RT6min(BaseModelArgs):
    """6分钟模型参数"""

    output_file: str = "results/predicted_RT_RP6min_results.csv"
    output_column: str = "C18_RP_6min_Predicted_RT(min)"
    model_dir: str = "model/AttentiveModel_RP6min"
    max_rt: float = 6.0


@dataclass
class Args_RT12min(BaseModelArgs):
    """12分钟模型参数"""

    output_file: str = "results/predicted_RT_RP12min_results.csv"
    output_column: str = "C18_RP_12min_Predicted_RT(min)"
    model_dir: str = "model/AttentiveModel_RP12min"
    max_rt: float = 12.0


if __name__ == "__main__":
    # 使用示例
    args_6min = Args_RT6min(input_file="data/示例输入.csv")
    args_12min = Args_RT12min(input_file="data/示例输入.csv")

    # 使用参数运行主函数
    main(args_12min)  # 或 main(args_12min)
