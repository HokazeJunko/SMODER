import anndata as ad
import scanpy as sc
import numpy as np
import os
import scipy.sparse
from scipy.stats import gmean  # 用于CLR标准化计算几何均值
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF所需依赖
import scipy.sparse as sparse  # 明确稀疏矩阵依赖


# ----------------------
# CLR标准化核心函数
# ----------------------
def clr_normalization(matrix):
    """执行中心对数比（CLR）标准化：clr(x) = ln(x_i / 几何均值(x))"""
    mat = matrix.copy()
    mat = np.maximum(mat, 1e-10)  # 替换0和负值为极小值，避免log(0)错误
    row_gmeans = gmean(mat, axis=1, keepdims=True)  # 计算每行几何均值
    return np.log(mat / row_gmeans)  # CLR转换


# ----------------------
# 空间蛋白数据处理主函数（保留空间坐标）
# ----------------------
def process_spatial_adt_data(adata, min_genes=20, min_cells=100):
    """读取数据→过滤低质量位点和蛋白→CLR标准化→保留空间坐标"""
    # 读取原始数据（包含空间坐标）
    print(f"原始空间蛋白数据形状: {adata.shape}")

    # 验证并保留空间坐标信息
    if 'spatial' not in adata.obsm:
        print("警告：原始数据中未找到'spatial'空间坐标信息！")
    else:
        print(f"已检测到空间坐标，形状: {adata.obsm['spatial'].shape}（将保留在处理后数据中）")

    # 处理重复蛋白名
    adata.var_names_make_unique()
    print("空间蛋白名已去重")

    # 过滤低质量位点（每位点至少表达20种蛋白）
    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"过滤后（每位点≥20种蛋白）形状: {adata.shape}")
    # 过滤后空间坐标自动随细胞筛选同步保留

    # 过滤低丰度蛋白（每蛋白至少在100个位点表达）
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"过滤后（每蛋白≥100个位点）形状: {adata.shape}")
    adata.layers['counts'] = adata.X.copy()  ## save raw data
    # 转换为稠密矩阵（确保CLR计算兼容性）
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
        print("空间蛋白数据已转换为稠密矩阵")
    # 执行CLR标准化（仅处理表达矩阵，不影响空间坐标）
    adata.X = clr_normalization(adata.X)
    # sc.pp.clr(adata) ###
    print("空间蛋白数据已完成CLR标准化")

    # 再次确认空间坐标保留
    if 'spatial' in adata.obsm:
        print(f"处理后数据中空间坐标保留状态: 已保留（形状: {adata.obsm['spatial'].shape}）")

    return adata


def process_spatial_adt_data_path(st_adt_path):
    """通过文件路径读取数据→过滤低质量位点和蛋白→CLR标准化→保留空间坐标"""
    # 读取原始数据（包含空间坐标）
    adata = ad.read_h5ad(st_adt_path)
    print(f"原始空间蛋白数据形状: {adata.shape}")

    # 验证并保留空间坐标信息
    if 'spatial' not in adata.obsm:
        print("警告：原始数据中未找到'spatial'空间坐标信息！")
    else:
        print(f"已检测到空间坐标，形状: {adata.obsm['spatial'].shape}（将保留在处理后数据中）")

    # 处理重复蛋白名
    adata.var_names_make_unique()
    print("空间蛋白名已去重")

    # 过滤低质量位点（每位点至少表达20种蛋白）
    sc.pp.filter_cells(adata, min_genes=20)
    print(f"过滤后（每位点≥20种蛋白）形状: {adata.shape}")
    # 过滤后空间坐标自动随细胞筛选同步保留

    # 过滤低丰度蛋白（每蛋白至少在100个位点表达）
    sc.pp.filter_genes(adata, min_cells=100)
    print(f"过滤后（每蛋白≥100个位点）形状: {adata.shape}")
    adata.layers['counts'] = adata.X.copy()  ## save raw data
    # 转换为稠密矩阵（确保CLR计算兼容性）
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
        print("空间蛋白数据已转换为稠密矩阵")
    # 执行CLR标准化（仅处理表达矩阵，不影响空间坐标）
    adata.X = clr_normalization(adata.X)
    # sc.pp.clr(adata) ###
    print("空间蛋白数据已完成CLR标准化")

    # 再次确认空间坐标保留
    if 'spatial' in adata.obsm:
        print(f"处理后数据中空间坐标保留状态: 已保留（形状: {adata.obsm['spatial'].shape}）")

    return adata


# ----------------------
# Peak数据处理函数
# ----------------------
def process_peak_data(adata, n_top_features=20000):
    """Process peak data with TF-IDF normalization and top-feature selection."""
    print("=" * 50)
    print("开始处理Peak数据...")
    print(f"原始Peak数据形状: {adata.shape}")

    # 1. 数据预处理：确保输入为稀疏矩阵（符合TF-IDF要求）
    if not scipy.sparse.issparse(adata.X):
        print("将Peak数据转换为CSR稀疏矩阵（符合TF-IDF处理要求）...")
        adata.X = sparse.csr_matrix(adata.X)
    else:
        # 转换为CSR格式，提升后续计算效率
        adata.X = adata.X.tocsr()

    # 2. TF-IDF归一化（等效于 Signac RunTFIDF）
    print("Running TF-IDF normalization...")
    # 使用scikit-learn的TfidfTransformer，与Signac默认行为一致（l2范数、开启idf、平滑idf）
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    # 拟合并转换，转置以符合sklearn格式要求（sklearn默认样本在列，基因/peak在行）
    adata.X = sparse.csr_matrix(tfidf.fit_transform(adata.X.T).T)
    print("TF-IDF归一化完成，已保留l2范数标准化（每个细胞向量长度为1）")

    # 3. 选择高变特征/peaks（等效于 Signac FindTopFeatures, min.cutoff='q5'）
    print(f"Selecting top {n_top_features} variable features...")
    # 计算每个peak的离散度（方差），用于筛选高变特征
    if adata.n_vars > n_top_features:
        # 计算每个peak的均值
        mean_val = np.array(adata.X.mean(axis=0)).flatten()
        # 计算每个peak的方差（E[X²] - (E[X])²）
        var_val = np.array(adata.X.power(2).mean(axis=0)).flatten() - np.square(mean_val)
        # 获取方差最大的n_top_features个peak的索引
        top_feature_idx = np.argsort(var_val)[-n_top_features:]
        # 根据索引筛选peaks，生成新的AnnData对象（避免修改原数据）
        adata = adata[:, top_feature_idx].copy()
        print(f"  已筛选出 {n_top_features} 个高变Peak，当前数据形状: {adata.shape}")
    else:
        # 如果peaks数少于n_top_features，则选择全部
        print(f"  Peak总数（{adata.n_vars}）小于目标筛选数（{n_top_features}），保留全部Peak")
    print("Peak数据处理完成！")
    print("=" * 50)

    return adata


def process_peak_data_path(peak_h5ad_path, n_top_features=20000):
    """Load peak data from file, then apply TF-IDF normalization and top-feature selection."""
    # 第一步：从指定路径读取h5ad格式的peak数据
    print("=" * 50)
    print(f"从路径读取Peak数据：{peak_h5ad_path}")
    try:
        adata = ad.read_h5ad(peak_h5ad_path)
        print(f"成功读取Peak数据，原始形状: {adata.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：未找到指定路径下的文件 → {peak_h5ad_path}")
    except Exception as e:
        raise Exception(f"错误：读取Peak数据失败，异常信息 → {str(e)}")

    # 第二步：调用已实现的process_peak_data完成核心处理（复用逻辑，减少冗余）
    adata = process_peak_data(adata, n_top_features=n_top_features)

    # 第三步：返回处理完成的AnnData对象
    return adata