import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os
import scipy.sparse
import re


# ----------------------
# 前置核心筛选函数
# ----------------------
def filter_out_low_quality_data(adata, data_type="spatial"):
    """
    统一过滤低质量数据（仅执行最终筛选步骤）
    参数:
        adata: 已初始处理的AnnData对象
        data_type: 数据类型，"spatial"（空间）或 "single_cell"（单细胞）
    返回:
        最终过滤后的AnnData对象
    """
    print(f"开始对{data_type}数据执行最终低质过滤...")
    print(f"过滤前数据形状: {adata.shape}")

    # 根据数据类型执行最终筛选
    if data_type == "spatial":
        # 空间数据最终筛选
        sc.pp.filter_cells(adata, min_genes=10)  # 每个位点至少10个基因
        sc.pp.filter_genes(adata, min_cells=10)  # 每个基因至少在10个位点表达
        print(f"空间数据最终筛选后形状: {adata.shape}")

    elif data_type == "single_cell":
        # 单细胞数据最终筛选（仅执行低质基因和细胞过滤）
        sc.pp.filter_genes(adata, min_cells=100)  # 基因至少在100个细胞中表达
        sc.pp.filter_cells(adata, min_genes=100)  # 细胞至少表达100个基因
        print(f"单细胞数据最终过滤低质基因和细胞后形状: {adata.shape}")

    else:
        raise ValueError("data_type必须为'spatial'或'single_cell'")

    return adata


# ----------------------
# 单细胞初始处理函数
# ----------------------
def process_single_cell(adata, ref_celltype_col='CellType'):
    """
    单细胞数据初始处理（基础清洗，不含最终过滤步骤）
    参数: 数据文件adata
    返回: 初始处理后的AnnData对象
    """
    # adata = adata_ref.copy() # avoid changing the content of adata_ref
    print(f"原始单细胞数据形状: {adata.shape}")

    # 处理重复基因名
    adata.var_names_make_unique()
    print("单细胞数据基因名已去重")

    # 清洗线粒体基因
    mito_genes = adata.var_names.str.startswith(('MT-', 'mt-'))
    adata = adata[:, ~mito_genes]
    print(f"移除线粒体基因后形状: {adata.shape}")

    # 保留样本量>=1的细胞类型
    if ref_celltype_col in adata.obs.columns:
        cell_type_counts = adata.obs[ref_celltype_col].value_counts()
        keep_types = cell_type_counts[cell_type_counts >= 1].index
        adata = adata[adata.obs[ref_celltype_col].isin(keep_types)]
        print(f"过滤低丰度细胞类型后形状: {adata.shape}")
    else:
        print("警告：单细胞数据obs中未找到'CellType'列，跳过细胞类型筛选")

    # 初步过滤（低阈值，为后续最终过滤做准备）
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=1)
    print(f"初始过滤空基因和细胞后形状: {adata.shape}")

    return adata


def process_single_cell_path(sc_path):
    """
    单细胞数据初始处理（基础清洗，不含最终过滤步骤）
    参数: 数据文件路径
    返回: 初始处理后的AnnData对象
    """
    adata = ad.read_h5ad(sc_path)
    print(f"原始单细胞数据形状: {adata.shape}")

    # 处理重复基因名
    adata.var_names_make_unique()
    print("单细胞数据基因名已去重")

    # 清洗线粒体基因
    mito_genes = adata.var_names.str.startswith(('MT-', 'mt-'))
    adata = adata[:, ~mito_genes]
    print(f"移除线粒体基因后形状: {adata.shape}")

    # 保留样本量>1的细胞类型
    if 'CellType' in adata.obs.columns:
        cell_type_counts = adata.obs['CellType'].value_counts()
        keep_types = cell_type_counts[cell_type_counts > 1].index
        adata = adata[adata.obs['CellType'].isin(keep_types)]
        print(f"过滤低丰度细胞类型后形状: {adata.shape}")
    else:
        print("警告：单细胞数据obs中未找到'CellType'列，跳过细胞类型筛选")

    # 初步过滤（低阈值，为后续最终过滤做准备）
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=1)
    print(f"初始过滤空基因和细胞后形状: {adata.shape}")

    return adata


# 辅助函数：清理文件名中的特殊字符
def sanitize_filename(name):
    """替换文件名中的特殊字符，避免路径错误"""
    return re.sub(r'[\\/*?:"<>|/]', '_', name)


# 辅助函数：归一化表达矩阵
def normalize_expression_matrix(matrix, target_sum):
    """对表达矩阵进行归一化处理"""
    if isinstance(matrix, pd.DataFrame):
        mat = matrix.values.copy()
        index = matrix.index
        columns = matrix.columns
    else:
        mat = matrix.copy()
        index = None
        columns = None

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    normalized_mat = mat * (target_sum / row_sums)

    if isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(normalized_mat, index=index, columns=columns)
    return normalized_mat


# ----------------------
# 单细胞数据处理函数
# ----------------------
def normalize_single_cell(adata):
    """对筛选高变基因后的单细胞数据进行归一化"""
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1)
    print("单细胞数据归一化完成 (target_sum=1)")
    return adata_norm


# ----------------------
# 空间转录组数据处理函数
# ----------------------
def process_spatial_data(adata, common_genes, hvgs):
    """处理空间转录组数据：筛选基因+归一化+log转换"""
    # adata = ad.read_h5ad(st_path)
    print(f"原始空间转录组数据形状: {adata.shape}")

    adata.var_names_make_unique()
    print("空间转录组数据基因名已去重")

    # 筛选共同基因
    adata = adata[:, common_genes].copy()
    print(f"筛选共同基因后形状: {adata.shape}")

    # 从共同基因中筛选高变基因
    adata = adata[:, hvgs].copy()
    print(f"筛选高变基因后形状: {adata.shape}")
    adata.layers['counts'] = adata.X.copy()  ## 保存count matrix
    # 归一化和log转换
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("空间转录组归一化完成 (target_sum=1e4)")

    sc.pp.log1p(adata)
    print("空间转录组log转换完成")

    # 稀疏矩阵转稠密矩阵
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
        print("空间转录组数据已转换为稠密矩阵")

    return adata


def process_spatial_data_path(st_path, common_genes, hvgs):
    """处理空间转录组数据：筛选基因+归一化+log转换"""
    adata = ad.read_h5ad(st_path)
    print(f"原始空间转录组数据形状: {adata.shape}")

    adata.var_names_make_unique()
    print("空间转录组数据基因名已去重")

    # 筛选共同基因
    adata = adata[:, common_genes].copy()
    print(f"筛选共同基因后形状: {adata.shape}")

    # 从共同基因中筛选高变基因
    adata = adata[:, hvgs].copy()
    print(f"筛选高变基因后形状: {adata.shape}")
    adata.layers['counts'] = adata.X.copy()  ## 保存count matrix
    # 归一化和log转换
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("空间转录组归一化完成 (target_sum=1e4)")

    sc.pp.log1p(adata)
    print("空间转录组log转换完成")

    # 稀疏矩阵转稠密矩阵
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
        print("空间转录组数据已转换为稠密矩阵")

    return adata


def get_spatial_hvgs_only(st_path, common_genes, hvgs):
    """仅筛选高变基因，不进行其他处理"""
    adata = ad.read_h5ad(st_path)
    print(f"原始空间转录组数据形状: {adata.shape}")

    adata.var_names_make_unique()

    # 筛选共同基因
    adata = adata[:, common_genes].copy()
    print(f"筛选共同基因后形状: {adata.shape}")

    # 筛选高变基因
    adata = adata[:, hvgs].copy()
    print(f"仅筛选高变基因后的空间数据形状: {adata.shape}")

    return adata


# ----------------------
# 高变基因基因筛选
# ----------------------
def select_hvgs(adata_ref, celltype_col, num_per_group=200):
    adata_ref_log = adata_ref.copy()
    ### Perform log-normalization then do DEG maker identification
    sc.pp.log1p(adata_ref_log)
    sc.tl.rank_genes_groups(adata_ref_log, groupby=celltype_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref_log.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes


# def select_hvgs(adata_ref, celltype_col, num_per_group=200):
#     """筛选高变基因

#     Parameters:
#     adata_sc: AnnData 单细胞数据对象
#     celltype_col: 细胞类型列的列名
#     num_per_group: 每组选择的基因数量
#     """
#     # 创建数据副本
#     adata_temp = adata_ref.copy()

#     # 对数化处理数据（如果尚未处理）
#     if not np.all(adata_temp.X.data >= 0) if hasattr(adata_temp.X, 'data') else not np.all(adata_temp.X >= 0):
#         print("数据可能已经对数化或标准化，跳过对数化步骤")
#     else:
#         sc.pp.log1p(adata_temp)

#     # 使用更高效的方法进行差异表达分析
#     # 设置use_raw=False以避免使用原始数据
#     sc.tl.rank_genes_groups(
#         adata_temp,
#         groupby=celltype_col,
#         method="t-test",
#         key_added="ttest",
#         use_raw=False,
#         n_genes=num_per_group  # 限制每个组的基因数量
#     )

#     # 更高效地提取marker基因
#     marker_genes = set()
#     for group in adata_temp.uns['ttest']['names'].dtype.names:
#         # 直接获取每个组的前num_per_group个基因
#         group_genes = adata_temp.uns['ttest']['names'][group][:num_per_group]
#         marker_genes.update(group_genes)

#     genes = sorted(list(marker_genes))

#     print(f"筛选出 {len(genes)} 个高变基因")
#     return genes

# ----------------------
# 计算细胞类型平均表达
# ----------------------
def calculate_celltype_averages(adata_sc_norm, celltype_col, sample_id_col, target_sum, normalize):
    """计算细胞类型平均表达并可选归一化"""
    cell_types = adata_sc_norm.obs[celltype_col].unique()
    num_genes = adata_sc_norm.shape[1]
    avg_matrix = np.zeros((len(cell_types), num_genes), dtype=np.float32)
    sam_ids = adata_sc_norm.obs[sample_id_col].unique()
    num_sections = len(sam_ids)
    for sid in sam_ids:
        mask = adata_sc_norm.obs[sample_id_col] == sid
        adata_s = adata_sc_norm[mask]
        cell_types_s = adata_s.obs[celltype_col].unique()
        for i, cell_type in enumerate(cell_types):
            if cell_type in cell_types_s:
                mask = adata_s.obs[celltype_col] == cell_type
                cell_subset = adata_s[mask]  # 等价于adata_sc_norm[mask,:]

                gene_sums = cell_subset.X.sum(axis=0)
                num_cells = cell_subset.shape[0]
                gene_avg = gene_sums / num_cells if num_cells > 0 else np.zeros(num_genes)
                # 调试：打印形状信息

                if gene_avg.ndim > 1:
                    gene_avg = gene_avg.flatten()
                # print(f"gene_avg shape: {gene_avg.shape}")
                # print(f"avg_matrix[i, :] shape: {avg_matrix[i, :].shape}")
                avg_matrix[i:(i + 1), :] += gene_avg.flatten()

        print(f"计算section {sid} 的平均表达: {adata_s.shape[0]}个细胞, 基因数: {adata_s.shape[1]}")
    avg_matrix = avg_matrix / num_sections
    avg_df = pd.DataFrame(
        avg_matrix,
        index=cell_types,
        columns=adata_sc_norm.var_names
    )

    if normalize:
        avg_df = normalize_expression_matrix(avg_df, target_sum)
        print(f"细胞类型平均表达已归一化 (target_sum={target_sum})")

    return avg_df


# ----------------------
def calculate_celltype_averages2(adata_sc_norm, celltype_col, target_sum, normalize):
    """计算细胞类型平均表达并可选归一化"""
    cell_types = adata_sc_norm.obs[celltype_col].unique()
    num_genes = adata_sc_norm.shape[1]
    avg_matrix = np.zeros((len(cell_types), num_genes), dtype=np.float32)

    for i, cell_type in enumerate(cell_types):
        mask = adata_sc_norm.obs[celltype_col] == cell_type
        cell_subset = adata_sc_norm[mask]

        gene_sums = cell_subset.X.sum(axis=0)
        num_cells = cell_subset.shape[0]
        gene_avg = gene_sums / num_cells if num_cells > 0 else np.zeros(num_genes)

        if gene_avg.ndim > 1:
            gene_avg = gene_avg.flatten()
        avg_matrix[i, :] = gene_avg

        print(f"计算 {cell_type} 的平均表达: {num_cells}个细胞, 基因数: {num_genes}")

    avg_df = pd.DataFrame(
        avg_matrix,
        index=cell_types,
        columns=adata_sc_norm.var_names
    )

    if normalize:
        avg_df = normalize_expression_matrix(avg_df, target_sum)
        print(f"细胞类型平均表达已归一化 (target_sum={target_sum})")

    return avg_df
