import scanpy as sc
import numpy as np
import random  # set random seed
from sklearn.decomposition import PCA, TruncatedSVD  # LSI所需TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import warnings
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

# 导入外部模块
from smoder.preprocessing import rna as pr
from smoder.preprocessing import modality2 as pad  # ADT/Peak预处理模块
from smoder.models.gnn import *


def selectInfoGenes(Basis, sc_adata, commonGene, ct_select, ct_varname, log_FC=1.25,
                    top_n_filter=False, top_n=200):
    """
    选择信息基因的函数（新增top_n筛选选项）

    参数:
        Basis: 基础表达矩阵，行是基因，列是细胞类型
        sc_adata: scanpy的AnnData对象
        commonGene: 共同基因列表
        ct_select: 需要筛选的细胞类型列表
        ct_varname: 细胞类型在sc_adata.obs中的列名
        log_FC: 原有的log倍变化阈值，默认1.25
        top_n_filter: 是否启用top_n筛选模式，默认False（使用原有log_FC筛选）
        top_n: 每种细胞类型筛选的差异基因数量，默认2000

    返回:
        gene2: 最终筛选出的信息基因列表
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    # 第一步：筛选差异表达基因（新增top_n筛选逻辑）
    gene1_list = []
    for ict in ct_select:
        rest_columns = [col for col in Basis.columns if col != ict]
        rest = Basis[rest_columns].mean(axis=1)
        FC = np.log(Basis[ict] + 1e-06) - np.log(rest + 1e-06)

        if top_n_filter:
            # 启用top_n筛选：取FC最大的前top_n个基因（且表达量>0）
            # 先筛选表达量>0的基因
            valid_genes = Basis.index[Basis[ict] > 0]
            FC_valid = FC.loc[valid_genes]
            # 按FC降序排序，取前top_n个
            selected_genes = FC_valid.sort_values(ascending=False).head(top_n).index
        else:
            # 原有逻辑：按log_FC阈值筛选
            selected_genes = Basis.index[(FC > log_FC) & (Basis[ict] > 0)]

        gene1_list.extend(selected_genes)
        print(f"细胞类型 {ict} 筛选出 {len(selected_genes)} 个基因")

    # 去重并与commonGene取交集
    gene1 = list(set(gene1_list))
    gene1 = list(set(gene1) & set(commonGene))

    # 第二步：获取counts数据（原有逻辑不变）
    counts = sc_adata.X
    if sparse.issparse(counts):
        counts = counts.toarray()

    counts = counts.T
    counts_filtered = counts[np.isin(sc_adata.var_names, gene1), :]

    # 获取细胞类型信息
    cell_types = sc_adata.obs[ct_varname]
    ct_counts = cell_types.value_counts()
    ct_select_filtered = ct_counts[ct_counts > 1].index.tolist()

    # 第三步：计算within-group变异（原有逻辑不变）
    sd_within = pd.DataFrame(index=np.where(np.isin(sc_adata.var_names, gene1))[0])

    for ict in ct_select_filtered:
        mask = (cell_types == ict)
        temp = counts_filtered[:, mask]
        variance = np.var(temp, axis=1)
        mean_val = np.mean(temp, axis=1)
        ratio = np.divide(variance, mean_val, out=np.zeros_like(variance), where=mean_val != 0)
        sd_within[ict] = ratio

    mean_ratio = sd_within.mean(axis=1, skipna=True)
    threshold = np.quantile(mean_ratio, 0.99)
    gene2_indices = mean_ratio[mean_ratio < threshold].index
    gene2 = [sc_adata.var_names[i] for i in gene2_indices]

    return gene2

class SpaMultiDecon_two_modals:
    def __init__(self, ref_adata_dict = {'modal1': None}, smo_adata_dict = {'modal1': None, 'modal2': None},  nn_para_dict={},
                 modal2_type='adt'):
        ### 初始化参数：核心调整（第二模态仅一个维度参数 modal2_target_dim）
        if nn_para_dict == {}:
           nn_para_dict = {
               # 原始基础参数
               'epochs': 1000, 'learning_rate': 1e-3, 'device': 'cuda:0', 'hidden_dim': 64,
               'weight_loss': [1, 0.1], 'seed': 1, 'weight_spatial': 1, 'weight_consistency': 0.5,
               # 降维相关参数：仅一个参数管控第二模态最终维度
               'pca_n_components_rna': 256,  # RNA模态PCA最终维度
               'modal2_target_dim': 256,     # 第二模态（ADT/Peak）最终目标维度（仅一个输入参数）
               'lsi_random_state': 1
           }
        self.smo_adata_dict = smo_adata_dict
        self.ref_adata_dict = ref_adata_dict
        self.nn_para_dict = nn_para_dict
        self.modal2_type = modal2_type.lower()
        assert self.modal2_type in ['adt', 'peak'], "modal2_type仅支持 'adt' 或 'peak'！"

        # 标记：是否已完成特征工程（含模型输入映射）
        self.feature_engineering_done = False

    ### 属性装饰器：核心调整（第二模态仅暴露一个 target_dim 属性）
    # 原始属性（保持不变）
    @property
    def weight_loss(self):
        return self.nn_para_dict.get('weight_loss', [1, 0.001])
    @weight_loss.setter
    def weight_loss(self, value):
        self.nn_para_dict['weight_loss'] = value

    @property
    def hidden_dim(self):
        return self.nn_para_dict.get('hidden_dim', 64)
    @hidden_dim.setter
    def hidden_dim(self, value):
        self.nn_para_dict['hidden_dim'] = value

    @property
    def learning_rate(self):
        return self.nn_para_dict.get('learning_rate', 2e-3)
    @learning_rate.setter
    def learning_rate(self, value):
        self.nn_para_dict['learning_rate'] = value

    @property
    def device(self):
        return self.nn_para_dict.get('device', 'cuda:0')
    @device.setter
    def device(self, value):
        self.nn_para_dict['device'] = value

    @property
    def epochs(self):
        return self.nn_para_dict.get('epochs', 1000)
    @epochs.setter
    def epochs(self, value):
        self.nn_para_dict['epochs'] = value

    @property
    def seed(self):
        return self.nn_para_dict.get('seed', 1)
    @seed.setter
    def seed(self, value):
        self.nn_para_dict['seed'] = value

    @property
    def weight_spatial(self):
        return self.nn_para_dict.get('weight_spatial', 1e-5)
    @weight_spatial.setter
    def weight_spatial(self, value):
        self.nn_para_dict['weight_spatial'] = value

    @property
    def weight_consistency(self):
        return self.nn_para_dict.get('weight_consistency', 0.5)
    @weight_consistency.setter
    def weight_consistency(self, value):
        self.nn_para_dict['weight_consistency'] = value

    # 降维相关属性：核心简化（仅一个参数管控第二模态）
    @property
    def pca_n_components_rna(self):
        return self.nn_para_dict.get('pca_n_components_rna', 256)
    @pca_n_components_rna.setter
    def pca_n_components_rna(self, value):
        assert value > 0, "pca_n_components_rna必须大于0！"
        self.nn_para_dict['pca_n_components_rna'] = value

    @property
    def modal2_target_dim(self):
        """第二模态（ADT/Peak）最终目标维度（仅一个输入参数，统一管控）"""
        return self.nn_para_dict.get('modal2_target_dim', 256)
    @modal2_target_dim.setter
    def modal2_target_dim(self, value):
        assert value > 0, "modal2_target_dim必须大于0！"
        self.nn_para_dict['modal2_target_dim'] = value

    @property
    def lsi_random_state(self):
        return self.nn_para_dict.get('lsi_random_state', 1)
    @lsi_random_state.setter
    def lsi_random_state(self, value):
        self.nn_para_dict['lsi_random_state'] = value

    # 批量设置方法：同步简化，仅保留 modal2_target_dim
    def set_nn_para(self, **kwargs):
        valid_keys = [
            'epochs', 'learning_rate', 'device', 'hidden_dim', 'seed',
            'weight_loss', 'weight_consistency','weight_spatial',
            'pca_n_components_rna', 'modal2_target_dim', 'lsi_random_state'
        ]
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                print(f"警告: 忽略无效参数 '{key}'")

    ### 1. 预处理方法（仅完成数据清洗/归一化，无降维，保持不变）
    def preprocess(self, ref_celltype_col, sample_id_col, ct_select, log_FC=1.25, top_n_filter=False, top_n=200, do_select_info_genes=True):
        """
        仅完成数据预处理（清洗、归一化、位点对齐），不执行特征工程（降维）
        流程：RNA数据预处理 + 模态2（ADT/Peak）数据预处理（TF-IDF/CLR） + 位点对齐
        """
        print(f"="*60)
        print(f"开始数据预处理（modal2类型: {self.modal2_type.upper()}）")
        print(f"="*60)

        # Step 1: RNA参考数据预处理（保持原逻辑）
        print(f"[Step 1/3] 处理参考RNA数据...")
        sc_shape_before_filter = self.ref_adata_dict['modal1'].shape
        adata_sc_filtered = pr.process_single_cell(self.ref_adata_dict['modal1'], ref_celltype_col)
        print(f"参考RNA数据形状：预处理前 {sc_shape_before_filter} → 预处理后 {adata_sc_filtered.shape}")

        # Step 2: 空间RNA数据预处理（保持原逻辑）
        print(f"[Step 2/3] 处理空间RNA数据...")
        adata_st_filtered = self.smo_adata_dict['modal1'].copy()
        adata_st_filtered.var_names_make_unique()

        common_genes = adata_sc_filtered.var_names.intersection(adata_st_filtered.var_names)
        print(f"空间与参考RNA共同基因数量: {len(common_genes)}")

        adata_sc_common = adata_sc_filtered[:, common_genes].copy()
        sc.pp.normalize_total(adata_sc_common, target_sum=1e4)

        # 信息基因筛选
        marker_genes = common_genes
        if do_select_info_genes:
            print(f"筛选信息基因...")
            temp_Basis = pr.calculate_celltype_averages(
                adata_sc_norm=adata_sc_common,
                celltype_col=ref_celltype_col,
                sample_id_col=sample_id_col,
                target_sum=1,
                normalize=True
            )
            marker_genes = selectInfoGenes(temp_Basis.T, adata_sc_common, common_genes, ct_select, ref_celltype_col,
                                           log_FC, top_n_filter, top_n)
            print(f"信息基因筛选完成，共 {len(marker_genes)} 个")

        # 构建基础矩阵
        adata_sc_marker = adata_sc_common[:, marker_genes].copy()
        Basis = pr.calculate_celltype_averages(
            adata_sc_norm=adata_sc_marker,
            celltype_col=ref_celltype_col,
            sample_id_col=sample_id_col,
            target_sum=1,
            normalize=True
        )
        self.ref_adata_dict['basis_matrix'] = Basis

        # 空间RNA数据最终处理
        adata_gene_norm = pr.process_spatial_data(
            adata_st_filtered,
            common_genes=common_genes,
            hvgs=marker_genes
        )
        print(f"空间RNA数据预处理完成，形状: {adata_gene_norm.shape}")

        # Step 3: 模态2（ADT/Peak）数据预处理（仅清洗/归一化，不降维）
        print(f"[Step 3/3] 处理模态2（{self.modal2_type.upper()}）数据（仅归一化，不执行降维）...")
        adata_modal2_raw = self.smo_adata_dict['modal2'].copy()
        if self.modal2_type == 'adt':
            # ADT：仅执行CLR标准化（预处理）
            adata_modal2_norm = pad.process_spatial_adt_data(adata_modal2_raw)
            print(f"ADT数据预处理完成（CLR标准化），形状: {adata_modal2_norm.shape}")
        elif self.modal2_type == 'peak':
            # Peak：仅执行TF-IDF归一化（预处理）
            adata_modal2_norm = pad.process_peak_data(adata_modal2_raw)
            print(f"Peak数据预处理完成（TF-IDF归一化），形状: {adata_modal2_norm.shape}")

        # Step 4: 空间位点对齐（保持原逻辑）
        common_spots = adata_gene_norm.obs_names.intersection(adata_modal2_norm.obs_names)
        adata_gene_norm = adata_gene_norm[common_spots, :].copy()
        adata_modal2_norm = adata_modal2_norm[common_spots, :].copy()
        num_spots = len(common_spots)
        print(f"空间位点对齐完成，共保留 {num_spots} 个共同位点")

        # 对齐基因（仅RNA）
        common_genes = adata_gene_norm.var_names.intersection(Basis.columns)
        adata_gene_norm = adata_gene_norm[:, common_genes].copy()
        print(f"RNA基因对齐完成，共保留 {len(common_genes)} 个共同基因")

        # 存储预处理后的数据（未执行特征工程）
        self.smo_adata_dict['modal1'] = adata_gene_norm
        self.smo_adata_dict['modal2'] = adata_modal2_norm
        self.feature_engineering_done = False  # 未执行特征工程
        print(f"="*60)
        print(f"数据预处理全部完成！")
        print(f"="*60)

    ### 2. 核心合并：特征工程 + 模型输入映射（单方法完成，第二模态单维度管控）
    def run_feature_engineering_and_mapping(self, obsm_name_rna=None, obsm_name_modal2=None):
        # 默认键名（兼容原有逻辑）
        obsm_name_rna = obsm_name_rna or "X_feat_rna"
        obsm_name_modal2 = obsm_name_modal2 or "X_feat_modal2"

        # 1. 处理modal1：RNA模态（PCA降维）
        adata_rna = self.smo_adata_dict["modal1"]
        scaler = StandardScaler()
        mat_rna_norm = scaler.fit_transform(adata_rna.X if not sparse.issparse(adata_rna.X) else adata_rna.X.toarray())
        pca_rna = PCA(n_components=min(self.pca_n_components_rna, mat_rna_norm.shape[1]))
        adata_rna.obsm["X_pca_rna"] = pca_rna.fit_transform(mat_rna_norm)
        adata_rna.obsm[obsm_name_rna] = adata_rna.obsm["X_pca_rna"]

        # 2. 处理modal2：Peak/ADT模态 —— 固定用 256 做判断
        adata_modal2 = self.smo_adata_dict["modal2"]
        modal2_original_dim = adata_modal2.shape[1]
        target_dim = 256  # 固定用256判断

        print(f"模态二原始维度: {modal2_original_dim}, 判断阈值: {target_dim}")

        if self.modal2_type == "peak":
            if not sparse.issparse(adata_modal2.X):
                adata_modal2.X = sparse.csr_matrix(adata_modal2.X)
            else:
                adata_modal2.X = adata_modal2.X.tocsr()

            if modal2_original_dim <= target_dim:
                # 不做LSI降维，直接用原始数据
                feat_out = adata_modal2.X.toarray()
                print(f"维度≤256，直接使用原始数据")
            else:
                # LSI 降到 256
                svd = TruncatedSVD(n_components=target_dim + 1, random_state=self.lsi_random_state, algorithm='arpack')
                emb_full = svd.fit_transform(adata_modal2.X)
                feat_out = emb_full[:, 1:]
                print(f"维度>256，LSI降维到目标维度")

            adata_modal2.obsm["X_lsi_peak"] = feat_out
            adata_modal2.obsm[obsm_name_modal2] = feat_out

        elif self.modal2_type == "adt":
            mat_adt = adata_modal2.X if not sparse.issparse(adata_modal2.X) else adata_modal2.X.toarray()
            scaler = StandardScaler()
            mat_adt_norm = scaler.fit_transform(mat_adt)

            if modal2_original_dim <= target_dim:
                # 不做PCA，直接用标准化后的数据
                feat_out = mat_adt_norm
                print(f"维度≤256，直接使用标准化后数据")
            else:
                # PCA 降到 256
                pca_adt = PCA(n_components=target_dim)
                feat_out = pca_adt.fit_transform(mat_adt_norm)
                print(f"维度>256，PCA降维到目标维度")

            adata_modal2.obsm["X_pca_adt"] = feat_out
            adata_modal2.obsm[obsm_name_modal2] = feat_out

        # 3. 保存并标记完成
        self.smo_adata_dict["modal1"] = adata_rna
        self.smo_adata_dict["modal2"] = adata_modal2
        self.feature_engineering_done = True

    ### 3. 后续方法（保持不变，适配合并后的流程）
    def create_spatialgraph(self, obsm_spatial, K_neighbors = 8):
        if not self.feature_engineering_done:
            warnings.warn("请先完成特征工程+输入映射，再构建空间图！", UserWarning)

        coords = self.smo_adata_dict['modal1'].obsm[obsm_spatial]
        adj = kneighbors_graph(coords, K_neighbors, mode='connectivity')
        edge_index = np.array(np.nonzero(adj))
        print(f"空间图构建完成，存储在 smo_adata_dict['modal1'].uns['spatial_graph']")
        self.smo_adata_dict['modal1'].uns['spatial_graph'] = edge_index

    def get_laplacian(self, normalize=True):
        edge_index = self.smo_adata_dict['modal1'].uns.get('spatial_graph')
        if edge_index is None:
            raise ValueError("请先调用create_spatialgraph()生成空间图")

        edge_index = torch.tensor(edge_index, dtype=torch.long) if not isinstance(edge_index, torch.Tensor) else edge_index
        device = edge_index.device
        num_spots = self.smo_adata_dict['modal1'].shape[0]

        A = torch.zeros((num_spots, num_spots), device=device)
        A[edge_index[0], edge_index[1]] = 1.0
        A[edge_index[1], edge_index[0]] = 1.0

        D = torch.diag(A.sum(dim=1)).to(device)
        I = torch.eye(num_spots, device=device)
        if normalize:
            D_sqrt_inv = torch.inverse(torch.sqrt(D + 1e-8))
            L = I - D_sqrt_inv @ A @ D_sqrt_inv
        else:
            L = D - A

        return L, A, D

    def create_featuregraph(self, obsm_name='X_pca', K_neighbors = 8, name = 'namemod'):
        if not isinstance(self.smo_adata_dict, dict):
            raise ValueError("smo_adata_dict must be a dictionary of AnnData objects.")

        for key, adata in self.smo_adata_dict.items():
            if not isinstance(adata, ad.AnnData):
                warnings.warn(
                    f"Skipping key '{key}': Expected AnnData object, got {type(adata)}.",
                    UserWarning
                )
                continue

            if obsm_name not in adata.obsm:
                warnings.warn(
                    f"Skipping key '{key}': obsm '{obsm_name}' not found in AnnData object.",
                    UserWarning
                )
                continue
            adj = kneighbors_graph(adata.obsm[obsm_name], K_neighbors, mode='connectivity')
            edge_index = np.array(np.nonzero(adj))
            print(f"为 {key} 构建特征图完成，存储在 adata.uns['{name}']")
            adata.uns[name] = edge_index

    def train(self, embd_dim=50, method=1, name1='name1', name2='name2', obsm_name_rna='X_pca', obsm_name_adt='X_pca',
              plot=True, model_save=False, model_save_dir="trained_models"):
        """
        核心特点：输入维度为 modal2_target_dim（第二模态单维度），兼容合并后的流程
        """
        if not self.feature_engineering_done:
            raise ValueError("请先调用 run_feature_engineering_and_mapping() 完成特征工程+输入映射！")

        ### 1. 固定随机种子（保持原逻辑）
        device = self.nn_para_dict['device']
        seed = self.nn_para_dict['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        ### 2. 数据准备（保持原逻辑，输入维度已正确）
        gene_counts = self.smo_adata_dict['modal1'].layers['counts']
        gene_counts = gene_counts.toarray() if hasattr(gene_counts, 'toarray') else gene_counts
        gene_spot_counts_sum = np.sum(gene_counts, axis=1, keepdims=True)

        gene_pca = self.smo_adata_dict['modal1'].obsm[obsm_name_rna]
        modal2_pca = self.smo_adata_dict['modal2'].obsm[obsm_name_adt]

        # 空间图检查
        if 'spatial_graph' not in self.smo_adata_dict['modal1'].uns:
            raise ValueError("spatial_graph not found! Run create_spatialgraph() first.")
        g_spatial = torch.tensor(self.smo_adata_dict['modal1'].uns['spatial_graph'], dtype=torch.long, device=device)

        # 特征图检查
        if name1 not in self.smo_adata_dict['modal1'].uns or name2 not in self.smo_adata_dict['modal2'].uns:
            raise ValueError("feature_graph not found! Run create_featuregraph() first.")
        g_feature_modal1 = torch.tensor(self.smo_adata_dict['modal1'].uns[name1], dtype=torch.long, device=device)
        g_feature_modal2 = torch.tensor(self.smo_adata_dict['modal2'].uns[name2], dtype=torch.long, device=device)

        # 数据转tensor
        gene_pca_tensor = torch.tensor(gene_pca, dtype=torch.float, device=device)
        modal2_pca_tensor = torch.tensor(modal2_pca, dtype=torch.float, device=device)
        gene_recon_target = gene_pca_tensor
        modal2_recon_target = modal2_pca_tensor

        gene_counts_tensor = torch.tensor(gene_counts, dtype=torch.float, device=device)
        gene_spot_sum_tensor = torch.tensor(gene_spot_counts_sum, dtype=torch.float, device=device)

        cell_type_gene_tensor = torch.tensor(self.ref_adata_dict['basis_matrix'].values, dtype=torch.float, device=device)
        num_cell_types = cell_type_gene_tensor.shape[0]

        ### 3. 模型参数配置（保持原逻辑，输入维度自动适配）
        hidden_dim = self.nn_para_dict['hidden_dim']
        embedding_dim = embd_dim
        in_fea_dim_modal1 = gene_pca.shape[1]
        in_fea_dim_modal2 = modal2_pca.shape[1]

        print(f"="*60)
        print(f"模型输入维度确认：")
        print(f"  - RNA模态：{in_fea_dim_modal1} 维")
        print(f"  - 第二模态（{self.modal2_type.upper()}）：{in_fea_dim_modal2} 维（= modal2_target_dim）")
        print(f"="*60)

        ### 4. 初始化模型组件（保持原逻辑）
        if method == 1:
            cross_fusion = GCN_between_attention_encoder(
                in_fea_dim_modal1, in_fea_dim_modal2, hidden_dim, embedding_dim
            ).to(device)
            modal1_encoder = cross_fusion.encode_modal1
            modal2_encoder = cross_fusion.encode_modal2
        elif method == 2:
            cross_fusion = GCN_between_within_attention_encoder(
                in_fea_dim_modal1, in_fea_dim_modal2, hidden_dim, embedding_dim
            ).to(device)
            modal1_encoder = cross_fusion.encode_modal1
            modal2_encoder = cross_fusion.encode_modal2
        else:
            raise ValueError(f"Unsupported method {method}! Use 1 or 2.")

        reconstructor = ExpressionReconstructor(
            embedding_dim, pca_dim_rna=in_fea_dim_modal1, pca_dim_adt=in_fea_dim_modal2
        ).to(device)

        proportion_predictor = CellTypeProportionPredictor(embedding_dim, num_cell_types).to(device)
        consistency_verifier = ConsistencyVerifier(
            modal1_encoder=modal1_encoder,
            modal2_encoder=modal2_encoder,
            reconstructor=reconstructor
        ).to(device)

        ### 负二项分布参数（保持原逻辑）
        gene_theta = nn.Parameter(torch.ones(gene_counts.shape[1], device=device))
        gene_spot_offset = nn.Parameter(torch.zeros(gene_counts.shape[0], device=device))
        gene_gene_offset = nn.Parameter(torch.zeros(gene_counts.shape[1], device=device))

        ### 5. 优化器（保持原逻辑）
        optimizer = torch.optim.Adam(
            list(cross_fusion.parameters()) +
            list(reconstructor.parameters()) +
            list(proportion_predictor.parameters()) +
            list(consistency_verifier.parameters()) +
            [gene_theta, gene_spot_offset, gene_gene_offset],
            lr=self.nn_para_dict['learning_rate'],
            weight_decay=1e-5
        )

        ### 6. 计算拉普拉斯矩阵（保持原逻辑）
        L, _, _ = self.get_laplacian(normalize=True)
        L = L.to(device)

        ### 7. 训练流程（保持原逻辑）
        epochs = self.nn_para_dict['epochs']
        weight_nb, weight_recon = self.weight_loss
        weight_consistency = self.weight_consistency
        weight_spatial = self.weight_spatial

        # 初始化损失记录
        total_losses = []
        gene_nb_losses = []
        recon_losses = []
        consistency_losses = []
        spatial_losses = []

        for epoch in range(epochs):
            # 所有组件设为train模式
            cross_fusion.train()
            reconstructor.train()
            proportion_predictor.train()
            consistency_verifier.train()

            # 前向传播
            if method == 1:
                embedding, y1, y2, attn_gene2other, attn_other2gene = cross_fusion(
                    g_spatial_omics1=g_spatial,
                    feat_omics1=gene_pca_tensor,
                    g_spatial_omics2=g_spatial,
                    feat_omics2=modal2_pca_tensor
                )
            elif method == 2:
                embedding, y1, y2, attn_gene2other, attn_other2gene = cross_fusion(
                    g_spatial_omics1=g_spatial,
                    g_feature_omics1=g_feature_modal1,
                    feat_omics1=gene_pca_tensor,
                    g_spatial_omics2=g_spatial,
                    g_feature_omics2=g_feature_modal2,
                    feat_omics2=modal2_pca_tensor
                )

            ### 计算各损失项
            # 1. 重构损失
            gene_recon, modal2_recon = reconstructor(embedding)
            gene_recon_loss = F.mse_loss(gene_recon, gene_recon_target)
            modal2_recon_loss = F.mse_loss(modal2_recon, modal2_recon_target)
            total_recon_loss = weight_recon * (gene_recon_loss + modal2_recon_loss)

            # 2. 基因NB损失
            cell_proportions = proportion_predictor(embedding)
            gene_mu = compute_gene_nb_mu(cell_type_gene_tensor, cell_proportions)
            gene_mu = torch.exp(torch.log(gene_mu) + gene_spot_offset.unsqueeze(1) + gene_gene_offset.unsqueeze(0))
            gene_mu = gene_mu * gene_spot_sum_tensor
            gene_nb_loss = weight_nb * gene_negative_binomial_loss(gene_counts_tensor, gene_mu, gene_theta)

            # 3. 一致性损失
            y1_recon, y2_recon = consistency_verifier(
                y1=y1, y2=y2,
                g_spatial1=g_spatial, g_feature1=g_feature_modal1,
                g_spatial2=g_spatial, g_feature2=g_feature_modal2
            )
            consistency_loss = weight_consistency * (F.mse_loss(y1, y1_recon) + F.mse_loss(y2, y2_recon))

            # 4. 空间平滑损失
            L_P = torch.matmul(L, cell_proportions)
            P_T_L_P = torch.matmul(cell_proportions.T, L_P)
            spatial_loss = weight_spatial * 0.5 * torch.trace(P_T_L_P)

            ### 总损失拼接
            total_loss = gene_nb_loss + total_recon_loss + consistency_loss + spatial_loss

            ### 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ### 损失记录
            total_losses.append(total_loss.item())
            gene_nb_losses.append(gene_nb_loss.item())
            recon_losses.append(total_recon_loss.item())
            consistency_losses.append(consistency_loss.item())
            spatial_losses.append(spatial_loss.item())

            ### 训练日志
            if (epoch + 1) % 100 == 0:
                torch.cuda.empty_cache()
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"总损失: {total_loss.item():.4f} | "
                    f"基因NB损失: {gene_nb_loss.item():.4f} | "
                    f"平均重构损失: {total_recon_loss.item()/2:.4f} | "
                    f"一致性损失: {consistency_loss.item():.4f} | "
                    f"空间平滑损失: {spatial_loss.item():.4f}"
                )

                ### 早停逻辑
                if len(total_losses) > 60 and min(total_losses) != min(total_losses[-60:]):
                    print(f"Early stopped at epoch {epoch + 1}.")
                    final_embedding = embedding.detach().cpu()
                    final_gene_recon = gene_recon.detach().cpu()
                    final_modal2_recon = modal2_recon.detach().cpu()
                    final_cell_proportions = cell_proportions.detach().cpu()
                    final_attn_gene2other = attn_gene2other.detach().cpu()
                    final_attn_other2gene = attn_other2gene.detach().cpu()
                    break

            if epoch == epochs - 1:
                final_embedding = embedding.detach().cpu()
                final_gene_recon = gene_recon.detach().cpu()
                final_modal2_recon = modal2_recon.detach().cpu()
                final_cell_proportions = cell_proportions.detach().cpu()
                final_attn_gene2other = attn_gene2other.detach().cpu()
                final_attn_other2gene = attn_other2gene.detach().cpu()
        ### 8. 损失曲线绘制
        if plot:
            plt.figure(figsize=(25, 5))
            plt.subplot(1, 5, 1)
            plt.plot(total_losses, 'b-')
            plt.title('Total Loss')
            plt.xlabel('Epoch')

            plt.subplot(1, 5, 2)
            plt.plot(gene_nb_losses, 'g-')
            plt.title(f'Gene Negative Binomial Loss (w={weight_nb})')
            plt.xlabel('Epoch')

            plt.subplot(1, 5, 3)
            plt.plot(recon_losses, 'r-')
            plt.title(f'Average Reconstruction Loss (w={weight_recon})')
            plt.xlabel('Epoch')

            plt.subplot(1, 5, 4)
            plt.plot(consistency_losses, 'orange')
            plt.title(f'Consistency Loss (w={weight_consistency})')
            plt.xlabel('Epoch')

            plt.subplot(1, 5, 5)
            plt.plot(spatial_losses, 'purple')
            plt.title(f'Spatial Smooth Loss (w={weight_spatial})')
            plt.xlabel('Epoch')

            plt.tight_layout()
            os.makedirs(model_save_dir, exist_ok=True)
            plt.savefig(os.path.join(model_save_dir, "training_losses.png"), dpi=300)
            plt.close()

        ### 9. 模型保存
        if model_save:
            os.makedirs(model_save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_save_dir, f"SpaMultiDecon_model_{timestamp}.pth")

            checkpoint = {
                'method': method,
                'cross_fusion_state_dict': cross_fusion.state_dict(),
                'reconstructor_state_dict': reconstructor.state_dict(),
                'proportion_predictor_state_dict': proportion_predictor.state_dict(),
                'consistency_verifier_state_dict': consistency_verifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'losses': {
                    'total': total_losses,
                    'gene_nb': gene_nb_losses,
                    'recon': recon_losses,
                    'consistency': consistency_losses,
                    'spatial': spatial_losses
                },
                'model_config': {
                    'embd_dim': embedding_dim,
                    'in_fea_dim_modal1': in_fea_dim_modal1,
                    'in_fea_dim_modal2': in_fea_dim_modal2,
                    'hidden_dim': hidden_dim,
                    'modal2_type': self.modal2_type,
                    'modal2_target_dim': self.modal2_target_dim
                },
                'data_params': {
                    'gene_theta': gene_theta,
                    'gene_spot_offset': gene_spot_offset,
                    'gene_gene_offset': gene_gene_offset
                }
            }
            torch.save(checkpoint, model_path)
            print(f"模型已保存到: {model_path}")

        ### 10. 结果存储
        adata_result = self.smo_adata_dict['modal1'].copy()
        adata_result.obsm['cell_type_proportions'] = final_cell_proportions.numpy()
        proportion_df = pd.DataFrame(
            final_cell_proportions.numpy(),
            index=adata_result.obs_names,
            columns=self.ref_adata_dict['basis_matrix'].index.tolist()
        )
        adata_result.obs = pd.concat([adata_result.obs, proportion_df], axis=1)
        adata_result.obsm['embedding'] = final_embedding.numpy()
        adata_result.obsm['reconstructed_gene'] = final_gene_recon.numpy()
        adata_result.obsm['reconstructed_modal2'] = final_modal2_recon.numpy()
        adata_result.obsm['attn_gene2other'] = final_attn_gene2other.detach().cpu().numpy()
        adata_result.obsm['attn_other2gene'] = final_attn_other2gene.detach().cpu().numpy()

        return adata_result, cross_fusion