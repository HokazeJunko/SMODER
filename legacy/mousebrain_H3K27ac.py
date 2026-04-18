# ----------------------
# SpaMultiDecon_pipeline.py
# 空间多模态解卷积流程（一次特征工程+双模态独立obsm键名 | 高效可运行版）
# 核心：1. 一次调用内置特征工程函数处理双模态 2. 在save_results内部完整整合双模态信息到结果文件
# ----------------------
import anndata as ad
import os
import pandas as pd
import numpy as np
import torch
import warnings
from datetime import datetime
import scipy.sparse as sparse
# 导入原多模态类（确保 SpaMultiDecon 所在路径在 Python 环境变量中）
from SpaMultiDecon import SpaMultiDecon_two_modals

# 忽略冗余警告
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------------
# 1. 基础配置（修改为实际路径）
# ----------------------
def set_base_config():
    base_config = {
        "sc_rna_path": "/data/xiongsc/data/SMODER/mousebrain_H3K27ac/sc_mousebrain_processed.h5ad",
        "st_rna_path": "/data/xiongsc/data/SMODER/mousebrain_H3K27ac/RNA.h5ad",
        "st_adt_path": "/data/xiongsc/data/SMODER/mousebrain_H3K27ac/peak.h5ad",
        "output_dir": "/data/xiongsc/projects/SMODER/outputs/mousebrain_H3K27ac_run2",
        "model_save_dir": os.path.join(
            "/data/xiongsc/projects/SMODER/outputs/mousebrain_H3K27ac_run2",
            "trained_models"
        ),
    }
    os.makedirs(base_config["output_dir"], exist_ok=True)
    os.makedirs(base_config["model_save_dir"], exist_ok=True)
    return base_config

# ----------------------
# 2. 分析参数配置（双模态独立obsm键名 + 适配一次特征工程）
# ----------------------
def set_analysis_params():
    params = {
        # ---------------------- 1. 数据预处理参数 ----------------------
        "ref_celltype_col": "annotation_1",  # 单细胞参考数据的细胞类型列名
        "sample_id_col": "sample",           # 单细胞参考数据的样本列名（用于计算Basis矩阵）
        "log_FC": 1.25,                      # 筛选信息基因的logFC阈值
        "do_select_info_genes": True,        # 是否筛选信息基因
        "ct_select": None,                   # 需保留的细胞类型（None表示保留所有）
        "modal2_type": "peak",               # 第二模态类型（adt/peak，对应模型内置预处理）

        # ---------------------- 2. 图构建参数 ----------------------
        "K_spatial": 8,                      # 空间邻接图的邻域数量（K近邻）
        "K_feature": 8,                      # 特征邻接图的邻域数量（K近邻）

        # ---------------------- 3. 特征工程参数（双模态独立obsm键名 | 核心） ----------------------
        "pca_n_components_rna": 256,          # RNA模态PCA降维目标维度
        "modal2_target_dim": 256,            # 第二模态最终目标维度
        "obsm_name_rna": "X_feat_rna",       # RNA模态专属obsm键名（独立）
        "obsm_name_modal2": "X_feat_modal2", # 第二模态专属obsm键名（独立，区分Peak/ADT）

        # ---------------------- 4. 模型训练参数 ----------------------
        "method": 2,                         # 模型方法（1=仅跨模态注意力，2=模态内+跨模态）
        "embd_dim": 50,                      # 嵌入层维度
        "epochs": 10,                       # 最大训练轮次
        "learning_rate": 2e-3,               # 学习率
        "hidden_dim": 512,                   # GCN隐藏层维度
        "weight_loss": [1, 0.001],           # 损失权重：[负二项损失权重，PCA重构损失权重]
        "weight_consistency": 1,             # 一致性损失权重
        "weight_spatial": 1e-5,              # 空间平滑损失权重
        "seed": 1,                           # 随机种子（确保可复现）
        "model_save": False,                  # 是否保存训练后的模型
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",  # 训练设备
    }
    return params

# ----------------------
# 3. 数据加载与格式展示（仅加载原始数据，不自定义预处理）
# ----------------------
def load_and_show_data(base_config, params):
    print(f"=== 数据加载与格式展示（{datetime.now().strftime('%H:%M:%S')}）===")

    # 1. 加载单细胞参考RNA数据
    print(f"\n【1. 单细胞参考RNA数据】")
    adata_sc = ad.read_h5ad(base_config["sc_rna_path"])
    print(f"  数据形状（细胞数 × 基因数）：{adata_sc.shape}")
    print(f"  数据类型：{type(adata_sc.X)}（稀疏矩阵：{sparse.issparse(adata_sc.X)}）")
    print(f"  细胞类型列（{params['ref_celltype_col']}）唯一值数量：{len(adata_sc.obs[params['ref_celltype_col']].unique())}")
    print(f"  样本列（{params['sample_id_col']}）唯一值数量：{len(adata_sc.obs[params['sample_id_col']].unique()) if params['sample_id_col'] in adata_sc.obs.columns else '不存在'}")

    # 2. 加载空间RNA数据
    print(f"\n【2. 空间RNA数据】")
    adata_st_rna = ad.read_h5ad(base_config["st_rna_path"])
    if not adata_st_rna.var_names.is_unique:
        adata_st_rna.var_names_make_unique()
        print(f"  已处理重复基因名，确保变量名唯一")
    print(f"  数据形状（空间位点 × 基因数）：{adata_st_rna.shape}")
    print(f"  数据类型：{type(adata_st_rna.X)}（稀疏矩阵：{sparse.issparse(adata_st_rna.X)}）")

    # 3. 加载第二模态数据（ADT/Peak）
    print(f"\n【3. 第二模态数据（{params['modal2_type'].upper()}）】")
    adata_st_modal2 = ad.read_h5ad(base_config["st_adt_path"])
    print(f"  数据形状（空间位点 × 特征数）：{adata_st_modal2.shape}")
    print(f"  数据类型：{type(adata_st_modal2.X)}（稀疏矩阵：{sparse.issparse(adata_st_modal2.X)}）")


    # 4. 自动获取细胞类型列表
    all_celltypes = adata_sc.obs[params["ref_celltype_col"]].unique().tolist()
    params["ct_select"] = all_celltypes if params["ct_select"] is None else params["ct_select"]
    print(f"\n=== 数据加载完成 ===")
    print(f"  保留细胞类型数：{len(params['ct_select'])}（前5个：{params['ct_select'][:5]}...）")

    # 5. 组织数据字典（适配模型输入格式，独立容器隔离模态）
    ref_dict = {"modal1": adata_sc}
    smo_dict = {"modal1": adata_st_rna, "modal2": adata_st_modal2}

    return ref_dict, smo_dict, params

# ----------------------
# 4. 模型初始化与内置预处理（复用 SpaMultiDecon 预处理）
# ----------------------
def init_model_and_preprocess(ref_dict, smo_dict, params):
    print(f"\n=== 模型初始化与内置预处理（{datetime.now().strftime('%H:%M:%S')}）===")

    # 1. 初始化多模态模型
    model = SpaMultiDecon_two_modals(
        ref_adata_dict=ref_dict,
        smo_adata_dict=smo_dict,
        nn_para_dict={
            "epochs": params["epochs"],
            "learning_rate": params["learning_rate"],
            "device": params["device"],
            "hidden_dim": params["hidden_dim"],
            "weight_loss": params["weight_loss"],
            "weight_consistency": params["weight_consistency"],
            "weight_spatial": params["weight_spatial"],
            "seed": params["seed"],
            "pca_n_components_rna": params["pca_n_components_rna"],
            "modal2_target_dim": params["modal2_target_dim"]
        },
        modal2_type=params["modal2_type"]  # 指定第二模态类型，触发对应内置预处理/特征工程
    )

    # 2. 执行模型内置预处理（核心：无需自定义ADT/Peak预处理，模型自动处理）
    model.preprocess(
        ref_celltype_col=params["ref_celltype_col"],
        sample_id_col=params["sample_id_col"],
        ct_select=params["ct_select"],
        log_FC=params["log_FC"],
        do_select_info_genes=params["do_select_info_genes"]
    )

    # 3. 展示预处理结果
    print(f"\n=== 内置预处理结果展示 ===")
    print(f"  Basis矩阵形状（细胞类型 × 信息基因）：{model.ref_adata_dict['basis_matrix'].shape}")
    print(f"  预处理后空间RNA形状（位点 × 信息基因）：{model.smo_adata_dict['modal1'].shape}")
    print(f"  预处理后第二模态形状（位点 × 特征）：{model.smo_adata_dict['modal2'].shape}")
    print(f"  共同空间位点数：{len(model.smo_adata_dict['modal1'].obs_names)}")

    # 4. 提前传入双模态独立键名到模型（供内置特征工程函数使用）
    model.obsm_name_rna = params["obsm_name_rna"]
    model.obsm_name_modal2 = params["obsm_name_modal2"]

    return model

# ----------------------
# 5. 特征工程与图构建（一次调用内置函数 + 双模态独立键名保存 | 核心优化）
# ----------------------
def feature_engineering_and_graph_build(model, params):
    print(f"\n=== 特征工程与图构建（{datetime.now().strftime('%H:%M:%S')}）===")

    # 1. 提取核心参数
    obsm_name_rna = params["obsm_name_rna"]
    obsm_name_modal2 = params["obsm_name_modal2"]
    modal2_type = params["modal2_type"]

    # 2. 一次调用内置特征工程函数（核心优化）
    # 内置函数原生支持同时处理modal1（RNA）和modal2（Peak/ADT）
    # 我们在函数内部让两个模态分别保存到独立obsm键名，无需重复调用
    print(f"\n--- 一次执行内置特征工程（同时处理RNA+{modal2_type.upper()}，双模态独立键名）---")
    model.run_feature_engineering_and_mapping(
        obsm_name_rna=obsm_name_rna,
        obsm_name_modal2=obsm_name_modal2
    )

    # 3. 展示双模态处理结果（一次运行，两个模态均处理完成）
    adata_rna = model.smo_adata_dict["modal1"]
    adata_modal2 = model.smo_adata_dict["modal2"]
    rna_special_key = "X_pca_rna"
    modal2_special_key = "X_lsi_peak" if modal2_type == "peak" else "X_pca_adt"

    print(f"  双模态特征工程完成（一次运行，无冗余）：")
    print(f"  --- RNA模态 ---")
    print(f"    - 专属降维结果：obsm['{rna_special_key}']（形状：{adata_rna.obsm[rna_special_key].shape}）")
    print(f"    - 模型输入结果：obsm['{obsm_name_rna}']（形状：{adata_rna.obsm[obsm_name_rna].shape}）")
    print(f"  --- {modal2_type.upper()}模态 ---")
    print(f"    - 专属降维结果：obsm['{modal2_special_key}']（形状：{adata_modal2.obsm[modal2_special_key].shape}）")
    print(f"    - 模型输入结果：obsm['{obsm_name_modal2}']（形状：{adata_modal2.obsm[obsm_name_modal2].shape}）")
    if modal2_type == "peak":
        print(f"    - LSI参数记录：uns['lsi']（内置自动生成）")

    # 4. 标记特征工程完成
    model.feature_engineering_done = True

    # 5. 构建空间邻接图（RNA模态为基准，复用给第二模态）
    print(f"\n--- 构建空间邻接图（K={params['K_spatial']}）---")
    model.create_spatialgraph(
        obsm_spatial="spatial",
        K_neighbors=params["K_spatial"]
    )
    # 第二模态复用空间图（容器独立，仅复用邻接关系）
    model.smo_adata_dict["modal2"].uns["spatial_graph"] = model.smo_adata_dict["modal1"].uns["spatial_graph"]
    print(f"  空间图构建完成，已复用至第二模态（存储键：uns['spatial_graph']）")

    # 6. 构建分模态特征邻接图（使用各自专属obsm键名，复用内置create_featuregraph）
    print(f"\n--- 构建分模态特征邻接图（K={params['K_feature']}）---")
    # RNA模态：使用RNA专属obsm键名
    print(f"  构建RNA特征邻接图（基于obsm['{obsm_name_rna}']）...")
    model.create_featuregraph(
        obsm_name=obsm_name_rna,
        K_neighbors=params["K_feature"],
        name='rna'
    )
    # 第二模态：使用第二模态专属obsm键名
    print(f"  构建{modal2_type.upper()}特征邻接图（基于obsm['{obsm_name_modal2}']）...")
    model.create_featuregraph(
        obsm_name=obsm_name_modal2,
        K_neighbors=params["K_feature"],
        name='adt'
    )
    print(f"  分模态特征图构建完成：")
    print(f"    - RNA特征图：uns['rna'] | 基于obsm['{obsm_name_rna}']")
    print(f"    - {modal2_type.upper()}特征图：uns['adt'] | 基于obsm['{obsm_name_modal2}']")

    # 7. 提取降维后维度信息（用于后续日志）
    dim_rna = adata_rna.obsm[obsm_name_rna].shape[1]
    dim_modal2 = adata_modal2.obsm[obsm_name_modal2].shape[1]

    return model, dim_rna, dim_modal2

# ----------------------
# 6. 模型训练与编码器结果提取（适配双模态独立键名）
# ----------------------
def train_model(model, params, dim_rna, dim_modal2, base_config):
    print(f"\n=== 模型训练（{datetime.now().strftime('%H:%M:%S')}，设备：{params['device']}）===")

    # 1. 提取训练核心参数（双模态独立obsm键名）
    method = params.get("method", 2)
    embd_dim = params.get("embd_dim", 50)
    obsm_name_rna = params.get("obsm_name_rna", "X_feat_rna")
    obsm_name_modal2 = params.get("obsm_name_modal2", "X_feat_modal2")
    model_save = params.get("model_save", True)
    model_save_dir = base_config["model_save_dir"]

    # 2. 执行模型训练（传入双模态专属键名，适配模型输入）
    adata_result, cross_fusion = model.train(
        embd_dim=embd_dim,
        method=method,
        name1='rna',
        name2='adt',
        obsm_name_rna=obsm_name_rna,
        obsm_name_adt=obsm_name_modal2,  # 传入第二模态专属键名
        plot=True,
        model_save=model_save,
        model_save_dir=model_save_dir
    )

    # 3. 提取编码器结果（关闭梯度计算，节省内存）
    device = params["device"]
    with torch.no_grad():
        cross_fusion.eval()  # 切换至评估模式

        # 准备双模态输入张量（使用各自专属obsm键名提取特征，一次获取完成）
        rna_feat = torch.tensor(
            model.smo_adata_dict['modal1'].obsm[obsm_name_rna].copy(),
            dtype=torch.float32,
            device=device
        )
        modal2_feat = torch.tensor(
            model.smo_adata_dict['modal2'].obsm[obsm_name_modal2].copy(),
            dtype=torch.float32,
            device=device
        )
        spatial_graph = torch.tensor(
            model.smo_adata_dict['modal1'].uns['spatial_graph'],
            dtype=torch.long,
            device=device
        )

        # 提取分模态特征图（仅method=2需要）
        rna_feature_graph = torch.tensor(
            model.smo_adata_dict['modal1'].uns['rna'],
            dtype=torch.long,
            device=device
        ) if method == 2 else None
        modal2_feature_graph = torch.tensor(
            model.smo_adata_dict['modal2'].uns['adt'],
            dtype=torch.long,
            device=device
        ) if method == 2 else None

        # 编码器前向传播（双模态独立提取，复用内置模型逻辑）
        if method == 1:
            rna_encoder = cross_fusion.encode_modal1(spatial_graph, None, rna_feat).cpu().numpy()
            modal2_encoder = cross_fusion.encode_modal2(spatial_graph, None, modal2_feat).cpu().numpy()
        else:
            rna_encoder = cross_fusion.encode_modal1(spatial_graph, rna_feature_graph, rna_feat).cpu().numpy()
            modal2_encoder = cross_fusion.encode_modal2(spatial_graph, modal2_feature_graph, modal2_feat).cpu().numpy()

    # 4. 保存编码器结果到结果AnnData（标记双模态来源）
    adata_result.obsm['rna_encoder'] = rna_encoder
    adata_result.obsm[f"{params['modal2_type']}_encoder"] = modal2_encoder
    print(f"\n=== 训练完成 ===")
    print(f"  细胞类型比例形状：{adata_result.obsm['cell_type_proportions'].shape}")
    print(f"  RNA编码器结果形状：{rna_encoder.shape}（存储于obsm['rna_encoder']）")
    print(f"  {params['modal2_type'].upper()}编码器结果形状：{modal2_encoder.shape}（存储于obsm['{params['modal2_type']}_encoder']）")

    return adata_result

# ----------------------
# 辅助函数：清理AnnData特殊字符（确保可保存）
# ----------------------
def clean_anndata_for_save(adata):
    """清理AnnData中所有层级的特殊字符，避免h5ad保存报错"""
    illegal_chars = {'/': '_', '\\': '_', ':': '_', '*': '_', '?': '_',
                     '"': '_', '<': '_', '>': '_', '|': '_'}

    def replace_illegal(text):
        if pd.isna(text):
            return text
        text_str = str(text)
        for illegal, legal in illegal_chars.items():
            text_str = text_str.replace(illegal, legal)
        return text_str

    # 处理obs
    adata.obs.columns = [replace_illegal(col) for col in adata.obs.columns]
    adata.obs.index = [replace_illegal(idx) for idx in adata.obs.index]

    # 处理var
    adata.var.columns = [replace_illegal(col) for col in adata.var.columns]
    adata.var.index = [replace_illegal(idx) for idx in adata.var.index]

    # 处理obsm键名
    new_obsm = {replace_illegal(k): v for k, v in adata.obsm.items()}
    adata.obsm = new_obsm

    # 处理uns
    new_uns = {replace_illegal(k): v for k, v in adata.uns.items()}
    adata.uns = new_uns

    return adata

# ----------------------
# 7. 结果保存与日志记录（内部直接整合双模态信息，不新增额外函数）
# ----------------------
def save_results(adata_result, model, params, base_config, dim_rna, dim_modal2):
    print(f"\n=== 结果保存（{datetime.now().strftime('%H:%M:%S')}）===")

    # ---------------------- 核心：在函数内部直接整合双模态所有信息（无外部函数） ----------------------
    print(f"\n=== 双模态信息完整整合（保留所有核心数据，内部直接处理）===")
    # 1. 提取核心标识和模态数据（内部临时提取，无需外部传入）
    modal2_type = params["modal2_type"]
    obsm_name_rna = params["obsm_name_rna"]
    obsm_name_modal2 = params["obsm_name_modal2"]
    adata_rna = model.smo_adata_dict["modal1"]
    adata_modal2 = model.smo_adata_dict["modal2"]
    adata_result_integrated = adata_result.copy()  # 复制结果数据，避免修改原始数据

    # 2. 内部直接整合obsm数据（feat特征 + 专属降维结果 + 空间坐标，模态前缀区分，无冲突）
    print(f"  整合obsm数据（RNA+{modal2_type.upper()}）...")
    # RNA模态obsm整合（添加rna_前缀，清晰标识来源）
    rna_obsm_items = [
        (obsm_name_rna, f"rna_{obsm_name_rna}"),  # RNA专属feat特征
        ("X_pca_rna", "rna_X_pca"),                # RNA专属PCA降维结果
        ("spatial", "spatial")                     # 空间坐标（共用，仅保留一份高质量版本）
    ]
    for src_key, dst_key in rna_obsm_items:
        if src_key in adata_rna.obsm:
            adata_result_integrated.obsm[dst_key] = adata_rna.obsm[src_key].copy()
            print(f"    - 已整合RNA模态：obsm['{dst_key}']（形状：{adata_result_integrated.obsm[dst_key].shape}）")

    # 第二模态obsm整合（添加{modal2_type}_前缀，清晰标识来源）
    modal2_special_key = "X_lsi_peak" if modal2_type == "peak" else "X_pca_adt"
    modal2_obsm_items = [
        (obsm_name_modal2, f"{modal2_type}_{obsm_name_modal2}"),  # 第二模态专属feat特征
        (modal2_special_key, f"{modal2_type}_X_{'lsi' if modal2_type == 'peak' else 'pca'}")  # 第二模态专属降维结果
    ]
    for src_key, dst_key in modal2_obsm_items:
        if src_key in adata_modal2.obsm:
            adata_result_integrated.obsm[dst_key] = adata_modal2.obsm[src_key].copy()
            print(f"    - 已整合{modal2_type.upper()}模态：obsm['{dst_key}']（形状：{adata_result_integrated.obsm[dst_key].shape}）")

    # 3. 内部直接整合uns数据（特征图 + LSI参数，模态前缀区分）
    print(f"\n  整合uns数据（RNA+{modal2_type.upper()}）...")
    # RNA模态uns整合（特征图）
    rna_uns_items = [
        ("spatial_graph", "spatial_graph"),  # 空间邻接图（共用，仅保留一份）
        ("rna", f"rna_feature_graph")        # RNA专属特征邻接图
    ]
    for src_key, dst_key in rna_uns_items:
        if src_key in adata_rna.uns:
            adata_result_integrated.uns[dst_key] = adata_rna.uns[src_key].copy()
            print(f"    - 已整合RNA模态：uns['{dst_key}']")

    # 第二模态uns整合（特征图 + LSI参数）
    modal2_uns_items = [
        ("adt", f"{modal2_type}_feature_graph"),  # 第二模态专属特征邻接图
    ]
    if modal2_type == "peak" and "lsi" in adata_modal2.uns:
        modal2_uns_items.append(("lsi", f"{modal2_type}_lsi_params"))  # Peak模态专属LSI参数

    for src_key, dst_key in modal2_uns_items:
        if src_key in adata_modal2.uns:
            adata_result_integrated.uns[dst_key] = adata_modal2.uns[src_key].copy()
            print(f"    - 已整合{modal2_type.upper()}模态：uns['{dst_key}']")

    # 4. 标记整合完成信息（内部记录，方便后续追溯）
    adata_result_integrated.uns["bimodal_integration_info"] = {
        "modal1_type": "rna",
        "modal2_type": modal2_type,
        "integration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "obsm_keys": list(adata_result_integrated.obsm.keys()),
        "uns_keys": list(adata_result_integrated.uns.keys())
    }
    print(f"\n  双模态信息整合完成！结果文件包含所有核心数据，可通过obsm/uns键名快速定位")

    # ---------------------- 原有保存逻辑（仅替换为整合后的数据） ----------------------
    # 1. 数据清理（处理特殊字符，确保h5ad可正常保存）
    adata_to_save = clean_anndata_for_save(adata_result_integrated)
    print(f"\n  数据清理完成，待保存数据形状：{adata_to_save.shape}（位点 × 基因）")

    # 2. 提取细胞类型比例并保存CSV
    cell_type_names = model.ref_adata_dict["basis_matrix"].index.tolist()
    proportion_df = pd.DataFrame(
        adata_to_save.obsm["cell_type_proportions"],
        index=adata_to_save.obs.index,
        columns=cell_type_names
    ).reset_index().rename(columns={"index": "位点名称"})

    proportion_path = os.path.join(base_config["output_dir"], "cell_type_proportions.csv")
    proportion_df.to_csv(proportion_path, index=False, encoding="utf-8-sig")
    print(f"  细胞类型比例CSV：{proportion_path}（{proportion_df.shape[0]}个位点，{len(cell_type_names)}个细胞类型）")

    # 3. 保存完整整合的AnnData结果（包含双模态所有信息）
    adata_path = os.path.join(base_config["output_dir"], "spatial_decon_result.h5ad")
    adata_to_save.write_h5ad(adata_path, compression="gzip")
    print(f"  完整双模态整合结果：{adata_path}（包含RNA+{modal2_type.upper()}所有核心信息）")

    # 4. 保存训练日志（记录双模态整合信息）
    log_path = os.path.join(base_config["output_dir"], "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# 空间多模态解卷积分析日志（一次特征工程+双模态完整整合版）\n")
        f.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=========================================\n\n")

        # 数据信息
        f.write(f"## 1. 数据信息\n")
        f.write(f"- 单细胞参考数据：{os.path.basename(base_config['sc_rna_path'])}\n")
        f.write(f"- 空间RNA数据：{os.path.basename(base_config['st_rna_path'])}\n")
        f.write(f"- 第二模态数据：{os.path.basename(base_config['st_adt_path'])}\n")
        f.write(f"- 第二模态类型：{modal2_type.upper()}\n")
        f.write(f"- 共定位点数量：{len(adata_to_save.obs.index)}\n")
        f.write(f"- RNA信息基因数：{model.smo_adata_dict['modal1'].shape[1]}（降维后{dim_rna}维）\n")
        f.write(f"- 第二模态特征数：{model.smo_adata_dict['modal2'].shape[1]}（降维后{dim_modal2}维）\n")
        f.write(f"- 细胞类型数量：{len(cell_type_names)}\n\n")

        # 核心亮点：一次特征工程+双模态完整整合
        f.write(f"## 2. 核心亮点：一次特征工程+双模态信息完整整合（内部直接处理）\n")
        f.write(f"- 内置函数调用次数：1次（无冗余，高效运行）\n")
        f.write(f"- 双模态信息统一存储：单个h5ad文件包含所有核心数据（feat、降维、特征图、编码器、细胞比例）\n")
        f.write(f"- RNA模态专属数据（obsm）：rna_{obsm_name_rna}、rna_X_pca\n")
        f.write(f"- {modal2_type.upper()}模态专属数据（obsm）：{modal2_type}_{obsm_name_modal2}、{modal2_type}_X_{'lsi' if modal2_type=='peak' else 'pca'}\n")
        f.write(f"- 特征图存储（uns）：rna_feature_graph、{modal2_type}_feature_graph、spatial_graph\n")
        f.write(f"- 附加信息：Peak模态自动记录LSI参数到uns['{modal2_type}_lsi_params']\n\n")

        # 模型参数
        f.write(f"## 3. 模型参数\n")
        for key, val in params.items():
            f.write(f"- {key}：{val}\n")

        # 整合结果明细
        f.write(f"\n## 4. 双模态整合结果明细\n")
        f.write(f"- obsm包含键名（共{len(adata_to_save.obsm.keys())}个）：{list(adata_to_save.obsm.keys())}\n")
        f.write(f"- uns包含键名（共{len(adata_to_save.uns.keys())}个）：{list(adata_to_save.uns.keys())}\n")

    print(f"  训练日志：{log_path}")
    print(f"\n=== 结果保存完成 ===")

# ----------------------
# 主函数：串联全流程（可直接运行）
# ----------------------
if __name__ == "__main__":
    # 1. 全局随机种子初始化（确保结果可复现）
    seed = 1
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. 初始化配置
    base_config = set_base_config()
    params = set_analysis_params()

    # 3. 数据加载与格式展示
    ref_dict, smo_dict, params = load_and_show_data(base_config, params)

    # 4. 模型初始化与内置预处理
    model = init_model_and_preprocess(ref_dict, smo_dict, params)

    # 5. 特征工程与图构建（一次运行，双模态独立键名）
    model, dim_rna, dim_modal2 = feature_engineering_and_graph_build(model, params)

    # 6. 模型训练
    adata_result = train_model(model, params, dim_rna, dim_modal2, base_config)

    # 7. 结果保存（内部直接整合双模态所有信息，无额外函数）
    save_results(adata_result, model, params, base_config, dim_rna, dim_modal2)

    print(f"\n=== 全流程结束（{datetime.now().strftime('%H:%M:%S')}）===")