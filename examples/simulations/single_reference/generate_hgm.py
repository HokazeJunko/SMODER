import os
import random
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import muon as mu

# ====================== 全局配置 ======================
# 路径配置
RAW_DATA_PATH = "path/to/HGM/reference_data"
OUT_PATH = "outputs/single_reference_hgm"
# Output directory is created after parsing command-line arguments.

# 数据文件配置
RNA_REF_FILE = "sc.h5ad"
CELL_TYPE_COL = "Assignment"

# ATAC Peak生成参数
TOTAL_PEAKS = 2000  # 全局peak总数
CORE_PEAKS_PER_CT = 10  # 每个细胞类型核心峰数量
WEAK_PEAKS_PER_CT = 90  # 每个细胞类型弱信号峰数量
CORE_WEIGHT = 1  # 核心峰权重
WEAK_WEIGHT_MIN = 0.05  # 弱峰最小权重
WEAK_WEIGHT_MAX = 0.25  # 弱峰最大权重
RANDOM_SEED = 1234 # 全局固定随机种子
ATAC_NOISE_SCALE = 0.3  # ATAC重尾噪声强度

# 噪声与空间扰动配置
GLOBAL_POISSON_NOISE = {
    "rna": {"enable_noise": True, "poisson_scale": 15, "zero_protection": True, "spatial_gradient_coeff": 3.0},
    "adt": {"enable_noise": True, "poisson_scale": 0, "zero_protection": True, "spatial_gradient_coeff": 3.0},
    "noise_seed": 1234
}

SPACE_DISRUPTION_CONFIG = {
    "extreme_proportion": 0.9,
    "shuffle_target_types": True,
    "cell_count_min": 10,
    "cell_count_max": 20,
    "random_seed": 1234
}

FEATURE_SHUFFLE_CONFIG = {
    "enable_shuffle": True,
    "target_modality": ["rna"],
    "spot_proportion": 1.0,
    "hvg_top_n": 200,
    "hvg_min_disp": 0.5,
    "hvg_dilution_factor": 0.2,
    "shuffle_seed": 1234
}

# 空间采样配置
BASE_SIZE = 12
STRUCTURED_SHAPES = ["squares", "rings", "corners", "checkers"]
CELL_NUMBER_MEAN = [10] * 8
CELL_TYPE_NUMBER = [4, 3, 2, 2, 4, 2, 2, 3]
SECOND_REGIONS = [(40, 40, 60), (200, 40, 60), (40, 200, 60), (200, 200, 60), (120, 120, 60)]


# ====================== 工具函数 ======================
def set_random_seeds(seed: int) -> None:
    """统一设置随机种子"""
    np.random.seed(seed)
    random.seed(seed)
    if 'scipy' in locals():
        import scipy.stats
        scipy.stats.random_state = np.random.RandomState(seed)


def generate_20k_peaks_atac(
        adata_rna,
        cell_type_col,
        total_peaks=2000,
        core_peaks_per_ct=10,
        weak_peaks_per_ct=90,
        core_weight=1000,
        weak_weight_min=50,
        weak_weight_max=250,
        seed=1234,
        noise_scale=0.3
):
    """
    生成2000个ATAC peak，每个细胞类型分配10个核心峰+90个弱峰
    """
    set_random_seeds(seed)

    # 1. 提取RNA基础信息
    X_rna = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
    cell_types = adata_rna.obs[cell_type_col].values
    ct_list = np.unique(cell_types)
    n_cell_types = len(ct_list)
    n_genes = X_rna.shape[1]
    print(f"基础信息：{n_cell_types}个细胞类型 | {n_genes}个基因 | {total_peaks}个Peak")

    # 2. 计算细胞类型平均RNA表达
    rna_ct_mean = np.vstack([
        X_rna[cell_types == ct].mean(axis=0)
        for ct in ct_list
    ])

    # 3. 初始化基因-Peak关联矩阵
    beta_cut = np.zeros((n_genes, total_peaks), dtype=np.float32)
    peak_names = [f"ADT_{i + 1}" for i in range(total_peaks)]

    # 4. 为每个细胞类型划分Peak区间
    base_peaks_per_ct = total_peaks // n_cell_types
    ct_peak_ranges = {}
    for ct_idx, ct in enumerate(ct_list):
        start = ct_idx * base_peaks_per_ct
        end = (ct_idx + 1) * base_peaks_per_ct if ct_idx < n_cell_types - 1 else total_peaks
        ct_peak_ranges[ct] = (start, end)
        print(f"细胞类型{ct}：基础Peak区间 [{start}, {end})")

    # 5. 为每个细胞类型分配核心峰和弱峰
    ct_peak_assignment = {}
    for ct_idx, ct in enumerate(ct_list):
        # 获取细胞类型专属Peak区间
        ct_peak_start, ct_peak_end = ct_peak_ranges[ct]
        ct_all_peaks = np.arange(ct_peak_start, ct_peak_end)

        # 选择核心峰和弱峰
        core_peaks = np.random.choice(ct_all_peaks, size=core_peaks_per_ct, replace=False)
        global_all_peaks = np.arange(total_peaks)
        non_core_peaks = global_all_peaks[~np.isin(global_all_peaks, core_peaks)]
        weak_peaks = np.random.choice(non_core_peaks, size=weak_peaks_per_ct, replace=False)

        ct_peak_assignment[ct] = {
            "core_peaks": core_peaks,
            "weak_peaks": weak_peaks,
            "core_count": len(core_peaks),
            "weak_count": len(weak_peaks)
        }
        print(f"细胞类型{ct}：{len(core_peaks)}个核心峰 | {len(weak_peaks)}个弱峰")

        # 6. 填充关联矩阵权重
        # 核心峰：固定高权重
        for peak_idx in core_peaks:
            core_gene_idx = np.random.choice(n_genes, 1)[0]
            beta_cut[core_gene_idx, peak_idx] = core_weight

        # 弱峰：随机低权重
        for peak_idx in weak_peaks:
            weak_gene_idx = np.random.choice(n_genes, 1)[0]
            beta_cut[weak_gene_idx, peak_idx] = np.random.uniform(weak_weight_min, weak_weight_max)

    # 7. 计算细胞类型级ATAC信号并添加重尾噪声
    print("计算细胞类型ADT信号...")
    atac_ct_mean = rna_ct_mean @ beta_cut
    atac_ct_mean += t.rvs(df=3, size=atac_ct_mean.shape, random_state=seed) * noise_scale
    atac_ct_mean = np.clip(atac_ct_mean, 0, None)

    # 8. 扩展到单细胞水平
    print("扩展到单细胞ADT信号...")
    cell_ct_mapping = np.array([np.where(ct_list == ct)[0][0] for ct in cell_types])
    X_atac = atac_ct_mean[cell_ct_mapping]

    # 添加单细胞水平微小噪声
    set_random_seeds(seed + 100)
    X_atac += np.random.normal(scale=0.01, size=X_atac.shape)
    X_atac = np.clip(X_atac, 0, None)

    # 9. 构建ATAC AnnData对象
    adata_atac = ad.AnnData(
        X=scipy.sparse.csr_matrix(X_atac.astype(np.float32)),
        obs=adata_rna.obs.copy(),
        var=pd.DataFrame(index=peak_names)
    )

    # 10. 保存关键信息到uns
    adata_atac.uns.update({
        "beta_cut": beta_cut,
        "peak_assignment": ct_peak_assignment,
        "cell_types": ct_list.tolist(),
        "seed": seed,
        "params": {
            "total_peaks": total_peaks,
            "core_peaks_per_ct": core_peaks_per_ct,
            "weak_peaks_per_ct": weak_peaks_per_ct,
            "core_weight": core_weight,
            "weak_weight_range": (weak_weight_min, weak_weight_max)
        }
    })

    # 输出统计信息
    print(f"\n✅ ADT数据生成完成：")
    print(f"   - beta_cut矩阵形状：{beta_cut.shape}")
    print(f"   - 非零权重数：{np.count_nonzero(beta_cut)}")
    print(f"   - 核心峰权重：{core_weight}")
    print(f"   - 弱峰权重范围：[{weak_weight_min}, {weak_weight_max}]")
    print(f"   - ADT值范围：[{adata_atac.X.min():.4f}, {adata_atac.X.max():.4f}]")

    return adata_atac, beta_cut, ct_peak_assignment


# ====================== 主流程 ======================

def parse_args():
    """Parse command-line arguments for the public example script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a single-reference simulated spatial multi-omics dataset."
    )
    parser.add_argument(
        "--raw-data-path",
        default=RAW_DATA_PATH,
        help="Directory containing the input single-cell RNA reference file.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT_PATH,
        help="Directory for generated simulated output files.",
    )
    parser.add_argument(
        "--rna-ref-file",
        default=RNA_REF_FILE,
        help="Single-cell RNA reference .h5ad filename inside --raw-data-path.",
    )
    parser.add_argument(
        "--cell-type-col",
        default=CELL_TYPE_COL,
        help="Column in the single-cell RNA reference .obs that stores cell-type labels.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    RAW_DATA_PATH = args.raw_data_path
    OUT_PATH = args.output_dir
    RNA_REF_FILE = args.rna_ref_file
    CELL_TYPE_COL = args.cell_type_col
    os.makedirs(OUT_PATH, exist_ok=True)

    # Step 1: 加载RNA参考数据
    print("=" * 80)
    print("Step 1/6: 加载RNA参考数据")
    print("=" * 80)
    adata_rna = sc.read_h5ad(os.path.join(RAW_DATA_PATH, RNA_REF_FILE))
    if 'counts' in adata_rna.layers:
        adata_rna.X = adata_rna.layers['counts'].copy()
    adata_rna.obs[CELL_TYPE_COL] = adata_rna.obs[CELL_TYPE_COL].astype('category')
    print(f"✅ RNA参考数据：{adata_rna.n_obs}细胞 × {adata_rna.n_vars}基因")

    # Step 2: 生成2000 Peak的ATAC参考数据
    print("\n" + "=" * 80)
    print("Step 2/6: 生成2000个ADT Peak（10核心+90弱峰/细胞类型）")
    print("=" * 80)
    adata_atac, beta_cut, peak_assignment = generate_20k_peaks_atac(
        adata_rna=adata_rna,
        cell_type_col=CELL_TYPE_COL,
        total_peaks=TOTAL_PEAKS,
        core_peaks_per_ct=CORE_PEAKS_PER_CT,
        weak_peaks_per_ct=WEAK_PEAKS_PER_CT,
        core_weight=CORE_WEIGHT,
        weak_weight_min=WEAK_WEIGHT_MIN,
        weak_weight_max=WEAK_WEIGHT_MAX,
        seed=RANDOM_SEED,
        noise_scale=ATAC_NOISE_SCALE
    )

    # Step 3: 构建双模态MuData
    print("\n" + "=" * 80)
    print("Step 3/6: 构建双模态参考MuData")
    print("=" * 80)
    mdata_ref = mu.MuData({"rna": adata_rna, "adt": adata_atac})
    print(f"✅ 双模态参考数据：{mdata_ref}")

    import simulate_new as sm
    # Step 4: 计算高变特征
    print("\n" + "=" * 80)
    print("Step 4/6: 计算细胞类型高变特征")
    print("=" * 80)
    cell_type_hvgs_dict = {}
    for mod_name in mdata_ref.mod.keys():
        print(f"\n计算{mod_name}高变特征：")
        cell_type_hvgs_dict[mod_name] = sm.compute_cell_type_hvgs(
            mdata_ref[mod_name].copy(),
            cell_type_col=CELL_TYPE_COL,
            top_n=FEATURE_SHUFFLE_CONFIG["hvg_top_n"],
            min_disp=FEATURE_SHUFFLE_CONFIG["hvg_min_disp"]
        )

    # Step 5: 空间采样（第一轮+第二轮）
    print("\n" + "=" * 80)
    print("Step 5/6: 双模态空间数据采样")
    print("=" * 80)
    # 导入采样模块
    import simulate_new as sm

    sim_params = {
        "cell_type_key": CELL_TYPE_COL,
        "num_spots": (BASE_SIZE * 2) ** 2,
        "balance": "unbalanced",
        "cell_number_mean": CELL_NUMBER_MEAN,
        "cell_number_nu": 25.0,
        "cell_type_number": CELL_TYPE_NUMBER,
        "poisson_noise_scale": 0.0,
        "structured_shapes": STRUCTURED_SHAPES,
        "structured_base_size": BASE_SIZE,
        "random_seed": 42  # 固定第一轮种子
    }

    # 第一轮采样（完全固定）
    simulation_mdata, sampled_cells_df_first = sm.generate_spatial_data(
        reference=mdata_ref,
        **sim_params
    )
    sampled_cells_df_first['sampling_round'] = 1
    print(f"✅ 第一轮采样完成：{simulation_mdata.n_obs}个spots")

    # 第二轮空间破坏（参考代码逻辑）
    ref_mod = list(simulation_mdata.mod.keys())[0]
    spatial_coords = simulation_mdata[ref_mod].obsm['spatial'].copy()
    spot_names = simulation_mdata[ref_mod].obs_names.tolist()

    # 定义所有细胞类型
    all_cell_types = mdata_ref[ref_mod].obs[CELL_TYPE_COL].cat.categories

    # 筛选目标位点（函数已内置排序）
    target_spots = sm.get_spots_in_regions(spatial_coords, SECOND_REGIONS, spot_names)
    target_indices = [spot_names.index(name) for name in target_spots]
    n_target = len(target_spots)
    print(f"\n🎯 筛选出{n_target}个第二轮目标spots")

    if n_target > 0:
        main_seed = SPACE_DISRUPTION_CONFIG["random_seed"]
        # 参考代码：直接设置主种子
        np.random.seed(main_seed)
        random.seed(main_seed)
        print(f"🔢 第二轮空间破坏种子已固定：{main_seed}")

        # 清空目标区域数据
        for mod_name in simulation_mdata.mod.keys():
            adata = simulation_mdata[mod_name]
            adata_X_lil = adata.X.tolil()
            adata_X_lil[target_indices] = 0
            adata.X = adata_X_lil.tocsr()
            adata.obs.loc[target_spots, 'cell_count'] = 0
            if 'proportions' in adata.obsm:
                adata.obsm['proportions'][target_indices] = 0

        # 细胞类型-细胞映射
        cell_type_to_cells = {
            ct: mdata_ref[ref_mod].obs.index[mdata_ref[ref_mod].obs[CELL_TYPE_COL] == ct].tolist()
            for ct in all_cell_types
        }

        # 参考代码：打乱细胞类型单独设种子
        if SPACE_DISRUPTION_CONFIG["shuffle_target_types"]:
            np.random.seed(main_seed + 500)  # 固定偏移500
            random_cell_types = np.random.permutation(all_cell_types)
        else:
            random_cell_types = all_cell_types

        sampled_cells_df_second = []
        for spot_order_idx, spot_idx in enumerate(target_indices):
            spot_name = target_spots[spot_order_idx]
            sub_seed = main_seed + spot_order_idx + 1000
            # 参考代码：直接重置子种子
            np.random.seed(sub_seed)
            random.seed(sub_seed)

            # 随机细胞数
            cell_count = np.random.randint(
                SPACE_DISRUPTION_CONFIG["cell_count_min"],
                SPACE_DISRUPTION_CONFIG["cell_count_max"] + 1
            )

            # 极端比例采样
            dominant_ct = np.random.choice(random_cell_types)
            dominant_count = int(cell_count * SPACE_DISRUPTION_CONFIG["extreme_proportion"])
            other_cts = [ct for ct in random_cell_types if ct != dominant_ct]
            other_count = cell_count - dominant_count
            other_selected = np.random.choice(other_cts, size=other_count, replace=True)
            selected_types = [dominant_ct] * dominant_count + list(other_selected)
            np.random.shuffle(selected_types)

            # 采样细胞
            sampled_cells = []
            for ct in set(selected_types):
                ct_count = selected_types.count(ct)
                available_cells = cell_type_to_cells[ct]
                sampled_ct_cells = np.random.choice(
                    available_cells,
                    size=ct_count,
                    replace=len(available_cells) < ct_count
                )
                sampled_cells.extend(sampled_ct_cells)
            sampled_cells = list(set(sampled_cells))

            # 填充各模态表达值（参考代码逻辑）
            for mod_name in simulation_mdata.mod.keys():
                adata_sim = simulation_mdata[mod_name]
                adata_ref = mdata_ref[mod_name]
                exp_matrix = adata_ref[sampled_cells, :].X
                exp_sum = exp_matrix.sum(axis=0).A1 if scipy.sparse.issparse(exp_matrix) else exp_matrix.sum(axis=0)
                exp_sum_2d = exp_sum.reshape(1, -1)

                adata_X_lil = adata_sim.X.tolil()
                adata_X_lil[spot_idx] = exp_sum_2d
                adata_sim.X = adata_X_lil.tocsr()

            # 更新元数据（参考代码逻辑）
            cell_type_counts = pd.Series(selected_types).value_counts()
            proportions = np.zeros(len(all_cell_types))
            for ct in cell_type_counts.index:
                if ct in all_cell_types:
                    proportions[all_cell_types.get_loc(ct)] = cell_type_counts[ct] / cell_count

            dominant_type = cell_type_counts.index[0] if not cell_type_counts.empty else "Unknown"
            simulation_mdata[ref_mod].obs.loc[spot_name, 'cell_count'] = cell_count
            simulation_mdata[ref_mod].obs.loc[spot_name, 'dominant_cell_type'] = dominant_type
            if 'proportions' in simulation_mdata[ref_mod].obsm:
                simulation_mdata[ref_mod].obsm['proportions'][spot_idx] = proportions

            # 记录采样信息
            sampled_cells_df_second.append({
                "spot_name": spot_name, "sampling_round": 2, "main_seed": main_seed, "sub_seed": sub_seed,
                "cell_count": cell_count, "dominant_cell_type": dominant_type,
                "selected_types": list(set(selected_types)), "cell_type_proportions": proportions.tolist(),
                "sampled_cells": sampled_cells
            })

        # 标记第二轮spots
        simulation_mdata[ref_mod].obs['second_sampling'] = False
        simulation_mdata[ref_mod].obs.loc[target_spots, 'second_sampling'] = True
        sampled_cells_df_second = pd.DataFrame(sampled_cells_df_second)
        sampled_cells_df = pd.concat([sampled_cells_df_first, sampled_cells_df_second], ignore_index=True)
    else:
        proportions = simulation_mdata[ref_mod].obsm['proportions']
        dominant_indices = np.argmax(proportions, axis=1)
        dominant_types = [all_cell_types[i] for i in dominant_indices]
        simulation_mdata[ref_mod].obs['dominant_cell_type'] = dominant_types
        sampled_cells_df = sampled_cells_df_first
        print("⚠️  未找到目标区域spots")

    # Step 6: 噪声添加 + 高变特征稀释 + 保存
    print("\n" + "=" * 80)
    print("Step 6/6: 后处理并保存数据")
    print("=" * 80)
    # 添加空间梯度噪声
    simulation_mdata = sm.add_spatial_gradient_poisson_noise(simulation_mdata, GLOBAL_POISSON_NOISE)
    # 高变特征稀释
    simulation_mdata = sm.shuffle_cell_type_hvgs(simulation_mdata, FEATURE_SHUFFLE_CONFIG, cell_type_hvgs_dict)

    # 保存数据
    # 1. 双模态MuData
    h5mu_file = os.path.join(OUT_PATH, 'simulation_rna_atac.h5mu')
    mu.write_h5mu(h5mu_file, simulation_mdata)
    print(f"✅ 双模态数据：{h5mu_file}")

    # 2. 单独模态文件
    for mod_name in simulation_mdata.mod.keys():
        mod_file = os.path.join(OUT_PATH, f'simulation_{mod_name}.h5ad')
        simulation_mdata[mod_name].write_h5ad(mod_file)
        print(f"✅ {mod_name}数据：{mod_file}")

    # 3. 采样信息
    sampled_cells_df.to_csv(os.path.join(OUT_PATH, 'sampled_cells_info.csv'), index=False)

    # 4. 高变特征信息
    hvgs_df = pd.DataFrame([
        {'modality': mod, 'cell_type': ct, 'hvgs': ','.join(hvgs)}
        for mod, ct_hvgs in cell_type_hvgs_dict.items() for ct, hvgs in ct_hvgs.items()
    ])
    hvgs_df.to_csv(os.path.join(OUT_PATH, 'cell_type_hvgs_info.csv'), index=False)

    print("\n" + "=" * 100)
    print("📋 2000 ADT双模态空间数据模拟完成！")
    print(f"输出目录：{OUT_PATH}")
    print("=" * 100)