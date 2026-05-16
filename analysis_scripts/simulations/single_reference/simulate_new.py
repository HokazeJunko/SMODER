import math
import random
from typing import Optional, List, Union
import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scipy
import scanpy as sc


# ====================== 基础空间图案生成函数 ======================
def conway_maxwell_poisson(lambda_: int, nu: float, seed: Optional[int] = None) -> int:
    """
    生成Conway-Maxwell-Poisson分布的随机数（用于模拟每个位点的细胞数量）

    参数:
        lambda_: 分布的均值参数（整数）
        nu: 分布的离散参数（浮点数）
        seed: 随机种子，保证结果可复现

    返回:
        符合CMP分布的随机整数
    """
    # 关键修复：使用局部随机状态，避免污染全局种子
    local_rng = np.random.RandomState(seed) if seed is not None else np.random
    local_random = random.Random(seed) if seed is not None else random

    lambda_ = int(lambda_)
    nu = float(nu)

    # 计算归一化常数C（截断到1000项保证计算效率）
    C = np.sum([(pow(lambda_, k) / math.factorial(k)) ** nu for k in range(1000)])

    # 逆变换采样
    u, sum_p, k = local_rng.rand(), 0, 0
    while sum_p < u:
        sum_p += (pow(lambda_, k) / math.factorial(k)) ** nu / C
        k += 1
    return k - 1


def squares(base_size: int = 12) -> np.ndarray:
    """生成四个方块图案（左上、右上、左下、右下各一个方块）"""
    A = np.zeros([base_size, base_size])
    offset = base_size // 6
    size = base_size // 3
    A[offset:offset + size, offset:offset + size] = 1
    A[offset + size * 2:offset + size * 3, offset:offset + size] = 1
    A[offset:offset + size, offset + size * 2:offset + size * 3] = 1
    A[offset + size * 2:offset + size * 3, offset + size * 2:offset + size * 3] = 1
    return A


def corners(base_size: int = 12) -> np.ndarray:
    """生成四角填充的图案"""
    B = np.zeros([base_size // 2, base_size // 2])
    for i in range(base_size // 2):
        B[i, i:] = 1
    A = np.flip(B, axis=1)
    AB = np.hstack((A, B))
    CD = np.flip(AB, axis=0)
    return np.vstack((AB, CD))


def scotland(base_size: int = 12) -> np.ndarray:
    """生成苏格兰旗样式的交叉对角线图案"""
    A = np.eye(base_size)
    for i in range(base_size):
        A[-i - 1, i] = 1
    return A


def checkers(base_size: int = 12) -> np.ndarray:
    """生成棋盘格图案"""
    unit = base_size // 3
    A = np.zeros([unit, unit])
    B = np.ones([unit, unit])
    AB = np.hstack((A, B, A))
    BA = np.hstack((B, A, B))
    return np.vstack((AB, BA, AB))


def rings(base_size: int = 12) -> np.ndarray:
    """生成环形图案"""
    A = np.zeros([base_size, base_size])
    center = base_size // 2
    inner_radius = base_size // 4
    outer_radius = base_size // 2 - 1
    y, x = np.ogrid[:base_size, :base_size]
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    A[(dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)] = 1
    return A


def gen_spatial_factors(
        shapes: List[str] = ["squares", "corners", "scotland", "checkers"],
        base_size: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成空间区域矩阵（8个Region，4大区域×0/1）

    参数:
        shapes: 4种形状列表（顺序：左上、右上、左下、右下）
        base_size: 基础图案尺寸

    返回:
        F: 空间因子矩阵 (total_size², 8)
        full_pattern: 完整图案矩阵 (2*base_size, 2*base_size)
        region_matrix: Region ID矩阵 (2*base_size, 2*base_size)
    """
    shape_funcs = {
        "squares": squares,
        "corners": corners,
        "scotland": scotland,
        "checkers": checkers,
        "rings": rings
    }
    assert len(shapes) == 4, "必须指定4种形状（顺序：左上、右上、左下、右下）"
    for shape in shapes:
        assert shape in shape_funcs.keys(), f"不支持的形状：{shape}，可选：{list(shape_funcs.keys())}"

    # 生成4个大区域的结构化矩阵
    shape1 = shape_funcs[shapes[0]](base_size=base_size)  # 左上
    shape2 = shape_funcs[shapes[1]](base_size=base_size)  # 右上
    shape3 = shape_funcs[shapes[2]](base_size=base_size)  # 左下
    shape4 = shape_funcs[shapes[3]](base_size=base_size)  # 右下

    # 大区域内0/1分配独立Region ID
    total_size = base_size * 2
    region_matrix = np.zeros((total_size, total_size), dtype=int)

    # 区域映射规则：左上(0:4,1:5)、右上(0:6,1:7)、左下(0:0,1:1)、右下(0:2,1:3)
    region_mapping = {
        (0, 0): 4, (0, 1): 5,  # 左上
        (1, 0): 6, (1, 1): 7,  # 右上
        (2, 0): 0, (2, 1): 1,  # 左下
        (3, 0): 2, (3, 1): 3  # 右下
    }

    # 为每个大区域的像素分配Region ID
    quadrants = [
        (shape1, 0, 0),  # 左上：矩阵、x偏移、y偏移
        (shape2, 0, base_size),  # 右上
        (shape3, base_size, 0),  # 左下
        (shape4, base_size, base_size)  # 右下
    ]

    for q_idx, (q_matrix, x_offset, y_offset) in enumerate(quadrants):
        for i in range(base_size):
            for j in range(base_size):
                global_i = x_offset + i
                global_j = y_offset + j
                pixel_val = int(q_matrix[i, j])  # 0或1
                region_matrix[global_i, global_j] = region_mapping[(q_idx, pixel_val)]

    # 拼接完整图案
    top_row = np.hstack((shape1, shape2))
    bottom_row = np.hstack((shape3, shape4))
    full_pattern = np.vstack((top_row, bottom_row))

    # 生成F矩阵（兼容原有架构）
    F = np.zeros((total_size * total_size, 8))  # 8个Region
    for r_id in range(8):
        mask = region_matrix.flatten() == r_id
        F[mask, r_id] = 1

    return F, full_pattern, region_matrix


# ====================== 采样器类 ======================
class Sampler:
    def __init__(
            self,
            reference: Union[mu.MuData, ad.AnnData],
            cell_type_key: str,
            num_spots: int,
            cell_number_mean: Union[int, list] = [6, 8, 6, 8, 7, 9, 7, 9],
            cell_number_nu: Union[float, list] = 20.0,
            cell_type_number: Union[int, list] = [4, 4, 4, 4, 4, 4, 4, 4],
            balance: Optional[str] = "balanced",
            poisson_noise_scale: float = 1.0,
            structured_shapes: List[str] = ["squares", "corners", "scotland", "checkers"],
            structured_base_size: int = 12,
            random_seed: int = 1234
    ):
        self.reference = reference
        self.cell_type_key = cell_type_key
        self.num_spots = num_spots
        self.obs = reference.obs if isinstance(reference, ad.AnnData) else reference[list(reference.mod.keys())[0]].obs
        self.poisson_noise_scale = poisson_noise_scale
        self.structured_shapes = structured_shapes
        self.structured_base_size = structured_base_size
        self.n_regions = 8
        self.random_seed = random_seed

        # 校验总位点数
        expected_num_spots = (structured_base_size * 2) ** 2
        assert self.num_spots == expected_num_spots, \
            f"总位点数必须为(base_size×2)²={expected_num_spots}，当前输入：{self.num_spots}"

        # 标准化8个Region的参数
        if isinstance(cell_number_mean, int):
            self.cell_number_mean = np.ones(8, dtype=int) * cell_number_mean
        else:
            assert len(cell_number_mean) == 8, "必须指定8个Region的cell_number_mean"
            self.cell_number_mean = np.array(cell_number_mean)

        if isinstance(cell_number_nu, float):
            self.cell_number_nu = np.ones(8) * cell_number_nu
        else:
            assert len(cell_number_nu) == 8, "必须指定8个Region的cell_number_nu"
            self.cell_number_nu = np.array(cell_number_nu)

        if isinstance(cell_type_number, int):
            self.cell_type_number = np.ones(8, dtype=int) * cell_type_number
        else:
            assert len(cell_type_number) == 8, "必须指定8个Region的cell_type_number"
            self.cell_type_number = np.array(cell_type_number)

        if balance not in ["balanced", "unbalanced"]:
            raise ValueError('balance must be one of ["balanced", "unbalanced"].')
        self.init_sample_prob(balance=balance)

    def init_sample_prob(self, balance="unbalanced"):
        """初始化细胞类型采样概率"""
        cell_counts = self.obs[self.cell_type_key].value_counts(normalize=True)
        if balance == "unbalanced":
            self.cluster_p = cell_counts
            self.cell_p = self.obs[self.cell_type_key].map(self.cluster_p).astype(float)
        elif balance == "balanced":
            self.cluster_p = pd.Series(1 / len(cell_counts), index=cell_counts.index)
            self.cell_p = 1 / self.obs[self.cell_type_key].map(cell_counts).astype(float) / len(cell_counts)
        self.clusters = self.obs[self.cell_type_key].cat.categories
        self.cluster_p = self.cluster_p[self.clusters]

    def define_regions(self):
        """生成8个Region ID矩阵"""
        F, full_pattern, region_matrix = gen_spatial_factors(
            shapes=self.structured_shapes,
            base_size=self.structured_base_size
        )
        self.regions = region_matrix.flatten()
        self.region_matrix = region_matrix

    def sample_data(self):
        """核心采样逻辑：按Region分配细胞类型和数量"""
        # 固定全局种子
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # 打乱细胞类型并按Region均分（固定种子）
        all_clusters = self.cluster_p.index.tolist()
        np.random.shuffle(all_clusters)

        used_clusters = {}
        current_idx = 0
        for region_id in range(self.n_regions):
            n_needed = self.cell_type_number[region_id]
            if current_idx + n_needed <= len(all_clusters):
                selected = all_clusters[current_idx:current_idx + n_needed]
                current_idx += n_needed
            else:
                remaining = len(all_clusters) - current_idx
                selected = all_clusters[current_idx:] + all_clusters[:n_needed - remaining]
                current_idx = n_needed - remaining
            used_clusters[region_id] = np.array(selected)

        # 生成Region ID
        self.define_regions()

        # 为每个位点生成细胞数（位点级种子隔离）
        cell_count = []
        for idx, region_id in enumerate(self.regions):
            spot_seed = self.random_seed + idx
            cell_num = conway_maxwell_poisson(
                self.cell_number_mean[region_id],
                self.cell_number_nu[region_id],
                seed=spot_seed
            )
            cell_count.append(max(cell_num, 1))
        cell_count = np.array(cell_count)

        # 打印Region抽样信息
        region_names = {
            0: "左下-0", 1: "左下-1", 2: "右下-0", 3: "右下-1",
            4: "左上-0", 5: "左上-1", 6: "右上-0", 7: "右上-1"
        }
        print("=" * 60)
        for region_id in range(self.n_regions):
            mask = self.regions == region_id
            if mask.sum() > 0:
                mean_cells = cell_count[mask].mean()
                print(
                    f"Region {region_id} ({region_names[region_id]}): "
                    f"位点数量={mask.sum()}, 抽样类型={used_clusters[region_id]}, "
                    f"平均细胞数={mean_cells:.2f}"
                )
        print("=" * 60)

        # 整理采样参数
        used_clusters_list = [used_clusters[region_id] for region_id in self.regions]
        params = list(zip(cell_count, used_clusters_list, self.regions))
        return self.sample_spots(params)

    def sample_spots(self, params):
        """采样每个spot的细胞并计算表达量"""
        sample_exp = {"tmp": self.reference} if isinstance(self.reference, ad.AnnData) else self.reference.mod
        exp = {key: np.zeros((len(params), adata.shape[1])) for key, adata in sample_exp.items()}
        density = np.zeros((len(params), len(self.clusters)))
        sampled_cells_df = []

        for i, (num_cell, used_clusters, region_id) in enumerate(params):
            # 位点级独立种子（完全隔离）
            spot_seed = self.random_seed + i
            local_rng = np.random.RandomState(spot_seed)

            # 筛选细胞类型并计算采样概率
            cluster_mask = self.obs[self.cell_type_key].isin(used_clusters).values
            p = self.cell_p[cluster_mask] / self.cell_p[cluster_mask].sum()

            # 采样细胞（使用局部随机数生成器）
            sampled_cells = local_rng.choice(
                self.obs.index[cluster_mask],
                size=num_cell,
                p=p,
            )

            # 计算表达量
            for key, adata in sample_exp.items():
                if scipy.sparse.issparse(adata.X):
                    exp[key][i, :] = adata[sampled_cells, :].X.sum(axis=0).A1
                else:
                    exp[key][i, :] = adata[sampled_cells, :].X.sum(axis=0)

            # 计算细胞类型密度
            density[i, :] = self.obs.loc[sampled_cells, self.cell_type_key].value_counts().reindex(self.clusters,
                                                                                                   fill_value=0).values

            # 记录采样信息
            sampled_cells_df.append(
                {
                    "spot_name": f"spot_{i}",  # 新增：固定spot名称
                    "region_id": region_id,
                    "cell_count": num_cell,
                    "cell_ids": sampled_cells.tolist(),
                    "cell_types": self.obs.loc[sampled_cells, self.cell_type_key].values.tolist(),
                }
            )

        # 添加泊松噪声
        for key in exp.keys():
            exp[key] = add_poisson_noise(exp[key], self.poisson_noise_scale, seed=self.random_seed)

        return exp, density, pd.DataFrame(sampled_cells_df)

    def get_coords(self):
        """生成空间坐标"""
        grid_size = int(np.sqrt(self.num_spots))
        x = np.arange(0, grid_size)
        y = np.arange(0, grid_size)
        X, Y = np.meshgrid(x, y)
        return X, Y


# ====================== 噪声和数据处理函数 ======================
def add_poisson_noise(data, noise_scale=1.0, seed=None):
    """基础泊松噪声添加函数（修复稀疏矩阵处理）"""
    local_rng = np.random.RandomState(seed) if seed is not None else np.random

    if scipy.sparse.issparse(data):
        data_arr = data.toarray()
    else:
        data_arr = np.array(data)

    # 避免零值导致lambda为0
    lambda_ = np.maximum(data_arr * noise_scale, 1e-6)
    noise = local_rng.poisson(lambda_)
    noisy_data = data_arr + noise
    noisy_data = np.maximum(noisy_data, 0).astype(int)

    if scipy.sparse.issparse(data):
        return scipy.sparse.csr_matrix(noisy_data)
    else:
        return noisy_data


def add_spatial_gradient_poisson_noise(mdata: mu.MuData, noise_config: dict) -> mu.MuData:
    """
    对多模态数据添加空间梯度式泊松噪声（越靠近中心噪声越小，边缘噪声越大）

    参数:
        mdata: MuData对象
        noise_config: 全局噪声配置字典

    返回:
        加噪后的MuData对象
    """
    main_seed = noise_config["noise_seed"]
    ref_mod = list(mdata.mod.keys())[0]

    # 计算每个spot到空间中心的距离
    spatial_coords = mdata[ref_mod].obsm['spatial'].copy()
    center = spatial_coords.mean(axis=0)
    distances = np.sqrt(np.sum((spatial_coords - center) ** 2, axis=1))
    dist_max = distances.max() if distances.max() > 0 else 1
    normalized_dist = distances / dist_max

    # 逐模态添加噪声
    for mod_name in mdata.mod.keys():
        if mod_name not in noise_config:
            continue
        cfg = noise_config[mod_name]

        if not cfg["enable_noise"] or cfg["poisson_scale"] == 0.0:
            print(f"   - {mod_name}：未添加泊松噪声")
            continue

        # 模态专属种子
        mod_seed = main_seed + hash(mod_name) % 1000
        local_rng = np.random.RandomState(mod_seed)

        adata = mdata[mod_name]
        exp_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()

        # 计算每个spot的噪声强度（梯度）
        gradient_coeff = cfg.get("spatial_gradient_coeff", 3.0)
        spot_noise_scales = cfg["poisson_scale"] * (1 + normalized_dist * (gradient_coeff - 1))

        # 逐spot生成噪声（零值保护）
        non_zero_mask = (exp_matrix > 0)
        noise = np.zeros_like(exp_matrix)

        for spot_idx in range(exp_matrix.shape[0]):
            current_scale = spot_noise_scales[spot_idx]
            if current_scale <= 0:
                continue
            spot_noise = local_rng.poisson(current_scale, size=exp_matrix.shape[1])
            spot_noise[~non_zero_mask[spot_idx]] = 0
            noise[spot_idx] = spot_noise

        # 叠加噪声
        exp_noisy = exp_matrix + noise
        mdata[mod_name].X = scipy.sparse.csr_matrix(np.maximum(exp_noisy, 0))

        # 输出统计信息
        noisy_pixels = np.sum(non_zero_mask)
        total_pixels = exp_matrix.size
        avg_center_noise = cfg["poisson_scale"]
        avg_edge_noise = cfg["poisson_scale"] * gradient_coeff
        print(f"   - {mod_name}：已添加空间梯度式泊松噪声")
        print(f"     ✅ 仅对原始值不为0的区域加噪")
        print(f"     📍 中心噪声强度：{avg_center_noise} | 边缘噪声强度：{avg_edge_noise}")
        print(f"     📊 加噪像素数：{noisy_pixels} / {total_pixels} ({noisy_pixels / total_pixels:.2%})")

    return mdata


def compute_cell_type_hvgs(adata_ref: ad.AnnData, cell_type_col: str, top_n=200, min_disp=0.5) -> dict:
    """
    按细胞类型计算高变基因

    参数:
        adata_ref: 参考Anndata对象
        cell_type_col: 细胞类型列名
        top_n: 每个细胞类型取top N高变基因
        min_disp: 最小离散度阈值

    返回:
        字典 {细胞类型: [高变基因列表]}
    """
    cell_type_hvgs = {}
    cell_types = adata_ref.obs[cell_type_col].cat.categories

    for ct in cell_types:
        adata_ct = adata_ref[adata_ref.obs[cell_type_col] == ct].copy()

        # 计算高变基因
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
        sc.pp.highly_variable_genes(
            adata_ct,
            min_mean=0.0125,
            max_mean=3,
            min_disp=min_disp,
            n_top_genes=top_n
        )

        # 获取高变基因列表
        hvgs = adata_ct.var_names[adata_ct.var.highly_variable].tolist()
        cell_type_hvgs[ct] = hvgs
        print(f"   - {ct}：识别出 {len(hvgs)} 个高变基因")

    return cell_type_hvgs


def shuffle_cell_type_hvgs(mdata: mu.MuData, shuffle_config: dict, cell_type_hvgs: dict,
                           dominant_cell_type_col="dominant_cell_type") -> mu.MuData:
    """
    针对每个位点的主导细胞类型，稀释高变基因表达值

    参数:
        mdata: MuData对象
        shuffle_config: 稀释配置字典
        cell_type_hvgs: 细胞类型-高变基因字典
        dominant_cell_type_col: 主导细胞类型列名

    返回:
        处理后的MuData对象
    """
    if not shuffle_config["enable_shuffle"]:
        return mdata

    # 支持单/多模态
    target_mods = shuffle_config["target_modality"]
    if isinstance(target_mods, str):
        target_mods = [target_mods]

    # 固定种子
    local_rng = np.random.RandomState(shuffle_config["shuffle_seed"])
    dilution_factor = shuffle_config["hvg_dilution_factor"]

    for mod_name in target_mods:
        if mod_name not in mdata.mod:
            print(f"警告：模态 {mod_name} 不存在，跳过特征稀释")
            continue

        adata = mdata[mod_name]
        exp_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()
        n_spots, n_features = exp_matrix.shape
        feature_names = adata.var_names.tolist()

        # 选择目标位点（固定种子）
        n_shuffle_spots = int(n_spots * shuffle_config["spot_proportion"])
        shuffle_spot_idx = local_rng.choice(n_spots, n_shuffle_spots, replace=False)

        # 获取主导细胞类型
        ref_mod = list(mdata.mod.keys())[0]
        dominant_cell_types = mdata[ref_mod].obs[dominant_cell_type_col].values

        total_diluted = 0
        # 逐位点处理
        for spot_idx in shuffle_spot_idx:
            dom_ct = dominant_cell_types[spot_idx]

            # 跳过无高变基因的细胞类型
            if dom_ct not in cell_type_hvgs or len(cell_type_hvgs[dom_ct]) == 0:
                continue

            # 选择高变基因并稀释（固定种子）
            hvgs = cell_type_hvgs[dom_ct]
            n_dilute = int(len(hvgs) * 0.8)  # 随机选80%高变基因
            selected_hvgs = local_rng.choice(hvgs, size=n_dilute, replace=False).tolist()
            hvg_indices = [feature_names.index(hvg) for hvg in selected_hvgs if hvg in feature_names]

            if not hvg_indices:
                continue

            # 核心：稀释表达值
            exp_matrix[spot_idx, hvg_indices] = exp_matrix[spot_idx, hvg_indices] * dilution_factor
            total_diluted += len(hvg_indices)

        # 赋值回去
        mdata[mod_name].X = scipy.sparse.csr_matrix(np.maximum(exp_matrix, 0))
        print(f"   - {mod_name}：完成高变基因稀释（共稀释 {total_diluted} 个特征值，涉及 {len(shuffle_spot_idx)} 个位点）")
        print(
            f"   - 稀释策略：每个细胞类型top {shuffle_config['hvg_top_n']}高变基因中随机选择80%，稀释到原始值的{dilution_factor * 100}%")

    return mdata


def get_spots_in_regions(coords: np.ndarray, regions: list, spot_names: list) -> list:
    """
    根据空间坐标筛选目标区域的spots（固定顺序）

    参数:
        coords: 所有spot的空间坐标 (n_spots, 2)
        regions: 目标区域列表 [(cx, cy, size), ...]
        spot_names: spot名称列表

    返回:
        目标spot名称列表（按spot_names顺序排序）
    """
    target_spots = []
    for (cx, cy, size) in regions:
        x_min, x_max = cx - size / 2, cx + size / 2
        y_min, y_max = cy - size / 2, cy + size / 2
        in_region = (
                (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        )
        target_spots.extend([spot_names[i] for i in np.where(in_region)[0]])

    # 去重并按原始spot_names顺序排序（关键：固定顺序）
    target_spots = list(dict.fromkeys(target_spots))  # 保持首次出现顺序
    target_spots.sort(key=lambda x: spot_names.index(x))  # 按全局顺序排序
    return target_spots


def generate_spatial_data(
        reference: Union[mu.MuData, ad.AnnData],
        cell_type_key: str,
        num_spots: int = 576,
        balance: Optional[str] = None,
        cell_number_mean: Union[int, list] = [6, 8, 6, 8, 7, 9, 7, 9],
        cell_number_nu: Union[float, list] = 20.0,
        cell_type_number: Union[int, list] = [4, 4, 4, 4, 4, 4, 4, 4],
        poisson_noise_scale: float = 1.0,
        structured_shapes: List[str] = ["squares", "corners", "scotland", "checkers"],
        structured_base_size: int = 12,
        random_seed: int = 0
) -> tuple[Union[mu.MuData, ad.AnnData], pd.DataFrame]:
    """
    生成结构化空间转录组数据（对外统一接口）

    参数:
        reference: 参考数据（MuData/AnnData）
        cell_type_key: 细胞类型列名
        num_spots: 位点数
        balance: 采样平衡方式 ("balanced"/"unbalanced")
        cell_number_mean: 8个Region的平均细胞数
        cell_number_nu: CMP分布离散参数
        cell_type_number: 8个Region的细胞类型数
        poisson_noise_scale: 泊松噪声强度
        structured_shapes: 空间图案列表
        structured_base_size: 基础图案尺寸
        random_seed: 随机种子

    返回:
        模拟的空间数据（MuData/AnnData）、采样信息DataFrame
    """
    sampler = Sampler(
        reference=reference,
        cell_type_key=cell_type_key,
        num_spots=num_spots,
        cell_number_mean=cell_number_mean,
        cell_number_nu=cell_number_nu,
        cell_type_number=cell_type_number,
        balance=balance,
        poisson_noise_scale=poisson_noise_scale,
        structured_shapes=structured_shapes,
        structured_base_size=structured_base_size,
        random_seed=random_seed
    )

    exp, density, sampled_cells_df = sampler.sample_data()

    # 生成空间坐标
    X, Y = sampler.get_coords()
    coords = np.vstack((X.flatten(), Y.flatten())).T * 10

    # 封装为MuData/AnnData
    spatial_mod = {}
    spot_names = [f"spot_{i}" for i in range(len(exp[list(exp.keys())[0]]))]

    for key, data in exp.items():
        spatial_ann = ad.AnnData(scipy.sparse.csr_matrix(data))
        spatial_ann.var.index = reference.var_names if isinstance(reference, ad.AnnData) else reference[key].var_names
        spatial_ann.obs.index = spot_names  # 固定spot名称
        spatial_ann.obs["cell_count"] = density.sum(axis=1)
        spatial_ann.obs["region_id"] = sampler.regions
        spatial_ann.uns["density"] = pd.DataFrame(density, columns=sampler.clusters, index=spatial_ann.obs_names)
        spatial_ann.obsm["proportions"] = density / density.sum(axis=1)[:, None]
        spatial_ann.uns["proportion_names"] = sampler.clusters.values
        spatial_ann.obsm["spatial"] = coords.astype("int")
        spatial_mod[key] = spatial_ann

    if isinstance(reference, ad.AnnData):
        return spatial_mod["tmp"], sampled_cells_df
    elif isinstance(reference, mu.MuData):
        return mu.MuData(spatial_mod), sampled_cells_df