import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter


# --------------------------
# 基础注意力与GCN组件（核心保持原始结构）
# --------------------------
class FeatureAttentionFusion(nn.Module):
    """融合表达特征与空间特征的单向注意力模块"""

    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        self.scale = torch.sqrt(torch.tensor(feat_dim))
        self.fusion = nn.Linear(feat_dim, feat_dim)

    def forward(self, expr_feat, spatial_feat):
        q = self.query(expr_feat)
        k = self.key(spatial_feat).transpose(-2, -1)
        attn = torch.matmul(q, k) / self.scale
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, spatial_feat)
        fused = expr_feat + attended
        fused = self.fusion(fused)
        return F.relu(fused)


class AttentionLayer_within_modality(nn.Module):
    """单模态内：空间图与特征图嵌入的注意力融合（原始结构不变）"""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_within_modality, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        # 原始维度处理逻辑不变
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
        return torch.squeeze(emb_combined)


class GCNEncoder(nn.Module):
    """单模态GCN编码器：处理特征与图结构"""

    def __init__(self, expr_dim, hidden_dim, out_dim):
        super().__init__()
        self.expr_conv1 = GCNConv(expr_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, expr_x, edge_index):
        expr_feat = self.expr_conv1(expr_x, edge_index)
        expr_feat = F.relu(expr_feat)
        expr_feat = self.dropout(expr_feat)
        return self.output_layer(expr_feat)


# --------------------------
# 跨模态融合与编码器
# --------------------------
class CrossModalityFusion(nn.Module):
    """基因与蛋白质编码特征的双向注意力融合（移除投影层版本）"""

    def __init__(self, feat_dim, embedding_dim=256):
        super().__init__()
        # 移除 gene_proj 和 protein_proj
        self.scale = feat_dim ** 0.5
        self.fusion = nn.Sequential(
            nn.Linear(2 * feat_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, gene_feat, protein_feat):
        # 直接用原始编码特征计算注意力，无需投影
        attn_gene = (gene_feat @ protein_feat.transpose(-1, -2)) / self.scale
        attn_gene = F.softmax(attn_gene, dim=-1)
        gene_attended = attn_gene @ protein_feat

        attn_protein = (protein_feat @ gene_feat.transpose(-1, -2)) / self.scale
        attn_protein = F.softmax(attn_protein, dim=-1)
        protein_attended = attn_protein @ gene_feat

        fused = torch.cat([gene_feat + gene_attended, protein_feat + protein_attended], dim=-1)
        fused_feat = self.fusion(fused)

        return fused_feat, attn_gene, attn_protein


class GCN_between_attention_encoder(nn.Module):
    """仅模态间注意力的编码器（method=1）"""

    def __init__(self, in_feat_dim1, in_feat_dim2, hidden_dim, out_feat_dim):
        super().__init__()
        self.spatial_gcn_encoder1 = GCNEncoder(in_feat_dim1, hidden_dim, out_feat_dim)
        self.spatial_gcn_encoder2 = GCNEncoder(in_feat_dim2, hidden_dim, out_feat_dim)
        self.between_modality_atten = CrossModalityFusion(out_feat_dim, out_feat_dim)

    def forward(self, g_spatial_omics1, feat_omics1, g_spatial_omics2, feat_omics2):
        y1 = self.spatial_gcn_encoder1(feat_omics1, g_spatial_omics1)
        y2 = self.spatial_gcn_encoder2(feat_omics2, g_spatial_omics2)
        emb_latent, attn_gene2protein, attn_protein2gene = self.between_modality_atten(y1, y2)
        return emb_latent, y1, y2, attn_gene2protein, attn_protein2gene

    def encode_modal1(self, g_spatial, _, feat):
        return self.spatial_gcn_encoder1(feat, g_spatial)

    def encode_modal2(self, g_spatial, _, feat):
        return self.spatial_gcn_encoder2(feat, g_spatial)

class GCN_between_within_attention_encoder(nn.Module):
    """模态内+模态间注意力的编码器（method=2）"""

    def __init__(self, in_feat_dim1, in_feat_dim2, hidden_dim, out_feat_dim):
        super().__init__()
        self.atten_omics1 = AttentionLayer_within_modality(out_feat_dim, out_feat_dim)  # 原始注意力层
        self.atten_omics2 = AttentionLayer_within_modality(out_feat_dim, out_feat_dim)
        self.spatial_gcn_encoder1 = GCNEncoder(in_feat_dim1, hidden_dim, out_feat_dim)
        self.feature_gcn_encoder1 = GCNEncoder(in_feat_dim1, hidden_dim, out_feat_dim)
        self.spatial_gcn_encoder2 = GCNEncoder(in_feat_dim2, hidden_dim, out_feat_dim)
        self.feature_gcn_encoder2 = GCNEncoder(in_feat_dim2, hidden_dim, out_feat_dim)
        self.between_modality_atten = CrossModalityFusion(out_feat_dim, out_feat_dim)

    def forward(self, g_spatial_omics1, g_feature_omics1, feat_omics1,
                g_spatial_omics2, g_feature_omics2, feat_omics2):
        # 原始前向逻辑不变
        x_spatial_1 = self.spatial_gcn_encoder1(feat_omics1, g_spatial_omics1)
        x_feature_1 = self.feature_gcn_encoder1(feat_omics1, g_feature_omics1)
        y1 = self.atten_omics1(x_spatial_1, x_feature_1)  # 调用原始注意力层

        x_spatial_2 = self.spatial_gcn_encoder2(feat_omics2, g_spatial_omics2)
        x_feature_2 = self.feature_gcn_encoder2(feat_omics2, g_feature_omics2)
        y2 = self.atten_omics2(x_spatial_2, x_feature_2)

        emb_latent, attn_gene2protein, attn_protein2gene = self.between_modality_atten(y1, y2)
        return emb_latent, y1, y2, attn_gene2protein, attn_protein2gene

    # 暴露模态1/2的独立编码逻辑
    def encode_modal1(self, g_spatial, g_feature, feat):
        x_spatial = self.spatial_gcn_encoder1(feat, g_spatial)
        x_feature = self.feature_gcn_encoder1(feat, g_feature)
        return self.atten_omics1(x_spatial, x_feature)  # 复用原始注意力层

    def encode_modal2(self, g_spatial, g_feature, feat):
        x_spatial = self.spatial_gcn_encoder2(feat, g_spatial)
        x_feature = self.feature_gcn_encoder2(feat, g_feature)
        return self.atten_omics2(x_spatial, x_feature)  # 复用原始注意力层


# --------------------------
# 重构与预测组件
# --------------------------
class ExpressionReconstructor(nn.Module):
    def __init__(self, embedding_dim, pca_dim_rna, pca_dim_adt):
        super().__init__()
        # 模态1解码器：输入嵌入，输出模态1的PCA特征（用于重构损失）
        self.gene_pca_recon = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pca_dim_rna)
        )
        # 模态2解码器：输入嵌入，输出模态2的PCA特征（用于重构损失）
        self.modal2_pca_recon = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pca_dim_adt)
        )

    def forward(self, x):
        # 仅返回两个模态的PCA重构结果（无新增分支）
        gene_pca_out = self.gene_pca_recon(x)
        modal2_pca_out = self.modal2_pca_recon(x)
        return gene_pca_out, modal2_pca_out


# --------------------------
# 一致性检验器（复用现有解码器）
# --------------------------
class ConsistencyVerifier(nn.Module):
    def __init__(self, modal1_encoder, modal2_encoder, reconstructor):
        super().__init__()
        self.modal1_encoder = modal1_encoder  # 模态1编码器
        self.modal2_encoder = modal2_encoder  # 模态2编码器
        self.reconstructor = reconstructor    # 复用解码器

        # 关键：提前判断编码器需要的参数数量（区分method=1和method=2）
        import inspect
        # 获取模态1编码器的参数数量（排除self后的参数）
        self.modal1_param_count = len(inspect.signature(modal1_encoder).parameters)
        # 获取模态2编码器的参数数量
        self.modal2_param_count = len(inspect.signature(modal2_encoder).parameters)


    def forward(self, y1, y2,
                g_spatial1, g_feature1,
                g_spatial2, g_feature2):
        """
        自动适配两种method：
        - 若编码器需要2个参数（g_spatial, feat）→ method=1
        - 若编码器需要3个参数（g_spatial, g_feature, feat）→ method=2
        """
        # 步骤1-2：模态1→模态2→模态1（验证y1的一致性）
        modal2_pca_from_y1 = self.reconstructor.modal2_pca_recon(y1)
        # 根据模态2编码器的参数数量动态调用
        if self.modal2_param_count == 2:
            # method=1：编码器需要 (g_spatial, feat)
            y1_recon = self.modal2_encoder(g_spatial2, modal2_pca_from_y1)  # 不传g_feature2
        else:
            # method=2：编码器需要 (g_spatial, g_feature, feat)
            y1_recon = self.modal2_encoder(g_spatial2, g_feature2, modal2_pca_from_y1)  # 传全部


        # 步骤3-4：模态2→模态1→模态2（验证y2的一致性）
        modal1_pca_from_y2 = self.reconstructor.gene_pca_recon(y2)
        # 根据模态1编码器的参数数量动态调用
        if self.modal1_param_count == 2:
            # method=1：编码器需要 (g_spatial, feat)
            y2_recon = self.modal1_encoder(g_spatial1, modal1_pca_from_y2)  # 不传g_feature1
        else:
            # method=2：编码器需要 (g_spatial, g_feature, feat)
            y2_recon = self.modal1_encoder(g_spatial1, g_feature1, modal1_pca_from_y2)  # 传全部


        return y1_recon, y2_recon

class CellTypeProportionPredictor(nn.Module):
    """从嵌入预测细胞类型比例"""

    def __init__(self, embedding_dim, num_cell_types):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_cell_types),
            nn.Softmax(dim=1)
        )

    def forward(self, embedding):
        return self.predictor(embedding)


# --------------------------
# 负二项分布工具函数
# --------------------------
def compute_gene_nb_mu(cell_type_expr, proportions):
    """基于细胞类型比例计算基因表达均值μ"""
    return torch.matmul(proportions, cell_type_expr)


def gene_negative_binomial_loss(y_true, mu, theta):
    """数值稳定的负二项损失"""
    eps = 1e-8
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    lgamma_term = torch.lgamma(y_true + theta) - torch.lgamma(theta) - torch.lgamma(y_true + 1)
    log_term = theta * torch.log(theta) + y_true * torch.log(mu) - (theta + y_true) * torch.log(theta + mu)
    return -(lgamma_term + log_term).mean()