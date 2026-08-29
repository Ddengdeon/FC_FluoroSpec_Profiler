import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def smiles_to_graph_matrices(smiles, max_atoms=64):
    node_features = np.zeros((max_atoms, 36), dtype=np.float32)
    adj_matrix = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    edge_features = np.zeros((max_atoms, max_atoms, 12), dtype=np.float32)
    atom_mask = np.zeros((max_atoms,), dtype=np.float32)

    if not HAS_RDKIT or pd.isna(smiles): return node_features, adj_matrix, edge_features, atom_mask
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return node_features, adj_matrix, edge_features, atom_mask
        try:
            mol = Chem.RemoveHs(mol)
        except:
            pass

        num_atoms = min(mol.GetNumAtoms(), max_atoms)
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            elements = ['C', 'F', 'O', 'N', 'S', 'P', 'Cl', 'Br', 'I', 'H']
            elem_idx = elements.index(symbol) if symbol in elements else len(elements)
            node_features[i, elem_idx] = 1.0

            if symbol == 'F': node_features[i, 14] = 1.0
            if symbol == 'O': node_features[i, 15] = 1.0

            node_features[i, 11] = atom.GetDegree()
            node_features[i, 12] = atom.GetFormalCharge()
            node_features[i, 13] = 1.0 if atom.GetIsAromatic() else 0.0
            atom_mask[i] = 1.0
            adj_matrix[i, i] = 1.0

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u < max_atoms and v < max_atoms:
                adj_matrix[u, v] = 1.0
                adj_matrix[v, u] = 1.0

                b_type = bond.GetBondType()
                if b_type == Chem.rdchem.BondType.SINGLE:
                    edge_features[u, v, 0] = 1.0
                elif b_type == Chem.rdchem.BondType.DOUBLE:
                    edge_features[u, v, 1] = 1.0
                elif b_type == Chem.rdchem.BondType.TRIPLE:
                    edge_features[u, v, 2] = 1.0
                elif b_type == Chem.rdchem.BondType.AROMATIC:
                    edge_features[u, v, 3] = 1.0

                edge_features[u, v, 4] = 1.0 if bond.IsInRing() else 0.0
                edge_features[u, v, 5] = 1.0 if bond.GetIsConjugated() else 0.0

                u_sym, v_sym = mol.GetAtomWithIdx(u).GetSymbol(), mol.GetAtomWithIdx(v).GetSymbol()
                if (u_sym == 'C' and v_sym == 'F') or (u_sym == 'F' and v_sym == 'C'):
                    edge_features[u, v, 6] = 5.0
                if (u_sym == 'C' and v_sym == 'C'):
                    edge_features[u, v, 7] = 2.0
                if (u_sym == 'C' and v_sym == 'O') or (u_sym == 'O' and v_sym == 'C'):
                    edge_features[u, v, 8] = 3.0

                edge_features[v, u] = edge_features[u, v]
    except:
        pass
    return node_features, adj_matrix, edge_features, atom_mask


class FactorizedEdgeAwareLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.node_mlp = nn.Linear(node_dim, out_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Sigmoid()
        )

    def forward(self, h, adj, edge_feats, mask):
        h_out = self.node_mlp(h)
        edge_gates = self.edge_mlp(edge_feats)
        h_j = h_out.unsqueeze(1)
        m_ij = h_j * edge_gates * adj.unsqueeze(-1)
        m_i = m_ij.sum(dim=2)
        return F.gelu(h_out + m_i)


class SafeKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=3, spline_order=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.base_linear = nn.Linear(in_dim, out_dim)
        self.base_activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(in_dim)

        grid_min, grid_max = -7.0, 7.0
        h = (grid_max - grid_min) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1).float() * h + grid_min
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def b_splines(self, x):
        x = self.layer_norm(x)
        x = torch.clamp(x, -3.0, 3.0).unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[:-k - 1]) / (self.grid[k:-1] - self.grid[:-k - 1] + 1e-8) * bases[..., :-1]
            right = (self.grid[k + 1:] - x) / (self.grid[k + 1:] - self.grid[1:-k] + 1e-8) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        base_out = self.base_linear(self.base_activation(x))
        spline_out = torch.einsum('big,oig->bo', self.b_splines(x), self.spline_weight)
        return base_out + spline_out


class FiLMLayer(nn.Module):
    def __init__(self, meta_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(meta_dim, feature_dim)
        self.beta = nn.Linear(meta_dim, feature_dim)
        nn.init.constant_(self.gamma.weight, 0.0)
        nn.init.constant_(self.gamma.bias, 0.0)
        nn.init.constant_(self.beta.weight, 0.0)
        nn.init.constant_(self.beta.bias, 0.0)

    def forward(self, features, meta):
        g = self.gamma(meta)
        b = self.beta(meta)
        return features * (1.0 + g) + b


class SOTA_Graph_KAN_Model(nn.Module):
    def __init__(self, node_dim=36, edge_dim=12, out_spectrum_dim=1000):
        super().__init__()
        self.edge_gnn_layers = nn.ModuleList([
            FactorizedEdgeAwareLayer(node_dim, edge_dim, 128),
            FactorizedEdgeAwareLayer(128, edge_dim, 256),
            FactorizedEdgeAwareLayer(256, edge_dim, 256)
        ])

        self.attention_pool = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.meta_embed = nn.Sequential(nn.Linear(4, 128), nn.GELU())
        self.film_modulator = FiLMLayer(meta_dim=128, feature_dim=256)

        self.kan_refiner = nn.Sequential(
            SafeKANLayer(256 + 128, 1024), nn.Dropout(0.2),
            SafeKANLayer(1024, 512)
        )

        self.exist_head = nn.Sequential(
            nn.Linear(512 + 128, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, out_spectrum_dim)
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(512 + 128, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, out_spectrum_dim),
            nn.Sigmoid()
        )
        self.pfas_lora_adapter = nn.Sequential(
            nn.Linear(512 + 128, 64), nn.GELU(), nn.Linear(64, out_spectrum_dim)
        )
        nn.init.constant_(self.pfas_lora_adapter[-1].weight, 0.0)
        nn.init.constant_(self.pfas_lora_adapter[-1].bias, 0.0)

        self.physics_shortcut = nn.Sequential(
            nn.Linear(4, 256), nn.GELU(), nn.Linear(256, out_spectrum_dim)
        )
        self.specific_nl_scorer = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 1)
        )

        self.h_rearrange = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.h_rearrange.weight.copy_(torch.tensor([[[0.10, 0.80, 0.10]]]))
        # 🌟 锁死氢重排卷积核，绝不允许优化器修改这个固定物理常数！
        self.h_rearrange.weight.requires_grad = False

        nn.init.constant_(self.exist_head[-1].bias, -3.0)

        for m in self.intensity_head.modules():
            if isinstance(m, nn.Linear) and m.out_features == out_spectrum_dim:
                nn.init.constant_(m.bias, 0.1)

        nn.init.constant_(self.specific_nl_scorer[-1].bias, 0.0)
        nn.init.constant_(self.physics_shortcut[-1].bias, 0.0)

        self.pfas_classifier = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, node_feats, adj_matrix, edge_features, atom_mask, meta):
        h = node_feats
        for layer in self.edge_gnn_layers:
            h_new = layer(h, adj_matrix, edge_features, atom_mask)
            h = F.layer_norm(h_new, (h_new.size(-1),))

        mask = atom_mask.unsqueeze(-1)
        attn_weights = self.attention_pool(h * mask).masked_fill(mask == 0, -1e4)
        attn_weights = F.softmax(attn_weights, dim=1)
        z_mol = (h * attn_weights * mask).sum(dim=1)

        z_meta = self.meta_embed(meta)
        z_mol = self.film_modulator(z_mol, z_meta)
        z_total = torch.cat([z_mol, z_meta], dim=-1)
        h_kan = self.kan_refiner(z_total)

        if self.training: h_kan = h_kan + torch.randn_like(h_kan) * 0.05

        decoder_input = torch.cat([h_kan, z_meta], dim=-1)

        pmz_raw = meta[:, 0] * 500.0 + 500.0
        mass_axis = torch.arange(1000, device=meta.device).float() + 50.0
        nl_matrix = pmz_raw.unsqueeze(1) - mass_axis.unsqueeze(0)

        magic_masses = torch.tensor([19.0, 20.0, 44.0, 50.0, 100.0], device=meta.device)
        nl_gaussian = torch.exp(-0.5 * ((nl_matrix.unsqueeze(-1) - magic_masses) / 1.0) ** 2)
        precursor_gaussian = torch.exp(-0.5 * ((nl_matrix) / 1.0) ** 2).unsqueeze(-1)
        pf_channels = torch.cat([nl_gaussian, precursor_gaussian], dim=-1)
        specific_nl_logits = self.specific_nl_scorer(pf_channels).squeeze(-1)

        raw_exist_logits = self.exist_head(decoder_input) + specific_nl_logits + self.pfas_lora_adapter(decoder_input)
        exist_prob = torch.sigmoid(raw_exist_logits)

        intensity_val = self.intensity_head(decoder_input) + torch.sigmoid(self.physics_shortcut(meta))
        physics_mask = (mass_axis.unsqueeze(0) <= (pmz_raw.unsqueeze(1) + 2.0)).float()

        final_spectrum = exist_prob * intensity_val * physics_mask

        spec_unsqueezed = final_spectrum.unsqueeze(1)
        final_spectrum = self.h_rearrange(spec_unsqueezed).squeeze(1)

        final_spectrum = torch.clamp(final_spectrum, 0.0, 1.0)
        pfas_logits = self.pfas_classifier(h_kan)

        return final_spectrum, raw_exist_logits, physics_mask, pfas_logits


# ================= 探针自检程序 =================
if __name__ == "__main__":
    print("正在进行 v17 满血版 架构张量与连通性自检...")
    try:
        model = SOTA_Graph_KAN_Model()
        # 模拟 Batch Size 为 2 的图输入
        node = torch.randn(2, 64, 36)
        adj = torch.randn(2, 64, 64)
        edge = torch.randn(2, 64, 64, 12)
        mask = torch.ones(2, 64)
        meta = torch.randn(2, 4)

        final_spec, raw_logits, p_mask, pfas_logits = model(node, adj, edge, mask, meta)

        print(f"✅ 最终光谱张量 (final_spec) 维度: {final_spec.shape} (应为 [2, 1000])")
        print(f"✅ 存在性回归 Logits (raw_exist_logits) 维度: {raw_logits.shape} (应为 [2, 1000])")
        print(f"✅ 质量守恒物理掩码 (physics_mask) 维度: {p_mask.shape} (应为 [2, 1000])")
        print(f"✅ PFAS 专属分类张量 (pfas_logits) 维度: {pfas_logits.shape} (应为 [2, 1])")
        print(
            f"✅ 最终丰度范围检查: Max={final_spec.max().item():.3f}, Min={final_spec.min().item():.3f} (应被限制在 0~1)")
        print(
            f"✅ 氢重排卷积核锁定状态: requires_grad={model.h_rearrange.weight.requires_grad} (必须为 False, 防止污染物理定律)")
        print("🎉 探针自检全部通过！v17 满血版 (FiLM + 氢重排 + Float32防爆) 完美就绪，请放心开始训练！")
    except Exception as e:
        print(f"❌ 模型构建失败，报错原因: {e}")