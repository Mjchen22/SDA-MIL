import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial positional encoding
# ---------------------------------------------------------------------------
class SpatialPositionEncoding(nn.Module):
    def __init__(self, embed_dim=1024, alpha_init=0.0, max_scale=0.1):
        super().__init__()
        assert embed_dim % 2 == 0
        self.half_dim = embed_dim // 2
        quarter = self.half_dim // 2
        div_term = torch.exp(
            torch.arange(0, quarter, dtype=torch.float32) *
            -(math.log(10000.0) / quarter)
        )
        self.register_buffer('div_term', div_term)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.max_scale = max_scale

    def _encode_1d(self, pos):
        angle = pos.unsqueeze(1) * self.div_term.unsqueeze(0)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    def forward(self, h, coords):
        coords_f = coords.float()
        x, y = coords_f[:, 0], coords_f[:, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-6)
        pe = torch.cat([self._encode_1d(x_norm), self._encode_1d(y_norm)], dim=-1)
        alpha_eff = torch.sigmoid(self.alpha) * self.max_scale
        return h + alpha_eff * pe

    @property
    def alpha_effective(self):
        return (torch.sigmoid(self.alpha) * self.max_scale).item()


# ---------------------------------------------------------------------------
# Attention nets
# ---------------------------------------------------------------------------
class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.0, n_classes=1):
        super().__init__()
        drop_p = float(dropout) if isinstance(dropout, (float, int)) else 0.25
        module = [nn.Linear(L, D), nn.Tanh()]
        if drop_p > 0:
            module.append(nn.Dropout(drop_p))
        module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.0, n_classes=1):
        super().__init__()
        drop_p = float(dropout) if isinstance(dropout, (float, int)) else 0.25
        attention_a = [nn.Linear(L, D), nn.Tanh()]
        attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if drop_p > 0:
            attention_a.append(nn.Dropout(drop_p))
            attention_b.append(nn.Dropout(drop_p))
        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


# ---------------------------------------------------------------------------
# SCA: scale-aware feature decomposition
# ---------------------------------------------------------------------------
class ScaleDecomposer(nn.Module):
    def __init__(self, dim_in=1024, dim_out=512):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj_global = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_local = nn.Linear(dim_in, dim_out, bias=False)
        self.norm_global = nn.LayerNorm(dim_out)
        self.norm_local = nn.LayerNorm(dim_out)

    def forward(self, h):
        g = self.norm_global(self.proj_global(h))
        l = self.norm_local(self.proj_local(h))
        return g, l

    def orthogonality_loss(self):
        wg = F.normalize(self.proj_global.weight, p=2, dim=0)
        wl = F.normalize(self.proj_local.weight, p=2, dim=0)
        cross = wg @ wl.t()
        return (cross ** 2).mean()


# ---------------------------------------------------------------------------
# DD-TAGC layer
# ---------------------------------------------------------------------------
class DDTAGCLayer(nn.Module):
    def __init__(self, dim=1024, heads=4, k=8, candidate_k=16, k_geo=4,
                 dropout=0.1, radius=0.0, use_ffn=True,
                 knn_chunk_size=1024, rerank_chunk=4096,
                 use_dma=False, dma_beta=0.5):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.k = max(1, int(k))
        self.candidate_k = max(self.k, int(candidate_k))
        self.k_geo = min(max(0, int(k_geo)), self.k)
        self.radius = radius
        self.use_ffn = use_ffn
        self.knn_chunk_size = knn_chunk_size
        self.rerank_chunk = max(256, int(rerank_chunk))
        self.use_dma = use_dma
        self.dma_beta = float(dma_beta)

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, heads),
        )
        self.feat_scale = nn.Parameter(torch.tensor(0.5))
        self.rerank_feat_scale = nn.Parameter(torch.tensor(1.0))
        self.rerank_dist_scale = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.ones(heads))

        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -4.0)

        self.attn_dropout = nn.Dropout(dropout)
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim * 2, dim), nn.Dropout(dropout),
            )
            nn.init.zeros_(self.ffn[3].weight)
            nn.init.zeros_(self.ffn[3].bias)

    @torch.no_grad()
    def build_candidate_knn(self, coords):
        N = coords.size(0)
        if N <= 1:
            return None, None, None
        k0 = min(self.candidate_k, N - 1)
        device = coords.device
        coords = coords.float()
        all_idx, all_dist = [], []
        for start in range(0, N, self.knn_chunk_size):
            end = min(start + self.knn_chunk_size, N)
            q = coords[start:end]
            dist = torch.cdist(q, coords, p=2)
            row_ids = torch.arange(end - start, device=device)
            col_ids = torch.arange(start, end, device=device)
            dist[row_ids, col_ids] = float('inf')
            vals, idx = torch.topk(dist, k=k0, dim=1, largest=False)
            all_idx.append(idx)
            all_dist.append(vals)
        cand_idx = torch.cat(all_idx, dim=0)
        cand_dist = torch.cat(all_dist, dim=0)
        if self.radius is not None and self.radius > 0:
            valid_mask = cand_dist <= self.radius
            empty_rows = valid_mask.sum(dim=1) == 0
            if empty_rows.any():
                valid_mask[empty_rows, 0] = True
        else:
            valid_mask = torch.ones_like(cand_idx, dtype=torch.bool, device=device)
        return cand_idx, cand_dist, valid_mask

    def _chunked_feat_sim(self, h_norm, cand_idx):
        N = h_norm.size(0)
        ch = self.rerank_chunk
        if ch >= N:
            nbr = h_norm[cand_idx]
            return (h_norm.unsqueeze(1) * nbr).sum(dim=-1)
        parts = []
        for start in range(0, N, ch):
            end = min(start + ch, N)
            idx = cand_idx[start:end]
            nbr = h_norm[idx]
            parts.append((h_norm[start:end].unsqueeze(1) * nbr).sum(dim=-1))
        return torch.cat(parts, dim=0)

    def select_mixed_neighbors(self, cand_idx, cand_dist, valid_mask, h_norm):
        N, K0 = cand_idx.shape
        feat_sim = self._chunked_feat_sim(h_norm, cand_idx)
        sigma = cand_dist[valid_mask].median().detach().clamp_min(1e-6)
        rerank_score = (
            self.rerank_feat_scale * feat_sim
            - self.rerank_dist_scale * (cand_dist / sigma).pow(2)
        )
        rerank_score = rerank_score.masked_fill(~valid_mask, -1e6)
        k_geo = min(self.k_geo, K0)
        k_joint = max(0, min(self.k - k_geo, K0 - k_geo))
        pos_list = []
        if k_geo > 0:
            geo_pos = torch.arange(k_geo, device=cand_idx.device).view(1, -1).expand(N, -1)
            pos_list.append(geo_pos)
        if k_joint > 0:
            joint_score = rerank_score.clone()
            if k_geo > 0:
                joint_score[:, :k_geo] = -1e6
            joint_pos = torch.topk(joint_score, k=k_joint, dim=1, largest=True).indices
            pos_list.append(joint_pos)
        select_pos = pos_list[0] if len(pos_list) == 1 else torch.cat(pos_list, dim=1)
        final_idx = torch.gather(cand_idx, 1, select_pos)
        final_dist = torch.gather(cand_dist, 1, select_pos)
        final_valid = torch.gather(valid_mask, 1, select_pos)
        final_feat_sim = torch.gather(feat_sim, 1, select_pos)
        return final_idx, final_dist, final_valid, final_feat_sim

    def forward(self, h, coords):
        N = h.size(0)
        if N <= 1:
            return h
        h_res = h
        h = self.norm1(h)
        h_norm = F.normalize(h, p=2, dim=-1)
        cand_idx, cand_dist, valid_mask = self.build_candidate_knn(coords)
        if cand_idx is None:
            return h_res
        knn_idx, knn_dist, knn_valid, feat_sim = self.select_mixed_neighbors(
            cand_idx=cand_idx, cand_dist=cand_dist,
            valid_mask=valid_mask, h_norm=h_norm,
        )
        q = self.q_proj(h).view(N, self.heads, self.head_dim)
        k = self.k_proj(h).view(N, self.heads, self.head_dim)
        v = self.v_proj(h).view(N, self.heads, self.head_dim)
        nbr_k = k[knn_idx]
        nbr_v = v[knn_idx]
        q_i = q.unsqueeze(1)
        attn_feat = (q_i * nbr_k).sum(dim=-1) / math.sqrt(self.head_dim)
        nbr_coords = coords[knn_idx].float()
        ctr_coords = coords.unsqueeze(1).float()
        rel = nbr_coords - ctr_coords
        dist = torch.norm(rel, dim=-1, keepdim=True).clamp_min(1e-8)
        sigma = knn_dist[knn_valid].median().detach().clamp_min(1e-6)
        geom_input = torch.cat([rel / sigma, dist / sigma], dim=-1)
        geom_bias = self.edge_mlp(geom_input.reshape(-1, 3)).view(N, -1, self.heads)
        feat_sim_h = feat_sim.unsqueeze(-1)
        dist_penalty = self.gamma.view(1, 1, self.heads) * (dist / sigma).pow(2)
        logits = attn_feat + geom_bias + self.feat_scale * feat_sim_h - dist_penalty.squeeze(-1)
        logits = logits.masked_fill(~knn_valid.unsqueeze(-1), -1e4)

        if self.use_dma:
            d_norm = (dist / sigma).squeeze(-1)
            t_factor = 1.0 + self.dma_beta * d_norm
            logits = logits / t_factor.unsqueeze(-1).clamp_min(1e-3)

        alpha = F.softmax(logits, dim=1)
        alpha = alpha * knn_valid.unsqueeze(-1).float()
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-6)
        alpha = self.attn_dropout(alpha)
        msg = (alpha.unsqueeze(-1) * nbr_v).sum(dim=1)
        msg = msg.reshape(N, self.dim)
        msg = self.out_proj(msg)
        g = self.gate(torch.cat([h_res, msg], dim=-1))
        out = h_res + g * msg
        if self.use_ffn:
            out = out + self.ffn(self.norm2(out))
        return out


class DDTAGCStack(nn.Module):
    def __init__(self, dim=1024, layers=1, heads=4, k=8, candidate_k=16, k_geo=4,
                 dropout=0.1, radius=0.0, use_ffn=True, knn_chunk_size=1024,
                 rerank_chunk=4096, use_dma=False, dma_beta=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            DDTAGCLayer(
                dim=dim, heads=heads, k=k, candidate_k=candidate_k, k_geo=k_geo,
                dropout=dropout, radius=radius, use_ffn=use_ffn,
                knn_chunk_size=knn_chunk_size, rerank_chunk=rerank_chunk,
                use_dma=use_dma, dma_beta=dma_beta,
            )
            for _ in range(layers)
        ])

    def forward(self, h, coords):
        for layer in self.layers:
            h = layer(h, coords)
        return h


class SCAGraphRefiner(nn.Module):
    def __init__(self, embed_dim=1024, split_dim=512,
                 layers=1, heads=4, local_k=4, global_k=8,
                 candidate_k=16, k_geo=4, dropout=0.1, radius=0.0,
                 use_ffn=True, knn_chunk_size=1024, rerank_chunk=4096,
                 use_dma=False, dma_beta=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.split_dim = split_dim
        self.decomposer = ScaleDecomposer(dim_in=embed_dim, dim_out=split_dim)

        sub_heads = heads
        while split_dim % sub_heads != 0 and sub_heads > 1:
            sub_heads -= 1

        self.global_stack = DDTAGCStack(
            dim=split_dim, layers=layers, heads=sub_heads,
            k=global_k, candidate_k=max(candidate_k, global_k * 2),
            k_geo=min(k_geo, global_k),
            dropout=dropout, radius=radius, use_ffn=use_ffn,
            knn_chunk_size=knn_chunk_size, rerank_chunk=rerank_chunk,
            use_dma=use_dma, dma_beta=dma_beta)
        self.local_stack = DDTAGCStack(
            dim=split_dim, layers=layers, heads=sub_heads,
            k=local_k, candidate_k=max(candidate_k // 2, local_k * 2),
            k_geo=min(k_geo, local_k),
            dropout=dropout, radius=radius, use_ffn=use_ffn,
            knn_chunk_size=knn_chunk_size, rerank_chunk=rerank_chunk,
            use_dma=use_dma, dma_beta=dma_beta)

        self.up_global = nn.Linear(split_dim, embed_dim, bias=False)
        self.up_local = nn.Linear(split_dim, embed_dim, bias=False)
        nn.init.zeros_(self.up_global.weight)
        nn.init.zeros_(self.up_local.weight)

        self.fuse_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.Sigmoid())
        nn.init.zeros_(self.fuse_gate[0].weight)
        nn.init.constant_(self.fuse_gate[0].bias, -4.0)

        self._last_ortho = torch.tensor(0.0)

    def forward(self, h, coords):
        h_res = h
        g_feat, l_feat = self.decomposer(h)
        g_out = self.global_stack(g_feat, coords)
        l_out = self.local_stack(l_feat, coords)
        delta_g = self.up_global(g_out)
        delta_l = self.up_local(l_out)
        gate_input = torch.cat([h_res, delta_g, delta_l], dim=-1)
        gate = self.fuse_gate(gate_input)
        out = h_res + gate * (delta_g + delta_l)
        self._last_ortho = self.decomposer.orthogonality_loss()
        return out

    @property
    def last_ortho_loss(self):
        return self._last_ortho


# ---------------------------------------------------------------------------
# CLAM_SB
# ---------------------------------------------------------------------------
class CLAM_SB(nn.Module):
    def __init__(
        self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
        use_graph=True, graph_layers=1, graph_heads=4, graph_k=8,
        graph_candidate_k=16, graph_k_geo=4, graph_dropout=0.1, graph_radius=0.0,
        graph_use_ffn=True, graph_chunk_size=1024, graph_rerank_chunk=4096,
        # SCA
        use_sca=False, sca_split_dim=512, sca_local_k=4, sca_global_k=8,
        # DMA
        use_dma=False, dma_beta=0.5,
        use_pos_enc=True, pos_enc_scale=0.1, pos_enc_alpha_init=0.0,
        ctx_inst_weight=0.1, inst_consistency_weight=0.01, use_ctx_branch=True,
    ):
        super().__init__()
        self.size_dict = {
            "micro": [embed_dim, 384, 192],
            "small": [embed_dim, 512, 256],
            "big":   [embed_dim, 512, 384],
        }
        size = self.size_dict[size_arg]

        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.use_graph = use_graph
        self.use_sca = use_sca and use_graph
        self.use_pos_enc = use_pos_enc
        self.use_ctx_branch = use_ctx_branch
        self.ctx_inst_weight = ctx_inst_weight
        self.inst_consistency_weight = inst_consistency_weight

        if self.use_sca:
            self.graph_refiner = SCAGraphRefiner(
                embed_dim=embed_dim, split_dim=sca_split_dim,
                layers=graph_layers, heads=graph_heads,
                local_k=sca_local_k, global_k=sca_global_k,
                candidate_k=graph_candidate_k, k_geo=graph_k_geo,
                dropout=graph_dropout, radius=graph_radius,
                use_ffn=graph_use_ffn, knn_chunk_size=graph_chunk_size,
                rerank_chunk=graph_rerank_chunk,
                use_dma=use_dma, dma_beta=dma_beta)
        elif self.use_graph:
            self.graph_refiner = DDTAGCStack(
                dim=embed_dim, layers=graph_layers, heads=graph_heads,
                k=graph_k, candidate_k=graph_candidate_k, k_geo=graph_k_geo,
                dropout=graph_dropout, radius=graph_radius,
                use_ffn=graph_use_ffn, knn_chunk_size=graph_chunk_size,
                rerank_chunk=graph_rerank_chunk,
                use_dma=use_dma, dma_beta=dma_beta)
        else:
            self.graph_refiner = None

        self.pos_enc = SpatialPositionEncoding(
            embed_dim=embed_dim, alpha_init=pos_enc_alpha_init, max_scale=pos_enc_scale)

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = (Attn_Net_Gated if gate else Attn_Net)(
            L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)])
        if self.use_ctx_branch:
            self.ctx_instance_classifiers = nn.ModuleList(
                [nn.Linear(embed_dim, 2) for _ in range(n_classes)])
        else:
            self.ctx_instance_classifiers = None

    # ---- helpers ----
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def _safe_topk(self, A):
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_eff = min(self.k_sample, A.size(1))
        top_p_ids = torch.topk(A, k_eff, dim=1)[1][-1]
        top_n_ids = torch.topk(-A, k_eff, dim=1)[1][-1]
        return top_p_ids, top_n_ids

    def _safe_topk_pos(self, A):
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_eff = min(self.k_sample, A.size(1))
        return torch.topk(A, k_eff, dim=1)[1][-1]

    def _gather_pos_neg_instances(self, h, pos_ids, neg_ids):
        device = h.device
        top_p = torch.index_select(h, dim=0, index=pos_ids)
        top_n = torch.index_select(h, dim=0, index=neg_ids)
        p_targets = self.create_positive_targets(len(pos_ids), device)
        n_targets = self.create_negative_targets(len(neg_ids), device)
        return torch.cat([top_p, top_n], dim=0), torch.cat([p_targets, n_targets], dim=0)

    def _gather_pos_only_instances(self, h, pos_ids):
        device = h.device
        top_p = torch.index_select(h, dim=0, index=pos_ids)
        p_targets = self.create_negative_targets(len(pos_ids), device)
        return top_p, p_targets

    def _run_classifier(self, instances, targets, classifier):
        logits = classifier(instances)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, targets)
        return loss, preds, targets, logits

    def _consistency_loss(self, logits_a, logits_b):
        return F.mse_loss(F.softmax(logits_a, dim=1), F.softmax(logits_b, dim=1))

    def get_aux_losses(self):
        out = {}
        if self.use_sca and self.graph_refiner is not None:
            out['sca_ortho'] = self.graph_refiner.last_ortho_loss
        return out

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False, coords=None):
        coords_dev = coords.to(h.device) if coords is not None else None

        if self.use_graph and self.graph_refiner is not None \
                and coords_dev is not None and h.size(0) > 1:
            h = self.graph_refiner(h, coords_dev)

        h_ctx = h
        h_attn = self.pos_enc(h_ctx, coords_dev) if (self.use_pos_enc and coords_dev is not None) else h_ctx

        A, h_proj = self.attention_net(h_attn)
        A = torch.transpose(A, 1, 0)

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_proj = torch.tensor(0.0, device=h.device)
            total_ctx = torch.tensor(0.0, device=h.device)
            total_cons = torch.tensor(0.0, device=h.device)
            all_preds, all_targets = [], []
            valid = 0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                proj_cls = self.instance_classifiers[i]
                ctx_cls = self.ctx_instance_classifiers[i] if self.use_ctx_branch else None

                if inst_label == 1:
                    pos_ids, neg_ids = self._safe_topk(A)
                    proj_inst, targets = self._gather_pos_neg_instances(h_proj, pos_ids, neg_ids)
                    pl, preds, targets, p_log = self._run_classifier(proj_inst, targets, proj_cls)
                    if self.use_ctx_branch:
                        ctx_inst, _ = self._gather_pos_neg_instances(h_ctx, pos_ids, neg_ids)
                        cl, _, _, c_log = self._run_classifier(ctx_inst, targets, ctx_cls)
                        col = self._consistency_loss(p_log, c_log)
                    else:
                        cl = torch.tensor(0.0, device=h.device)
                        col = torch.tensor(0.0, device=h.device)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        pos_ids = self._safe_topk_pos(A)
                        proj_inst, targets = self._gather_pos_only_instances(h_proj, pos_ids)
                        pl, preds, targets, p_log = self._run_classifier(proj_inst, targets, proj_cls)
                        if self.use_ctx_branch:
                            ctx_inst, _ = self._gather_pos_only_instances(h_ctx, pos_ids)
                            cl, _, _, c_log = self._run_classifier(ctx_inst, targets, ctx_cls)
                            col = self._consistency_loss(p_log, c_log)
                        else:
                            cl = torch.tensor(0.0, device=h.device)
                            col = torch.tensor(0.0, device=h.device)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_proj = total_proj + pl
                total_ctx = total_ctx + cl
                total_cons = total_cons + col
                valid += 1
            if valid > 0:
                total_proj = total_proj / valid
                total_ctx = total_ctx / valid
                total_cons = total_cons / valid
            instance_loss = (total_proj
                             + self.ctx_inst_weight * total_ctx
                             + self.inst_consistency_weight * total_cons)

        M = torch.mm(A, h_proj)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {
                'instance_loss': instance_loss,
                'proj_instance_loss': total_proj,
                'ctx_instance_loss': total_ctx,
                'inst_consistency_loss': total_cons,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
            }
        else:
            results_dict = {}

        if return_features:
            results_dict.update({'features': M, 'context_features': h_ctx})

        return logits, Y_prob, Y_hat, A_raw, results_dict


# ---------------------------------------------------------------------------
# CLAM_MB
# ---------------------------------------------------------------------------
class CLAM_MB(CLAM_SB):
    def __init__(
        self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,
        use_graph=True, graph_layers=1, graph_heads=4, graph_k=8,
        graph_candidate_k=16, graph_k_geo=4, graph_dropout=0.1, graph_radius=0.0,
        graph_use_ffn=True, graph_chunk_size=1024, graph_rerank_chunk=4096,
        use_sca=False, sca_split_dim=512, sca_local_k=4, sca_global_k=8,
        use_dma=False, dma_beta=0.5,
        use_pos_enc=True, pos_enc_scale=0.1, pos_enc_alpha_init=0.0,
        ctx_inst_weight=0.1, inst_consistency_weight=0.01, use_ctx_branch=True,
    ):
        nn.Module.__init__(self)
        self.size_dict = {
            "micro": [embed_dim, 384, 192],
            "small": [embed_dim, 512, 256],
            "big":   [embed_dim, 512, 384],
        }
        size = self.size_dict[size_arg]
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.use_graph = use_graph
        self.use_sca = use_sca and use_graph
        self.use_pos_enc = use_pos_enc
        self.use_ctx_branch = use_ctx_branch
        self.ctx_inst_weight = ctx_inst_weight
        self.inst_consistency_weight = inst_consistency_weight

        if self.use_sca:
            self.graph_refiner = SCAGraphRefiner(
                embed_dim=embed_dim, split_dim=sca_split_dim,
                layers=graph_layers, heads=graph_heads,
                local_k=sca_local_k, global_k=sca_global_k,
                candidate_k=graph_candidate_k, k_geo=graph_k_geo,
                dropout=graph_dropout, radius=graph_radius,
                use_ffn=graph_use_ffn, knn_chunk_size=graph_chunk_size,
                rerank_chunk=graph_rerank_chunk,
                use_dma=use_dma, dma_beta=dma_beta)
        elif self.use_graph:
            self.graph_refiner = DDTAGCStack(
                dim=embed_dim, layers=graph_layers, heads=graph_heads,
                k=graph_k, candidate_k=graph_candidate_k, k_geo=graph_k_geo,
                dropout=graph_dropout, radius=graph_radius,
                use_ffn=graph_use_ffn, knn_chunk_size=graph_chunk_size,
                rerank_chunk=graph_rerank_chunk,
                use_dma=use_dma, dma_beta=dma_beta)
        else:
            self.graph_refiner = None

        self.pos_enc = SpatialPositionEncoding(
            embed_dim=embed_dim, alpha_init=pos_enc_alpha_init, max_scale=pos_enc_scale)

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = (Attn_Net_Gated if gate else Attn_Net)(
            L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.ModuleList(
            [nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)])
        if self.use_ctx_branch:
            self.ctx_instance_classifiers = nn.ModuleList(
                [nn.Linear(embed_dim, 2) for _ in range(n_classes)])
        else:
            self.ctx_instance_classifiers = None

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False, coords=None):
        coords_dev = coords.to(h.device) if coords is not None else None

        if self.use_graph and self.graph_refiner is not None \
                and coords_dev is not None and h.size(0) > 1:
            h = self.graph_refiner(h, coords_dev)

        h_ctx = h
        h_attn = self.pos_enc(h_ctx, coords_dev) if (self.use_pos_enc and coords_dev is not None) else h_ctx
        A, h_proj = self.attention_net(h_attn)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_proj = torch.tensor(0.0, device=h.device)
            total_ctx = torch.tensor(0.0, device=h.device)
            total_cons = torch.tensor(0.0, device=h.device)
            all_preds, all_targets = [], []
            valid = 0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                proj_cls = self.instance_classifiers[i]
                ctx_cls = self.ctx_instance_classifiers[i] if self.use_ctx_branch else None
                if inst_label == 1:
                    pos_ids, neg_ids = self._safe_topk(A[i])
                    proj_inst, targets = self._gather_pos_neg_instances(h_proj, pos_ids, neg_ids)
                    pl, preds, targets, p_log = self._run_classifier(proj_inst, targets, proj_cls)
                    if self.use_ctx_branch:
                        ctx_inst, _ = self._gather_pos_neg_instances(h_ctx, pos_ids, neg_ids)
                        cl, _, _, c_log = self._run_classifier(ctx_inst, targets, ctx_cls)
                        col = self._consistency_loss(p_log, c_log)
                    else:
                        cl = torch.tensor(0.0, device=h.device)
                        col = torch.tensor(0.0, device=h.device)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        pos_ids = self._safe_topk_pos(A[i])
                        proj_inst, targets = self._gather_pos_only_instances(h_proj, pos_ids)
                        pl, preds, targets, p_log = self._run_classifier(proj_inst, targets, proj_cls)
                        if self.use_ctx_branch:
                            ctx_inst, _ = self._gather_pos_only_instances(h_ctx, pos_ids)
                            cl, _, _, c_log = self._run_classifier(ctx_inst, targets, ctx_cls)
                            col = self._consistency_loss(p_log, c_log)
                        else:
                            cl = torch.tensor(0.0, device=h.device)
                            col = torch.tensor(0.0, device=h.device)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_proj = total_proj + pl
                total_ctx = total_ctx + cl
                total_cons = total_cons + col
                valid += 1
            if valid > 0:
                total_proj = total_proj / valid
                total_ctx = total_ctx / valid
                total_cons = total_cons / valid
            instance_loss = (total_proj
                             + self.ctx_inst_weight * total_ctx
                             + self.inst_consistency_weight * total_cons)

        M = torch.mm(A, h_proj)
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {
                'instance_loss': instance_loss,
                'proj_instance_loss': total_proj,
                'ctx_instance_loss': total_ctx,
                'inst_consistency_loss': total_cons,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
            }
        else:
            results_dict = {}

        if return_features:
            results_dict.update({'features': M, 'context_features': h_ctx})

        return logits, Y_prob, Y_hat, A_raw, results_dict
