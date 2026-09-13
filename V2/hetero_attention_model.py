import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class ExplainableHeteroGNN(torch.nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for Polypharmacy Risk Reduction.

    Outputs:
        severity_logits : (N_patients, 1)      — BCE loss for severity prediction
        se_logits       : (N_patients, num_se)  — multi-label BCE for side effects
        x_dict          : final node embeddings for all node types
        attention_dict  : raw attention scores per relation, captured via hooks
                          shape per relation: (E, num_heads)
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        node_types: list,
        metadata: tuple,
        num_side_effects: int = 1000,
    ):
        super().__init__()

        # ── Input projection (Lazy: infers in_channels on first forward pass) ──
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # ── HGT layers ──────────────────────────────────────────────────────────
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # ── Task Head 1: Severity (binary) ───────────────────────────────────────
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1),  # raw logits for BCEWithLogitsLoss
        )

        # ── Task Head 2: Side-Effect Multi-label ────────────────────────────────
        self.side_effect_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_side_effects),  # logits for each SE class
        )

        # Attention storage (populated by forward hooks on each HGTConv layer)
        self.attention_dict: dict = {}
        self._hooks = []
        self._register_attention_hooks()

    # ── Attention Hooks ──────────────────────────────────────────────────────────
    def _register_attention_hooks(self):
        """
        PyG's HGTConv stores per-relation attention weights in `self._alpha`
        after each forward pass.  We register a forward hook on every conv
        layer so that we can capture these weights without modifying HGTConv.
        """
        def _make_hook(layer_idx):
            def hook(module, inputs, outputs):
                # HGTConv stores attention as _alpha: dict[rel_type, Tensor]
                if hasattr(module, "_alpha") and module._alpha is not None:
                    self.attention_dict[layer_idx] = {
                        rel: alpha.detach().cpu()
                        for rel, alpha in module._alpha.items()
                    }
            return hook

        for i, conv in enumerate(self.convs):
            h = conv.register_forward_hook(_make_hook(i))
            self._hooks.append(h)

    def remove_hooks(self):
        """Call this to free hook memory when not doing attribution."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Forward Pass ─────────────────────────────────────────────────────────────
    def forward(self, x_dict: dict, edge_index_dict: dict):
        # Reset attention store each pass
        self.attention_dict = {}

        # 1. Project all node types to common hidden dim
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # 2. Propagate through HGT layers
        # Residual fallback: HGTConv only outputs features for node types that
        # appear as edge *destinations*. Node types that are only sources (e.g.
        # 'patient' in patient->drug) would be silently dropped. We preserve
        # them by merging the pre-conv features back in when they are missing.
        for conv in self.convs:
            prev_x = x_dict                          # save before conv
            new_x  = conv(x_dict, edge_index_dict)  # may drop source-only types
            for ntype, feat in prev_x.items():
                if ntype not in new_x or new_x[ntype] is None:
                    new_x[ntype] = feat              # restore dropped node type
            x_dict = new_x

        # 3. Patient embeddings → dual prediction heads
        patient_embeds = x_dict["patient"]

        severity_logits = self.severity_head(patient_embeds)          # (N_p, 1)
        se_logits = self.side_effect_head(patient_embeds)              # (N_p, num_se)

        return severity_logits, se_logits, x_dict
