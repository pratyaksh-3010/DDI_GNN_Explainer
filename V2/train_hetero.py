"""
train_hetero.py  -- V2 HGT Training (Windows CUDA)
====================================================
Trains ExplainableHeteroGNN with two heads:
  1. Severity prediction  (binary BCE, NaN-masked)
  2. Side-effect pred     (multi-label BCE, top-1000 SEs)

Fixes applied over v1:
  - patient.x was all-ones (zero signal) -> replaced with degree features
  - patient.y has NaN entries            -> masked out of severity loss
  - SE labels 92% dense                 -> pos_weight + lower SE loss weight
  - LR 0.005 too high                   -> 0.001 + gradient clipping
"""

import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree

from hetero_attention_model import ExplainableHeteroGNN

# ── Config ─────────────────────────────────────────────────────────────────────
PROCESSED = r"V2\Processed"

CFG = {
    "hidden_channels":      128,
    "num_heads":             4,
    "num_layers":            2,
    "num_side_effects":   1000,
    "lr":                 1e-3,
    "weight_decay":       1e-4,
    "epochs":               50,
    "batch_size":          256,
    "num_neighbors":    [15, 5],
    "severity_loss_weight": 1.0,
    "se_loss_weight":       0.1,
    "grad_clip":            1.0,
}


def add_reverse_edges(data):
    """
    Add reverse edge types so every node type can receive messages.
    Without these, patient nodes are only message *sources*, never destinations,
    so HGT never updates patient embeddings -> model predicts majority class.

    Adds:
      (drug, prescribed_by, patient)   -- reverse of (patient, prescribed, drug)
      (protein, targeted_by, drug)     -- reverse of (drug, targets, protein)
      (side_effect, caused_by, drug)   -- reverse of (drug, causes, side_effect)
    """
    # (drug, prescribed_by, patient)
    fwd = data["patient", "prescribed", "drug"].edge_index      # [2, E]
    data["drug", "prescribed_by", "patient"].edge_index = fwd.flip(0)

    # (protein, targeted_by, drug)
    fwd = data["drug", "targets", "protein"].edge_index
    data["protein", "targeted_by", "drug"].edge_index = fwd.flip(0)

    # (side_effect, caused_by, drug)
    fwd = data["drug", "causes", "side_effect"].edge_index
    data["side_effect", "caused_by", "drug"].edge_index = fwd.flip(0)

    print(f"  Reverse edges added. New edge types: {len(data.edge_types)}")
    return data



def build_patient_degree_features(data):
    """
    Replace all-ones patient.x with degree-based features:
      - out-degree in patient->drug (number of drugs prescribed)
    Normalised to [0, 1] so magnitudes are comparable.
    """
    num_patients = data["patient"].num_nodes
    pd_edge = data["patient", "prescribed", "drug"].edge_index  # (2, E)
    deg = degree(pd_edge[0], num_nodes=num_patients).float()    # (num_patients,)
    # Normalise by max
    deg = deg / (deg.max() + 1e-8)
    return deg.unsqueeze(1)                                      # (num_patients, 1)


def build_se_labels(data, num_patients, num_se):
    """Multi-hot SE matrix via sparse patient->drug->SE join."""
    _DIRECT = ("patient", "experiences", "side_effect")
    if _DIRECT in data.edge_types:
        se_labels = torch.zeros(num_patients, num_se)
        pe = data[_DIRECT].edge_index
        valid = pe[1] < num_se
        se_labels[pe[0][valid], pe[1][valid]] = 1.0
        print(f"  [Path A] Direct patient->SE: {valid.sum().item()} entries")
        return se_labels

    print("  [Path B] Vectorised patient->drug->SE matmul ...")
    pd_edge = data["patient", "prescribed", "drug"].edge_index
    ds_edge = data["drug", "causes", "side_effect"].edge_index
    num_drugs = data["drug"].num_nodes

    A = torch.sparse_coo_tensor(
        pd_edge, torch.ones(pd_edge.shape[1]),
        size=(num_patients, num_drugs),
    ).coalesce()

    valid_ds = ds_edge[1] < num_se
    ds_f = ds_edge[:, valid_ds]
    B = torch.sparse_coo_tensor(
        ds_f, torch.ones(ds_f.shape[1]),
        size=(num_drugs, num_se),
    ).coalesce()

    C = torch.sparse.mm(A, B.to_dense())
    labels = (C > 0).float()
    print(f"  [Path B] Done: {labels.sum().long().item():,} entries "
          f"(density {labels.mean().item():.1%})")
    return labels


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)

    print("=" * 60)
    print("  V2 HGT -- CUDA Training  (NaN-safe, degree features)")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Set device='cpu' to override.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    print(f"\nGPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    data = torch.load(os.path.join(PROCESSED, "hetero_graph_data.pt"),
                      weights_only=False)
    print(data)

    # ── Add reverse edges so patient/protein/SE receive messages ─────────────
    print("\n      Adding reverse edges for bidirectional message passing...")
    data = add_reverse_edges(data)

    # ── Fix patient.x: replace all-1s with degree features ───────────────────
    print("\n      Replacing degenerate patient.x with degree features...")
    data["patient"].x = build_patient_degree_features(data)
    print(f"      patient.x range: [{data['patient'].x.min():.4f}, "
          f"{data['patient'].x.max():.4f}]")

    # ── Fix patient.y: mask NaN labels ───────────────────────────────────────
    y_raw = data["patient"].y
    nan_mask = y_raw.isnan()
    print(f"      patient.y: {(~nan_mask).sum().item():,} valid  |  "
          f"{nan_mask.sum().item():,} NaN (will be masked from severity loss)")
    # Replace NaN with 0 (won't matter — masked in loss)
    data["patient"].y = y_raw.nan_to_num(0.0)
    data["patient"].valid_mask = ~nan_mask  # True where label is reliable

    # ── SE labels ────────────────────────────────────────────────────────────
    print("\n[2/5] Building multi-hot side-effect labels...")
    num_patients = data["patient"].num_nodes
    num_se = CFG["num_side_effects"]
    data["patient"].se_labels = build_se_labels(data, num_patients, num_se)

    # ── Splits ───────────────────────────────────────────────────────────────
    print("\n[3/5] Creating train/val splits...")
    perm = torch.randperm(num_patients)
    train_idx = perm[:int(0.8 * num_patients)]
    val_idx   = perm[int(0.8 * num_patients):]
    data["patient"].train_mask = torch.zeros(num_patients, dtype=torch.bool)
    data["patient"].train_mask[train_idx] = True
    data["patient"].val_mask = torch.zeros(num_patients, dtype=torch.bool)
    data["patient"].val_mask[val_idx] = True
    print(f"  Train: {train_idx.shape[0]:,}  |  Val: {val_idx.shape[0]:,}")

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_loader = NeighborLoader(
        data,
        num_neighbors=CFG["num_neighbors"],
        batch_size=CFG["batch_size"],
        input_nodes=("patient", data["patient"].train_mask),
        shuffle=True,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=CFG["num_neighbors"],
        batch_size=CFG["batch_size"],
        input_nodes=("patient", data["patient"].val_mask),
        shuffle=False,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Initializing model...")
    model = ExplainableHeteroGNN(
        hidden_channels=CFG["hidden_channels"],
        num_heads=CFG["num_heads"],
        num_layers=CFG["num_layers"],
        node_types=data.node_types,
        metadata=data.metadata(),
        num_side_effects=CFG["num_side_effects"],
    ).to(device)

    print("  Warming up Lazy layers...")
    with torch.no_grad():
        dummy = next(iter(train_loader)).to(device)
        model(dummy.x_dict, dummy.edge_index_dict)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-5)

    # Severity: standard BCE (NaN-masked inside loss)
    sev_crit = torch.nn.BCEWithLogitsLoss(reduction="none")

    # SE: pos_weight to counter ~92% positive label density
    # Approx ratio of negatives to positives per class: (1-0.92)/0.92 ~ 0.087
    se_pos_w = torch.tensor([0.09] * num_se, device=device)
    se_crit  = torch.nn.BCEWithLogitsLoss(pos_weight=se_pos_w)

    W_SEV, W_SE = CFG["severity_loss_weight"], CFG["se_loss_weight"]
    CLIP = CFG["grad_clip"]

    # ── Train / Eval ──────────────────────────────────────────────────────────
    def train_epoch():
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            sev_logits, se_logits, _ = model(batch.x_dict, batch.edge_index_dict)
            bs = batch["patient"].batch_size

            # Severity loss — skip NaN-labelled patients
            sev_y    = batch["patient"].y[:bs].view(-1, 1).float()
            vmask    = batch["patient"].valid_mask[:bs]
            sev_loss_all = sev_crit(sev_logits[:bs], sev_y)   # (bs, 1)
            sev_loss = sev_loss_all[vmask].mean() if vmask.any() else torch.tensor(0.0, device=device)

            # SE multi-label loss
            se_y    = batch["patient"].se_labels[:bs].float()
            se_loss = se_crit(se_logits[:bs], se_y)

            loss = W_SEV * sev_loss + W_SE * se_loss

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            sev_logits, se_logits, _ = model(batch.x_dict, batch.edge_index_dict)
            bs = batch["patient"].batch_size

            sev_y  = batch["patient"].y[:bs].view(-1, 1).float()
            vmask  = batch["patient"].valid_mask[:bs]
            se_y   = batch["patient"].se_labels[:bs].float()

            sev_loss_all = sev_crit(sev_logits[:bs], sev_y)
            sev_loss = sev_loss_all[vmask].mean() if vmask.any() else torch.tensor(0.0, device=device)
            se_loss  = se_crit(se_logits[:bs], se_y)
            loss = W_SEV * sev_loss + W_SE * se_loss

            if not torch.isnan(loss):
                total_loss += loss.item()

            # Accuracy only on valid (non-NaN-label) patients
            if vmask.any():
                preds   = (torch.sigmoid(sev_logits[:bs][vmask]) > 0.5).float()
                correct += (preds == sev_y[vmask]).sum().item()
                total   += vmask.sum().item()

        acc = correct / total if total > 0 else 0.0
        return total_loss / len(loader), acc

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("\n[5/5] Starting training...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8}")
    print("-" * 44)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch()
        val_loss, val_acc = evaluate(val_loader)
        scheduler.step()

        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(PROCESSED, "hgt_model.pth"))
            flag = " <- saved"

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_acc:>7.2%}{flag}")

    print(f"\nTraining complete. Best Epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Weights -> {os.path.join(PROCESSED, 'hgt_model.pth')}")

    with open(os.path.join(PROCESSED, "training_config.json"), "w") as f:
        json.dump(CFG, f, indent=2)
    print(f"Config  -> {os.path.join(PROCESSED, 'training_config.json')}")
