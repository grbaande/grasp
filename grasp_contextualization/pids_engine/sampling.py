"""Turning an anomaly into a target + training-drawn candidate-sample pool,
and loading each sample's PyG neighborhood."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import cast

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from grasp.graph import graph_storage
from grasp.utils.graph_helpers import clean_graph_attributes_for_neighborloader, load_graph_data
from grasp_contextualization.shared import neighbor_fanout

from .models import Anomaly, Config, ExecWindowStats, Role, SampleRef


def executable_window_stats(gs: graph_storage.GraphStorage) -> defaultdict[str, ExecWindowStats]:
    stats: defaultdict[str, ExecWindowStats] = defaultdict(
        lambda: {"train_windows": [], "counts": [], "ids": []}
    )
    inv = {int(v): str(k) for k, v in gs.train_subject_cmd_to_id.items()}
    for path in gs.extended_train_data_paths:
        data: Data = load_graph_data(path)
        subject_indices = data.subject_mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        cmd_ids = data.y[data.subject_mask].argmax(dim=1).detach().cpu().tolist()  # type: ignore
        node_ids = [str(data.node_index_id[i]) for i in subject_indices]
        by_label: defaultdict[str, list[str]] = defaultdict(list)
        for cmd_id, node_id in zip(cmd_ids, node_ids):
            by_label[inv.get(int(cmd_id), f"UNKNOWN_{cmd_id}")].append(node_id)
        for label, ids in by_label.items():
            stats[label]["train_windows"].append(str(path))
            stats[label]["counts"].append(len(ids))
            stats[label]["ids"].append(ids)
    return stats


def label_count(stats: defaultdict[str, ExecWindowStats], label: str) -> int:
    return int(sum(stats[label]["counts"])) if label in stats else 0


def build_sample_refs(
    anom: Anomaly, stats: defaultdict[str, ExecWindowStats], c: Config, rng: random.Random
) -> list[SampleRef]:
    refs = [
        SampleRef(
            anom.process_id,
            "anomaly_0",
            anom.process_id,
            anom.data_path,
            "anomaly",
            anom.true_label,
            anom.data_path,
        )
    ]
    for role, label in [("true_sample", anom.true_label), ("pred_sample", anom.pred_label)]:
        candidates: list[tuple[str, str]] = []
        if label in stats:
            for window, ids in zip(stats[label]["train_windows"], stats[label]["ids"]):
                candidates.extend((str(pid), str(window)) for pid in ids)
        rng.shuffle(candidates)
        for i, (pid, window) in enumerate(candidates[: c.candidate_pool_per_label]):
            refs.append(
                SampleRef(
                    anom.process_id,
                    f"{role}_{i}_{pid}",
                    pid,
                    window,
                    cast(Role, role),
                    label,
                    anom.data_path,
                )
            )
    return refs


def load_neighbor_batch(ref: SampleRef, c: Config) -> Data:
    data: Data = load_graph_data(ref.data_path)
    _uuids, node_ids, cmd_labels = clean_graph_attributes_for_neighborloader(data)
    node_ids = [str(v) for v in node_ids]  # type: ignore
    cmd_labels = [str(v) for v in cmd_labels]  # type: ignore
    pos = node_ids.index(str(ref.process_id))
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # type: ignore
    mask[pos] = True
    sizes = neighbor_fanout(c.dataset, c.hop_mode)
    batch: Data = next(
        iter(
            NeighborLoader(
                data=data, num_neighbors=sizes, batch_size=1, input_nodes=mask, shuffle=False
            )
        )
    )
    try:
        sampled_ids = (
            batch.n_id.detach().cpu().tolist()
            if hasattr(batch, "n_id")
            else list(range(int(batch.num_nodes)))  # type: ignore
        )
        batch.cmd_label_list = [
            cmd_labels[int(i)] if int(i) < len(cmd_labels) else str(i) for i in sampled_ids
        ]
    except Exception:  # noqa: BLE001
        batch.cmd_label_list = []
    return batch
