"""Edge-vocabulary mapping and loading a raw PyG neighborhood for one process."""

from __future__ import annotations

import json
from enum import EnumMeta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
from torch_geometric.loader import NeighborLoader

import grasp.config as grasp_config
from grasp.utils.graph_helpers import clean_graph_attributes_for_neighborloader, load_graph_data
from grasp_contextualization.shared import neighbor_fanout

# -----------------------------------------------------------------------------
# Operation mapping and raw graph helpers
# -----------------------------------------------------------------------------


def enum_map(obj: Any) -> dict[int, str]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        out: dict[int, str] = {}
        for k, v in obj.items():
            try:
                out[int(k)] = str(v)
            except Exception:  # noqa: BLE001, S110
                pass
        return out
    if isinstance(obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(obj)}
    if isinstance(obj, EnumMeta) or hasattr(obj, "__members__"):
        out: dict[int, str] = {}
        for i, (name, member) in enumerate(obj.__members__.items()):
            try:
                out[int(getattr(member, "value", i))] = str(name)
            except Exception:  # noqa: BLE001
                out[i] = str(name)
        return out
    return {}


def select_op_map(vocab: str, dataset: str) -> tuple[str, dict[int, str]]:
    if grasp_config is None:
        return "none", {}
    ds = dataset.lower()
    choices = {
        "atlasv2": "ATLASV2_EDR_OPERATIONS",
        "carbanakv2": "CARBANAKV2_EDR_OPERATIONS",
        "spade_grasp": "SPADE_GRASP_OPERATIONS",
        "sysdig": "SYSDIG_OPERATIONS",
        "tc": "TC_OPERATIONS",
        "optc": "OPTC_OPERATIONS",
    }
    if vocab == "auto":
        # Mirrors grasp.utils.graph_helpers.get_operations()'s dataset-name mapping.
        if ds.startswith("optc"):
            name = "OPTC_OPERATIONS"
        elif ds.startswith("atlasv2_edr"):
            name = "ATLASV2_EDR_OPERATIONS"
        elif ds.startswith("carbanakv2_edr"):
            name = "CARBANAKV2_EDR_OPERATIONS"
        elif ds.startswith("spade_grasp"):
            name = "SPADE_GRASP_OPERATIONS"
        elif ds.startswith("sysdig"):
            name = "SYSDIG_OPERATIONS"
        else:
            name = "TC_OPERATIONS"
    else:
        name = choices.get(vocab, "")
    return (name or "none"), enum_map(getattr(grasp_config, name, None)) if name else {}


def resolve_op_map_json(vocab: str, dataset: str) -> tuple[str, dict[int, str], str]:
    """select_op_map plus the JSON encoding load_raw_batch needs, in one call."""
    op_name, op_map = select_op_map(vocab, dataset)
    op_json = json.dumps({str(k): v for k, v in op_map.items()})
    return op_name, op_map, op_json


def resolve_path(path: str, project_root: str) -> str:
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    candidates = [Path(project_root) / p, Path.cwd() / p]
    candidates.extend(parent / p for parent in Path(__file__).resolve().parents)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(p)


def decode_edge_attr(row: Any, op_map: dict[int, str]) -> tuple[int | None, str]:
    try:
        if torch is not None and isinstance(row, torch.Tensor):
            t = row.detach().cpu()
            idx = (
                int(t.reshape(-1)[0].item())
                if t.ndim == 0 or t.numel() == 1
                else int(t.argmax().item())
            )
        else:
            idx = int(row)
        return idx, op_map.get(idx, f"EDGE_TYPE_{idx}")
    except Exception:  # noqa: BLE001
        return None, "UNKNOWN"


def node_kind(row: pd.Series) -> str:
    if bool(row.get("is_target", False)):
        return "target"
    if bool(row.get("is_subject", False)):
        return "subject"
    if bool(row.get("is_file", False)):
        return "file"
    if bool(row.get("is_netflow", False)):
        return "netflow"
    if bool(row.get("is_memory", False)):
        return "memory"
    return "other"


@st.cache_resource(show_spinner=True)
def load_raw_batch(
    process_id: str, data_path: str, hop_mode: str, op_map_json: str, project_root: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        torch is None
        or NeighborLoader is None
        or load_graph_data is None
        or clean_graph_attributes_for_neighborloader is None
    ):
        raise RuntimeError("Missing torch, torch_geometric, or GRASP graph helpers")
    op_map = {int(k): str(v) for k, v in json.loads(op_map_json).items()}
    data = load_graph_data(resolve_path(data_path, project_root))
    node_uuids, node_ids, cmd_labels = clean_graph_attributes_for_neighborloader(data)
    node_ids = [str(v) for v in node_ids]  # type: ignore
    cmd_labels = [str(v) for v in cmd_labels]  # type: ignore
    pos = node_ids.index(str(process_id))
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # type: ignore
    mask[pos] = True
    batch = next(
        iter(
            NeighborLoader(
                data=data,
                num_neighbors=neighbor_fanout(data_path, hop_mode),
                batch_size=1,
                input_nodes=mask,
                shuffle=False,
            )
        )
    )
    sampled = (
        batch.n_id.detach().cpu().tolist()
        if hasattr(batch, "n_id")
        else list(range(int(batch.num_nodes)))
    )
    counts = (
        [int(v) for v in batch.num_sampled_nodes] if hasattr(batch, "num_sampled_nodes") else []
    )
    has_x = hasattr(batch, "x") and batch.x is not None and batch.x.ndim >= 2
    rows = []
    for i in range(int(batch.num_nodes)):
        if i == 0:
            hop = 0
        elif len(counts) >= 2 and i <= counts[1]:
            hop = 1
        elif len(counts) >= 3:
            hop = 2
        else:
            hop = -1
        orig = sampled[i]
        assert batch.x is not None
        rows.append(
            {
                "local_idx": i,
                "original_idx": orig,
                "node_index_id": node_ids[orig] if orig < len(node_ids) else str(orig),
                "uuid": str(node_uuids[orig]) if orig < len(node_uuids) else "",  # type: ignore
                "display_exec_full": cmd_labels[orig] if orig < len(cmd_labels) else str(orig),
                "is_target": i == 0,
                "hop": hop,
                "is_subject": bool(batch.x[i][0].item() == 1.0)  # type: ignore
                if has_x and batch.x.shape[1] > 0
                else False,
                "is_file": bool(batch.x[i][1].item() == 1.0)  # type: ignore
                if has_x and batch.x.shape[1] > 1
                else False,
                "is_netflow": bool(batch.x[i][2].item() == 1.0)
                if has_x and batch.x.shape[1] > 2
                else False,
                "is_memory": bool(batch.x[i][3].item() == 1.0)
                if has_x and batch.x.shape[1] > 3
                else False,
            }
        )
    nodes = pd.DataFrame(rows)
    nodes["kind"] = nodes.apply(node_kind, axis=1)
    edge_attr = getattr(batch, "edge_attr", None)
    edge_attr = edge_attr.detach().cpu() if edge_attr is not None else None
    edge_rows = []
    for edge_id, (src, dst) in enumerate(batch.edge_index.detach().cpu().t().tolist()):
        src, dst = int(src), int(dst)
        op_id, op_name = decode_edge_attr(
            edge_attr[edge_id] if edge_attr is not None and edge_id < edge_attr.shape[0] else None,
            op_map,
        )
        src_row = nodes.iloc[src] if src < len(nodes) else pd.Series(dtype=object)
        dst_row = nodes.iloc[dst] if dst < len(nodes) else pd.Series(dtype=object)
        edge_rows.append(
            {
                "edge_id": edge_id,
                "src": src,
                "dst": dst,
                "src_exec": src_row.get("display_exec_full", ""),
                "dst_exec": dst_row.get("display_exec_full", ""),
                "src_kind": src_row.get("kind", ""),
                "dst_kind": dst_row.get("kind", ""),
                "touches_target": src == 0 or dst == 0,
                "edge_type_id": op_id,
                "edge_type": op_name,
                "edge_signature": f"{src_row.get('kind', '?')}->{dst_row.get('kind', '?')}:{op_name}",
                "exec_edge_signature": f"{src_row.get('display_exec_full', '?')}->{dst_row.get('display_exec_full', '?')}:{op_name}",
            }
        )
    return nodes, pd.DataFrame(edge_rows)


def load_neighborhood_or_error(
    process_id: str,
    data_path: str,
    hop_mode: str,
    op_json: str,
    project_root: str,
    error_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """load_raw_batch wrapped with the try/except + st.error every page needs."""
    try:
        return load_raw_batch(process_id, data_path, hop_mode, op_json, project_root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load {error_label}: {exc}")
        return None
