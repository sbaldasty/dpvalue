"""File I/O for NoisyValue containers."""

import json
import numpy as np
import pandas as pd
import sympy as sp

from .consolidate import consolidate
from .core import (
    NoisyValue, NoisyFloat, NoisyInt, NoisyBool,
)
from .graph import DerivedNode
from .graph import LatentNode
from .graph import NoiseNode
from .pandas_ext import NoisyBoolArray, NoisyFloatArray, NoisyIntArray

_VERSION = 3

_TYPE_CLASSES = {
    "NoisyFloat": NoisyFloat,
    "NoisyInt": NoisyInt,
    "NoisyBool": NoisyBool,
}

_TYPE_NAMES = {v: k for k, v in _TYPE_CLASSES.items()}

_NOISY_COLUMN_CLASSES = {
    "noisyfloat": NoisyFloatArray,
    "noisyint": NoisyIntArray,
    "noisybool": NoisyBoolArray,
}

_SP_NAMESPACE = vars(sp)


# ── serialization ──────────────────────────────────────────────────────────────

def _flatten_container(container):
    flat = []

    def _walk(item):
        if isinstance(item, NoisyValue):
            flat.append(item)
        elif isinstance(item, np.ndarray):
            for v in item.flat:
                flat.append(v)
        elif isinstance(item, pd.DataFrame):
            for col in item.columns:
                arr = item[col].array
                if type(arr) in _NOISY_COLUMN_CLASSES.values():
                    for i in range(len(arr)):
                        if not arr._mask[i]:
                            flat.append(arr[i])
        elif isinstance(item, (list, tuple)):
            for sub in item:
                _walk(sub)

    _walk(container)
    return flat


def _rebuild_table(df, it):
    columns = {}
    for col in df.columns:
        arr = df[col].array
        if type(arr) not in _NOISY_COLUMN_CLASSES.values():
            columns[col] = df[col]
            continue
        values = [pd.NA if arr._mask[i] else next(it) for i in range(len(arr))]
        columns[col] = pd.Series(type(arr)._from_sequence(values), index=df.index)
    return pd.DataFrame(columns, index=df.index)


def _rebuild_container(container, it):
    if isinstance(container, NoisyValue):
        return next(it)
    if isinstance(container, np.ndarray):
        arr = np.empty(container.shape, dtype=object)
        for i in range(arr.size):
            arr.flat[i] = next(it)
        return arr
    if isinstance(container, pd.DataFrame):
        return _rebuild_table(container, it)
    if isinstance(container, (list, tuple)):
        rebuilt = [_rebuild_container(sub, it) for sub in container]
        return tuple(rebuilt) if isinstance(container, tuple) else rebuilt
    raise TypeError(f"Unsupported container type: {type(container)}")


def _node_name(node):
    """Return a serialization name for a node that is stable within a save() call."""
    if isinstance(node, DerivedNode):
        return f"derived_{id(node)}"
    return str(node.expr)


def _collect_nodes(container):
    nodes = {}

    def visit_value(v):
        for node in v._root.closure():
            name = _node_name(node)
            if name not in nodes:
                nodes[name] = node

    def visit(item):
        if isinstance(item, NoisyValue):
            visit_value(item)
        elif isinstance(item, np.ndarray):
            for v in item.flat:
                visit_value(v)
        elif isinstance(item, pd.DataFrame):
            for col in item.columns:
                arr = item[col].array
                if type(arr) in _NOISY_COLUMN_CLASSES.values():
                    for i in range(len(arr)):
                        if not arr._mask[i]:
                            visit_value(arr[i])
        elif isinstance(item, (list, tuple)):
            for sub in item:
                visit(sub)

    visit(container)
    return nodes


def _noise_node_params_to_dict(node):
    t = type(node).type_name
    if NoiseNode.registry.get(t) is not type(node):
        raise TypeError(f"Unknown NoiseNode type: {type(node)}")
    return {"type": t, "params": [sp.srepr(p) for p in node.params]}


def _node_to_dict(node):
    if isinstance(node, LatentNode):
        return {"kind": "latent"}
    if isinstance(node, NoiseNode):
        return {
            "kind": "noise",
            "source": _noise_node_params_to_dict(node),
            "deps": [_node_name(dep) for dep in node.deps],
        }
    if isinstance(node, DerivedNode):
        return {
            "kind": "derived",
            "definition": sp.srepr(node.expr),
            "constraints": [sp.srepr(c) for c in node.constraints],
            "deps": [_node_name(dep) for dep in node.deps],
        }
    raise TypeError(f"Unknown Node type: {type(node)}")


def _value_to_dict(v):
    return {
        "kind": "value",
        "type": _TYPE_NAMES[type(v)],
        "obs": v._obs,
        "root": _node_name(v._root),
    }


def _array_to_dict(arr):
    return {
        "kind": "array",
        "shape": list(arr.shape),
        "elements": [
            {"type": _TYPE_NAMES[type(v)], "obs": v._obs, "root": _node_name(v._root)}
            for v in arr.flat
        ],
    }


def _column_to_dict(series):
    arr = series.array
    for column_kind, cls in _NOISY_COLUMN_CLASSES.items():
        if type(arr) is cls:
            elements = [
                None if arr._mask[i]
                else {"obs": arr[i]._obs, "root": _node_name(arr[i]._root)}
                for i in range(len(arr))
            ]
            return {"kind": column_kind, "elements": elements}
    return {"kind": "plain", "dtype": str(series.dtype), "values": series.tolist()}


def _table_to_dict(df):
    return {
        "kind": "table",
        "columns": {col: _column_to_dict(df[col]) for col in df.columns},
    }


def _container_to_dict(container):
    if isinstance(container, NoisyValue):
        return _value_to_dict(container)
    if isinstance(container, np.ndarray):
        return _array_to_dict(container)
    if isinstance(container, pd.DataFrame):
        return _table_to_dict(container)
    if isinstance(container, (list, tuple)):
        kind = "tuple" if isinstance(container, tuple) else "list"
        items = []
        for item in container:
            if isinstance(item, NoisyValue):
                items.append(_value_to_dict(item))
            elif isinstance(item, np.ndarray):
                items.append(_array_to_dict(item))
            elif isinstance(item, pd.DataFrame):
                items.append(_table_to_dict(item))
            else:
                raise TypeError(
                    f"List/tuple items must be NoisyValue, ndarray, or DataFrame, got {type(item)}"
                )
        return {"kind": kind, "items": items}
    raise TypeError(f"Unsupported container type: {type(container)}")


def save(path, container):
    """Save a NoisyValue, ndarray of NoisyValues, or list/tuple of either to a JSON file."""
    flat = _flatten_container(container)
    consolidated = consolidate(*flat)
    container = _rebuild_container(container, iter(consolidated))
    nodes = _collect_nodes(container)
    doc = {
        "version": _VERSION,
        "nodes": {name: _node_to_dict(node) for name, node in nodes.items()},
        "container": _container_to_dict(container),
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


# ── deserialization ────────────────────────────────────────────────────────────

def _topo_sort(nodes_dict):
    visited = set()
    order = []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        for dep in nodes_dict[name].get("deps", []):
            visit(dep)
        order.append(name)

    for name in nodes_dict:
        visit(name)
    return order


def _parse_expr(s, name_map):
    expr = eval(s, _SP_NAMESPACE)  # noqa: S307 — we wrote the file
    for old, new_sym in name_map.items():
        expr = expr.subs(sp.Symbol(old), new_sym)
    return expr


def _load_noise_node(source_dict, name_map, deps=()):
    t = source_dict["type"]
    cls = NoiseNode.registry.get(t)
    if cls is None:
        raise ValueError(f"Unknown source type: {t!r}")
    params = [_parse_expr(p, name_map) for p in source_dict["params"]]
    return cls(params, deps)


def _load_nodes(nodes_dict):
    order = _topo_sort(nodes_dict)
    name_map = {}  # old symbol name str -> new Symbol
    built = {}     # old symbol name str -> Node

    for old_name in order:
        nd = nodes_dict[old_name]
        kind = nd["kind"]
        deps = [built[dep_name] for dep_name in nd.get("deps", [])]

        def remap(s, _map=name_map):
            return _parse_expr(s, _map)

        if kind == "latent":
            node = LatentNode()
            name_map[old_name] = node.expr
        elif kind == "noise":
            node = _load_noise_node(nd["source"], name_map, deps=deps)
            name_map[old_name] = node.expr
        elif kind == "derived":
            node = DerivedNode(
                remap(nd["definition"]),
                constraints=[remap(c) for c in nd["constraints"]],
                deps=deps,
            )
        else:
            raise ValueError(f"Unknown node kind: {kind!r}")

        built[old_name] = node

    return built


def _load_element(edict, built):
    cls = _TYPE_CLASSES[edict["type"]]
    return cls(edict["obs"], built[edict["root"]])


def _load_array(adict, built):
    shape = tuple(adict["shape"])
    arr = np.empty(shape, dtype=object)
    for i, edict in enumerate(adict["elements"]):
        arr.flat[i] = _load_element(edict, built)
    return arr


def _load_column(cdict, built):
    if cdict["kind"] == "plain":
        return pd.Series(cdict["values"], dtype=cdict["dtype"])
    scalar_cls = _NOISY_COLUMN_CLASSES[cdict["kind"]]._scalar_cls
    values = [
        pd.NA if e is None else scalar_cls(e["obs"], built[e["root"]])
        for e in cdict["elements"]
    ]
    array_cls = _NOISY_COLUMN_CLASSES[cdict["kind"]]
    return pd.Series(array_cls._from_sequence(values))


def _load_table(tdict, built):
    columns = {name: _load_column(col, built) for name, col in tdict["columns"].items()}
    return pd.DataFrame(columns)


def _load_container(cdict, built):
    kind = cdict["kind"]
    if kind == "value":
        return _load_element(cdict, built)
    if kind == "array":
        return _load_array(cdict, built)
    if kind == "table":
        return _load_table(cdict, built)
    if kind in ("list", "tuple"):
        loaders = {"value": _load_element, "array": _load_array, "table": _load_table}
        items = [loaders[item["kind"]](item, built) for item in cdict["items"]]
        return tuple(items) if kind == "tuple" else items
    raise ValueError(f"Unknown container kind: {kind!r}")


def load(path):
    """Load a container saved by save()."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("version") != _VERSION:
        raise ValueError(f"Unsupported file version: {doc.get('version')!r}")
    built = _load_nodes(doc["nodes"])
    return _load_container(doc["container"], built)
