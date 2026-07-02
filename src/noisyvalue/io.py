"""File I/O for NoisyValue containers."""

import json
import numpy as np
import sympy as sp

from .consolidate import consolidate
from .core import (
    NoisyValue, NoisyFloat, NoisyInt, NoisyBool,
)
from .graph import BinomialNode
from .graph import DerivedNode
from .graph import LatentNode
from .graph import NormalNode
from .graph import NoiseNode
from .util import fresh_name

_VERSION = 2

_TYPE_CLASSES = {
    "NoisyFloat": NoisyFloat,
    "NoisyInt": NoisyInt,
    "NoisyBool": NoisyBool,
}

_TYPE_NAMES = {v: k for k, v in _TYPE_CLASSES.items()}

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
        elif isinstance(item, (list, tuple)):
            for sub in item:
                _walk(sub)

    _walk(container)
    return flat


def _rebuild_container(container, it):
    if isinstance(container, NoisyValue):
        return next(it)
    if isinstance(container, np.ndarray):
        arr = np.empty(container.shape, dtype=object)
        for i in range(arr.size):
            arr.flat[i] = next(it)
        return arr
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
        elif isinstance(item, (list, tuple)):
            for sub in item:
                visit(sub)

    visit(container)
    return nodes


_NOISE_NODE_TYPE_NAMES = {NormalNode: "normal", BinomialNode: "binomial"}
_NOISE_NODE_TYPES = {v: k for k, v in _NOISE_NODE_TYPE_NAMES.items()}


def _noise_node_params_to_dict(node):
    t = _NOISE_NODE_TYPE_NAMES.get(type(node))
    if t is None:
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


def _container_to_dict(container):
    if isinstance(container, NoisyValue):
        return _value_to_dict(container)
    if isinstance(container, np.ndarray):
        return _array_to_dict(container)
    if isinstance(container, (list, tuple)):
        kind = "tuple" if isinstance(container, tuple) else "list"
        items = []
        for item in container:
            if isinstance(item, NoisyValue):
                items.append(_value_to_dict(item))
            elif isinstance(item, np.ndarray):
                items.append(_array_to_dict(item))
            else:
                raise TypeError(
                    f"List/tuple items must be NoisyValue or ndarray, got {type(item)}"
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
    cls = _NOISE_NODE_TYPES.get(t)
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


def _load_container(cdict, built):
    kind = cdict["kind"]
    if kind == "value":
        return _load_element(cdict, built)
    if kind == "array":
        return _load_array(cdict, built)
    if kind in ("list", "tuple"):
        items = [
            _load_element(item, built) if item["kind"] == "value" else _load_array(item, built)
            for item in cdict["items"]
        ]
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
