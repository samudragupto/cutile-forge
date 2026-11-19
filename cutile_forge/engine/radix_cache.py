"""Radix-tree prefix KV-cache for zero-recompute shared prompt prefixes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch


@dataclass
class _RadixNode:
    token_id: int
    children: dict[int, _RadixNode] = field(default_factory=dict)
    entry_key: tuple[int, ...] | None = None


@dataclass
class PrefixCacheEntry:
    """Cached KV tensors per layer and final prefill logits for a token prefix."""

    token_ids: tuple[int, ...]
    kv_per_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    last_logits: torch.Tensor


class RadixPrefixCache:
    """Prefix-tree KV-cache with LRU eviction and hit-rate telemetry."""

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max(1, max_entries)
        self._root = _RadixNode(token_id=-1)
        self._entries: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()
        self.lookups: int = 0
        self.hits: int = 0
        self.partial_hits: int = 0

    @property
    def prefix_cache_hit_rate(self) -> float:
        return self.hits / float(max(1, self.lookups))

    def match_prefix(
        self, token_ids: list[int]
    ) -> tuple[int, dict[int, tuple[torch.Tensor, torch.Tensor]] | None, torch.Tensor | None]:
        """Find the longest cached prefix of `token_ids`."""
        self.lookups += 1
        node = self._root
        best_key: tuple[int, ...] | None = None
        for tok in token_ids:
            nxt = node.children.get(tok)
            if nxt is None:
                break
            node = nxt
            if node.entry_key is not None and node.entry_key in self._entries:
                best_key = node.entry_key

        if best_key is None:
            return 0, None, None

        self._entries.move_to_end(best_key)
        entry = self._entries[best_key]
        self.hits += 1
        if len(best_key) < len(token_ids):
            self.partial_hits += 1
        return len(best_key), entry.kv_per_layer, entry.last_logits

    def insert(
        self,
        token_ids: list[int],
        kv_per_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
        last_logits: torch.Tensor,
    ) -> None:
        """Insert or refresh a cached prefix entry with LRU eviction."""
        key = tuple(token_ids)
        if not key:
            return

        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = PrefixCacheEntry(
                token_ids=key,
                kv_per_layer={
                    l_idx: (k.detach().clone(), v.detach().clone())
                    for l_idx, (k, v) in kv_per_layer.items()
                },
                last_logits=last_logits.detach().clone(),
            )
            return

        while len(self._entries) >= self.max_entries:
            old_key, _ = self._entries.popitem(last=False)
            self._clear_tree_key(old_key)

        node = self._root
        for tok in key:
            if tok not in node.children:
                node.children[tok] = _RadixNode(token_id=tok)
            node = node.children[tok]
        node.entry_key = key
        self._entries[key] = PrefixCacheEntry(
            token_ids=key,
            kv_per_layer={
                l_idx: (k.detach().clone(), v.detach().clone())
                for l_idx, (k, v) in kv_per_layer.items()
            },
            last_logits=last_logits.detach().clone(),
        )

    def _clear_tree_key(self, key: tuple[int, ...]) -> None:
        node = self._root
        for tok in key:
            nxt = node.children.get(tok)
            if nxt is None:
                return
            node = nxt
        if node.entry_key == key:
            node.entry_key = None

    def clear(self) -> None:
        self._root = _RadixNode(token_id=-1)
        self._entries.clear()
        self.lookups = 0
        self.hits = 0
        self.partial_hits = 0
