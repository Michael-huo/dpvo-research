"""Canonical budgeted admission, after every hidden FMap packet is generated."""
from __future__ import annotations

import numpy as np

from latent_vslam.protocol import canonical_sha256


ADMISSION_CONTRACT = {
    "type": "uniform_budgeted",
    "hidden_count": "n = stride - 1; complete anchor bracket",
    "k_rule": "K = n // 2",
    "ordinal_rule": "1 + ((2*j + 1)*n // (2*K)), j=0..K-1; one-based",
    "order": "strictly increasing hidden timestamp_ns",
    "zero_budget": "empty selection",
    "anchors": "all submitted to native DPVO",
    "generation": "all hidden predictor + H1 Bridge + FMap packets",
    "gate": "after FMap packet construction, before native packet conversion and DPVO insertion",
}


def uniform_ordinals(stride):
    if type(stride) is not int or stride < 2:
        raise ValueError("anchor stride must be an integer >= 2")
    n = stride - 1
    k = n // 2
    return tuple(1 + (2*j + 1)*n // (2*k) for j in range(k))


def uniform_hidden_keys(intervals, stride):
    ordinals = uniform_ordinals(stride)
    seen, selected = set(), set()
    for interval in intervals:
        hidden = sorted(interval.hidden, key=lambda q: q.identity.timestamp_ns)
        timestamps = [q.identity.timestamp_ns for q in hidden]
        if (len(hidden) != stride - 1 or len(set(timestamps)) != len(hidden)
                or not interval.anchor0.timestamp_ns < timestamps[0]
                or not timestamps[-1] < interval.anchor1.timestamp_ns):
            raise ValueError("uniform admission requires a complete ordered bracket")
        for ordinal, query in enumerate(hidden, 1):
            if query.ordinal != ordinal or query.identity.key in seen:
                raise ValueError("invalid hidden ordinal or duplicate bracket identity")
            seen.add(query.identity.key)
            if ordinal in ordinals:
                selected.add(query.identity.key)
    return frozenset(selected)


def validate_admission_trajectory(arrays, records, roles, stride):
    from prediction.predictor import build_anchor_intervals
    selected = uniform_hidden_keys(build_anchor_intervals(records, roles), stride)
    keys = [r.identity.key for r in records]
    bits = np.asarray([roles[key] == "anchor" or key in selected for key in keys])
    expected = np.asarray([r.identity.timestamp_ns for r, keep in zip(records, bits) if keep], dtype=np.uint64)
    if (not np.array_equal(arrays["admission_identities"], keys)
            or arrays["admission_insert"].dtype != np.bool_
            or not np.array_equal(arrays["admission_insert"], bits)
            or not np.array_equal(arrays["timestamps_ns"], expected)
            or arrays["poses"].shape != (len(expected), 7)
            or not np.isfinite(arrays["poses"]).all()):
        raise RuntimeError("trajectory must match the exact canonical admission population")
    return expected


class AdmissionReceipt:
    """Receipt/insertion accounting; node acceptance remains native DPVO's job."""

    def __init__(self, roles, intervals, stride):
        self.roles = dict(roles)
        if any(role not in {"anchor", "hidden"} for role in self.roles.values()):
            raise ValueError("invalid scheduled observation role")
        self.expected = list(roles)
        self.selected = uniform_hidden_keys(intervals, stride)
        hidden = [q.identity.key for i in intervals for q in i.hidden]
        if len(hidden) != len(set(hidden)) or set(hidden) != {
                key for key, role in roles.items() if role == "hidden"}:
            raise ValueError("brackets must cover exactly the scheduled hidden population")
        self.received, self.inserted, self.discarded = [], [], []
        self.inserted_timestamps = []
        self.previous_timestamp = -1

    def receive(self, identity):
        offset = len(self.received)
        if offset >= len(self.expected) or identity.key != self.expected[offset]:
            raise RuntimeError("duplicate, missing, or out-of-order observation")
        if identity.timestamp_ns <= self.previous_timestamp:
            raise RuntimeError("observation timestamps must strictly increase")
        self.previous_timestamp = identity.timestamp_ns
        self.received.append(identity.key)
        insert = self.roles[identity.key] == "anchor" or identity.key in self.selected
        (self.inserted if insert else self.discarded).append(identity.key)
        if insert:
            self.inserted_timestamps.append(identity.timestamp_ns)
        return insert

    def finish(self):
        if self.received != self.expected:
            raise RuntimeError("incomplete observation receipt population")
        hidden = [key for key in self.expected if self.roles[key] == "hidden"]
        return {
            "contract": dict(ADMISSION_CONTRACT),
            "generated_hidden_count": len(hidden),
            "inserted_hidden_count": len(self.selected),
            "received_count": len(self.received),
            "inserted_count": len(self.inserted),
            "discarded_count": len(self.discarded),
            "received_exactly_once": True, "partition_exact": True,
            "all_hidden_prediction_and_bridge_completed": True,
            "insertion_means": "track_packet_called_not_node_accepted",
            "admission_mask_sha256": canonical_sha256(
                [(key, key in self.selected) for key in hidden]),
        }

    def arrays(self):
        return {
            "admission_identities": np.asarray(self.received, dtype=str),
            "admission_insert": np.asarray([
                self.roles[key] == "anchor" or key in self.selected
                for key in self.received], dtype=bool),
        }
