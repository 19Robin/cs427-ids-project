from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd


# same order as the training data
FEATURES = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]

ETHERNET_HEADER_BYTES = 14
DEFAULT_WINDOW_SECONDS = 5.0


@dataclass(frozen=True)
class PacketInfo:
    """The few fields of one IP packet that the features need."""

    timestamp: float
    src: str
    dst: str
    protocol: str          # "tcp", "udp", "icmp" or "other"
    sport: int             # port, or ICMP identifier / type for ICMP
    dport: int
    ttl: int
    frame_bytes: int       # IP datagram length + 14 (Ethernet header)


@dataclass
class _Flow:
    originator: str
    protocol: str
    endpoints: tuple
    first_ts: float
    last_ts: float
    packets: int = 0
    total_bytes: int = 0
    ttl: float = np.nan
    ttl_seen_this_window: bool = False


def protocol_one_hot(protocol):
    """One-hot encoding identical to preprocess_data.py (5G-NIDD)."""
    return {
        "TCP": 1 if protocol == "tcp" else 0,
        "UDP": 1 if protocol == "udp" else 0,
        "ICMP": 1 if protocol == "icmp" else 0,
    }


def flow_rate(packet_count, duration):
    """Argus Rate: (TotPkts - 1) / Dur, 0 for one packet or zero duration."""
    if packet_count <= 1 or duration <= 0:
        return 0.0
    return (packet_count - 1) / duration


class FlowAggregator:

    def __init__(self, max_flows_per_window=5000, max_remembered_flows=20000):
        self.max_flows_per_window = max_flows_per_window
        self.max_remembered_flows = max_remembered_flows
        self._flows = {}
        # flow key -> (originator ip, originator TTL), kept across windows
        self._directions = OrderedDict()
        self.dropped_packets = 0

    @staticmethod
    def flow_key(packet):
        a = (packet.src, packet.sport)
        b = (packet.dst, packet.dport)
        return (packet.protocol,) + tuple(sorted([a, b]))

    def add(self, packet):
        key = self.flow_key(packet)
        flow = self._flows.get(key)

        if flow is None:
            if len(self._flows) >= self.max_flows_per_window:
                self.dropped_packets += 1
                return
            if key in self._directions:
                originator, remembered_ttl = self._directions[key]
                self._directions.move_to_end(key)
            else:
                originator, remembered_ttl = packet.src, packet.ttl
                self._directions[key] = (originator, remembered_ttl)
                if len(self._directions) > self.max_remembered_flows:
                    self._directions.popitem(last=False)
            flow = _Flow(
                originator=originator,
                protocol=packet.protocol,
                endpoints=key[1:],
                first_ts=packet.timestamp,
                last_ts=packet.timestamp,
                ttl=remembered_ttl,
            )
            self._flows[key] = flow

        flow.packets += 1
        flow.total_bytes += packet.frame_bytes
        flow.first_ts = min(flow.first_ts, packet.timestamp)
        flow.last_ts = max(flow.last_ts, packet.timestamp)

        # use the originator's first TTL in this window
        if packet.src == flow.originator and not flow.ttl_seen_this_window:
            flow.ttl = packet.ttl
            flow.ttl_seen_this_window = True
            self._directions[key] = (flow.originator, packet.ttl)

    def flush(self):
        records = []
        for flow in self._flows.values():
            duration = max(flow.last_ts - flow.first_ts, 0.0)
            (ip_a, port_a), (ip_b, port_b) = flow.endpoints
            if ip_a == flow.originator:
                src, sport, dst, dport = ip_a, port_a, ip_b, port_b
            else:
                src, sport, dst, dport = ip_b, port_b, ip_a, port_a
            record = {
                "Rate": flow_rate(flow.packets, duration),
                "Packet_Count": float(flow.packets),
                "Mean_Packet_Size": flow.total_bytes / flow.packets,
                "TTL": float(flow.ttl),
                **protocol_one_hot(flow.protocol),
                "Protocol": flow.protocol.upper(),
                "Source": src if flow.protocol == "icmp" else f"{src}:{sport}",
                "Destination": dst if flow.protocol == "icmp" else f"{dst}:{dport}",
                "Duration": duration,
                "Bytes": flow.total_bytes,
            }
            records.append(record)
        self._flows = {}
        return records


def records_to_model_input(records):
    """DataFrame with exactly the model's columns, in training order."""
    if not records:
        return pd.DataFrame(columns=FEATURES, dtype=float)
    frame = pd.DataFrame(records)
    return frame[FEATURES].astype(float)


def summarise_window(records, window_seconds):
    """Window-level totals shown in the dashboard's 'Current traffic' panel."""
    packets = sum(r["Packet_Count"] for r in records)
    total_bytes = sum(r["Bytes"] for r in records)
    ttls = pd.Series([r["TTL"] for r in records], dtype=float).dropna()
    return {
        "Flows": len(records),
        "Packet_Count": int(packets),
        "Traffic_Rate": packets / window_seconds if window_seconds > 0 else 0.0,
        "Mean_Packet_Size": total_bytes / packets if packets else 0.0,
        "TTL": float(ttls.mode().iloc[0]) if not ttls.empty else np.nan,
        "TCP": sum(r["TCP"] for r in records),
        "UDP": sum(r["UDP"] for r in records),
        "ICMP": sum(r["ICMP"] for r in records),
    }
