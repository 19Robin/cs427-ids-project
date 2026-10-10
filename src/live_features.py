"""
live_features.py

Turns live packets into the same seven features the IDS models were trained
on, in the same order and with the same meaning as src/preprocess_data.py.

WHAT THE MODEL WAS TRAINED ON
-----------------------------
Each 5G-NIDD record is an Argus *bidirectional flow record*, and Argus emits a
status record for an active flow every ~5 seconds (99.9% of 5G-NIDD records
have Dur <= 5 s). preprocess_data.py maps one record to:

    Rate              = Argus Rate
    Packet_Count      = TotPkts
    Mean_Packet_Size  = TotBytes / TotPkts
    TTL               = sTtl
    TCP / UDP / ICMP  = one-hot of Proto (all three 0 for any other protocol)

These were checked against the raw 5G-NIDD file (1,215,890 records):

  * Rate == (TotPkts - 1) / Dur for 100% of records with Dur > 0, and
    Rate == 0 for every single-packet / zero-duration record. It is a packet
    rate measured between the first and last packet, NOT TotPkts / window.
  * TotBytes counts whole frames including the 14-byte Ethernet header: the
    smallest TCP record is 54 bytes (14 + 20 IP + 20 TCP) and the smallest
    UDP / ICMP record is 42 bytes (14 + 20 IP + 8).
  * sTtl is the IP TTL of packets sent by the flow's *source* (originator).
  * Proto "icmp" and "ipv6-icmp" both map to ICMP = 1.

HOW EACH FEATURE IS COMPUTED FROM LIVE PACKETS
----------------------------------------------
Packets are grouped into bidirectional flows (same protocol and the same two
endpoints in either direction) inside one monitoring window. The default window
is 5 s, matching the Argus status interval. One flow in one window = one
record = one model prediction.

  Rate              (n - 1) / (t_last - t_first) for the flow's packets in the
                    window; 0 when the flow has one packet or zero duration.
  Packet_Count      n, packets of the flow in the window (both directions).
  Mean_Packet_Size  sum of frame sizes / n, where frame size = IP datagram
                    length + 14 (Ethernet header). Normalising to an Ethernet
                    frame keeps sizes comparable to 5G-NIDD even when capturing
                    on loopback / VPN adapters that have a different link header.
  TTL               IP TTL (IPv6: hop limit) of the first packet sent by the
                    flow's originator. The originator is the endpoint that sent
                    the first packet we ever saw for that flow (remembered
                    across windows, as Argus keeps a flow's direction).
  TCP / UDP / ICMP  one-hot of the IP protocol, exactly as in training:
                    TCP=1 for TCP, UDP=1 for UDP, ICMP=1 for ICMP and ICMPv6,
                    all 0 for any other IP protocol (e.g. SCTP, IGMP, GRE).

Non-IP frames (ARP, LLDP, ...) are ignored: they have no TTL, and in 5G-NIDD
such records were dropped by preprocess_data.py for the same reason.

This module has no Scapy dependency, so it can be tested and used by the
Demo Mode without packet capture installed.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Same names and order as src/preprocess_data.py, src/train_model.py and
# test_lightweight_models.py (also verified against model.feature_names_in_
# in live_predictor.py).
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
    """
    Collects packets for one monitoring window and turns them into
    flow records with the seven model features.

    Not thread-safe by itself; the caller (LiveMonitor) serialises access.
    """

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

        # The originator's first TTL in this window replaces the remembered one.
        if packet.src == flow.originator and not flow.ttl_seen_this_window:
            flow.ttl = packet.ttl
            flow.ttl_seen_this_window = True
            self._directions[key] = (flow.originator, packet.ttl)

    def flush(self):
        """
        Return the window's flow records (one dict per flow) and start a new
        window. Each record holds the seven FEATURES plus readable metadata.
        """
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
