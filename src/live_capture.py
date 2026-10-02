"""
live_capture.py

Live IDS proof-of-concept: packet capture (or a safe simulation), windowed
flow feature extraction (live_features.py) and classification with the
EXISTING trained model (live_predictor.py). The model is never retrained.

  Live Capture Mode   Scapy + Npcap capture packets from one local interface.
                      Capture is passive: nothing is sent on the network.
  Demo Mode           Generates SYNTHETIC packets in memory (nothing is sent on
                      the network) and runs them through the same feature
                      extraction and model. SIMULATION - NOT REAL TRAFFIC.

Work runs in background threads so the Streamlit page never blocks; the page
reads a snapshot of the rolling history (only recent observations are kept,
packets themselves are never stored).

Command-line helpers (run from the project root):

    python -m src.live_capture --list-interfaces
    python -m src.live_capture --test-capture "WiFi" --seconds 15
    python -m src.live_capture --test-demo --seconds 15
"""

import ipaddress
import platform
import random
import threading
import time
from collections import deque
from datetime import datetime

try:
    from .live_features import (
        DEFAULT_WINDOW_SECONDS, ETHERNET_HEADER_BYTES, FlowAggregator,
        PacketInfo, summarise_window,
    )
    from .live_predictor import LABELS, MALICIOUS
except ImportError:  # run directly as a script
    from live_features import (
        DEFAULT_WINDOW_SECONDS, ETHERNET_HEADER_BYTES, FlowAggregator,
        PacketInfo, summarise_window,
    )
    from live_predictor import LABELS, MALICIOUS


IS_WINDOWS = platform.system() == "Windows"

NPCAP_HELP = (
    "Install Npcap from https://npcap.com/#download (tick \"Install Npcap in "
    "WinPcap API-compatible Mode\"), then restart the terminal and Streamlit."
)
PERMISSION_HELP = (
    "Packet capture was refused. Either start the terminal (and Streamlit) with "
    "\"Run as administrator\", or reinstall Npcap with \"Restrict Npcap driver's "
    "access to Administrators only\" unticked."
    if IS_WINDOWS else
    "Packet capture needs root privileges (for example: sudo streamlit run app.py)."
)


class CaptureError(Exception):
    """A user-facing explanation of why capture could not start."""


# ============================================================
# CAPTURE AVAILABILITY / INTERFACES
# ============================================================

def capture_status():
    """
    Check whether live capture can work. Returns (available, message).
    Never raises: problems are returned as a message for the dashboard.
    """
    try:
        from scapy.all import conf
    except ImportError:
        return False, (
            "Scapy is not installed, so live capture is unavailable. Install it "
            "with `pip install -r requirements.txt` (or `pip install scapy`)."
        )
    except Exception as error:
        return False, f"Scapy could not be loaded: {error}"

    if IS_WINDOWS and not conf.use_pcap:
        return False, "Npcap was not found, so Scapy cannot capture packets. " + NPCAP_HELP

    return True, "Packet capture library available (Scapy" + (
        " + Npcap)." if IS_WINDOWS else ")."
    )


def list_interfaces():
    """
    Interfaces Scapy can capture on, most useful first (those with a normal
    IPv4 address). Each item: {name, description, ip, label}.
    """
    from scapy.all import get_working_ifaces

    interfaces = []
    for iface in get_working_ifaces():
        ip = iface.ip or ""
        description = iface.description or ""
        label = iface.name
        if description and description != iface.name:
            label += f" — {description}"
        if ip:
            label += f" ({ip})"
        interfaces.append({
            "name": iface.name,
            "description": description,
            "ip": ip,
            "label": label,
        })

    def rank(item):
        ip = item["ip"]
        if not ip:
            return 3
        if ip.startswith("169.254.") or ip.startswith("127."):
            return 2
        return 0 if "wi-fi" in item["label"].lower() or "wifi" in item["label"].lower() else 1

    return sorted(interfaces, key=rank)


def build_bpf_filter(host_filter):
    """IP traffic only; optionally only traffic to/from one device."""
    bpf = "ip or ip6"
    if host_filter:
        try:
            address = ipaddress.ip_address(host_filter.strip())
        except ValueError:
            raise CaptureError(
                f"'{host_filter}' is not a valid IP address for the device filter."
            )
        bpf = f"({bpf}) and host {address}"
    return bpf


def explain_capture_error(error, interface):
    text = str(error)
    lowered = text.lower()
    if any(word in lowered for word in ("permission", "access is denied", "operation not permitted")):
        return f"{PERMISSION_HELP} (details: {text})"
    if "no such device" in lowered or ("interface" in lowered and "not" in lowered):
        return f"Interface '{interface}' cannot be opened. Pick another interface. (details: {text})"
    if "npcap" in lowered or "winpcap" in lowered or "wpcap" in lowered:
        return f"{NPCAP_HELP} (details: {text})"
    if "error opening adapter" in lowered or "failed to open" in lowered:
        return (
            f"Npcap could not open interface '{interface}'. The adapter may be "
            f"disconnected, or capture may need administrator rights. "
            f"{PERMISSION_HELP} (details: {text})"
        )
    return f"Packet capture failed on '{interface}': {text}"


# ============================================================
# PACKET PARSING
# ============================================================

_IP_PROTOCOLS = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp"}


def packet_to_info(packet):
    """
    Convert one Scapy packet to PacketInfo, or None for non-IP frames.
    Raises on malformed packets (the caller counts and skips them).
    """
    from scapy.layers.inet import IP
    from scapy.layers.inet6 import IPv6

    if IP in packet:
        ip = packet[IP]
        ttl = int(ip.ttl)
        protocol_number = int(ip.proto)
        # ip.len can be 0 or wrong on captured outgoing packets when the
        # network card does segmentation offload; fall back to the bytes seen.
        ip_length = int(ip.len) if ip.len else len(ip)
    elif IPv6 in packet:
        ip = packet[IPv6]
        ttl = int(ip.hlim)
        protocol_number = int(ip.nh)
        ip_length = 40 + int(ip.plen) if ip.plen else len(ip)
    else:
        return None

    protocol = _IP_PROTOCOLS.get(protocol_number, "other")
    payload = ip.payload
    sport = dport = 0

    if protocol in ("tcp", "udp"):
        sport = int(getattr(payload, "sport", 0) or 0)
        dport = int(getattr(payload, "dport", 0) or 0)
    elif protocol == "icmp":
        # Echo request/reply share an identifier, so a ping and its reply
        # form one bidirectional flow (as in Argus). Other ICMP messages are
        # grouped by type/code.
        icmp_type = int(getattr(payload, "type", 0) or 0)
        if icmp_type in (0, 8, 128, 129):
            sport = dport = int(getattr(payload, "id", 0) or 0)
        else:
            sport = dport = icmp_type * 256 + int(getattr(payload, "code", 0) or 0)

    return PacketInfo(
        timestamp=float(packet.time),
        src=str(ip.src),
        dst=str(ip.dst),
        protocol=protocol,
        sport=sport,
        dport=dport,
        ttl=ttl,
        frame_bytes=ip_length + ETHERNET_HEADER_BYTES,
    )


# ============================================================
# DEMO MODE (SYNTHETIC PACKETS, NOTHING IS SENT)
# ============================================================

DEMO_SCENARIOS = {
    "auto": "Auto — normal traffic with periodic simulated suspicious bursts",
    "normal": "Normal traffic only",
    "suspicious": "Simulated suspicious traffic only",
}


class DemoTrafficGenerator:
    """
    Creates synthetic PacketInfo objects for one window. Profiles are modelled
    on 5G-NIDD flow statistics (medians per traffic type in the raw dataset),
    so the trained model sees values from the domain it was trained on.
    Addresses use the 10.99.0.0/16 documentation-style range and nothing is
    ever transmitted. The model decides benign/malicious - the generator does
    not label anything.
    """

    NORMAL_PROFILES = ("udp_stream", "udp_small", "icmp_ping", "tcp_session")
    SUSPICIOUS_PROFILES = ("syn_flood_like", "icmp_flood_like", "slow_rate_like", "scan_like")

    def __init__(self, scenario="auto", seed=None):
        self.scenario = scenario
        self.random = random.Random(seed)
        self.window_index = 0

    def _phase(self):
        if self.scenario == "normal":
            return None
        if self.scenario == "suspicious":
            return self.SUSPICIOUS_PROFILES[(self.window_index // 3) % 4]
        # auto: 7 normal windows, then a 3-window simulated burst, rotating type
        cycle, position = divmod(self.window_index, 10)
        if position >= 7:
            return self.SUSPICIOUS_PROFILES[cycle % 4]
        return None

    def generate(self, window_start, window_seconds):
        rnd = self.random
        packets = []
        burst = self._phase()

        for _ in range(rnd.randint(4, 10) if burst else rnd.randint(6, 14)):
            packets += self._flow(rnd.choice(self.NORMAL_PROFILES), window_start, window_seconds)
        if burst:
            for _ in range(rnd.randint(10, 25)):
                packets += self._flow(burst, window_start, window_seconds)

        self.window_index += 1
        packets.sort(key=lambda p: p.timestamp)
        return packets, burst

    def _flow(self, profile, window_start, window_seconds):
        rnd = self.random
        client = f"10.99.0.{rnd.randint(2, 254)}"
        server = f"10.99.1.{rnd.randint(2, 254)}"
        cport = rnd.randint(32768, 60999)

        # (protocol, server port, packets, duration, frame-size range,
        #  originator TTL, reply TTL, share of packets from originator)
        if profile == "udp_stream":
            count = rnd.randint(150, 230)
            spec = ("udp", 443, count, rnd.uniform(4.0, 4.9), (1150, 1400), 117, 64, 0.9)
        elif profile == "udp_small":
            spec = ("udp", 53, rnd.randint(1, 2), rnd.uniform(0.0, 0.05), (42, 90), rnd.choice([249, 255]), 64, 0.5)
        elif profile == "icmp_ping":
            spec = ("icmp", 0, rnd.randint(1, 2), rnd.uniform(0.0, 0.05), (98, 98), 58, 64, 0.5)
        elif profile == "tcp_session":
            spec = ("tcp", 443, rnd.randint(10, 40), rnd.uniform(1.0, 4.5), (66, 1400), 249, 64, 0.5)
        elif profile == "syn_flood_like":
            spec = ("tcp", 80, 3, rnd.uniform(0.015, 0.03), (54, 58), 63, 64, 1.0)
        elif profile == "icmp_flood_like":
            spec = ("icmp", 0, 2, rnd.uniform(0.0015, 0.0045), (42, 42), 63, 64, 1.0)
        elif profile == "slow_rate_like":
            spec = ("tcp", 80, rnd.randint(6, 12), rnd.uniform(2.5, 4.5), (66, 110), 63, 64, 0.6)
        else:  # scan_like
            spec = ("tcp", rnd.randint(1, 1024), 1, 0.0, (58, 58), rnd.randint(40, 52), 64, 1.0)

        protocol, sport_server, count, duration, (low, high), ttl_out, ttl_back, out_share = spec
        # Profiles describe 5 s windows; for shorter windows keep the packet
        # rate realistic by sending proportionally fewer packets.
        longest = window_seconds * 0.95
        if duration > longest:
            count = max(min(count, 2), round(count * longest / duration))
            duration = longest
        start = window_start + rnd.uniform(0, max(window_seconds - duration, 0.0))
        icmp_id = rnd.randint(1, 65535)

        packets = []
        for i in range(count):
            outgoing = i == 0 or rnd.random() < out_share
            t = start + (duration * i / (count - 1) if count > 1 else 0.0)
            if protocol == "icmp":
                ports = (icmp_id, icmp_id)
            else:
                ports = (cport, sport_server) if outgoing else (sport_server, cport)
            packets.append(PacketInfo(
                timestamp=t,
                src=client if outgoing else server,
                dst=server if outgoing else client,
                protocol=protocol,
                sport=ports[0],
                dport=ports[1],
                ttl=ttl_out if outgoing else ttl_back,
                frame_bytes=rnd.randint(low, high),
            ))
        return packets


# ============================================================
# LIVE MONITOR
# ============================================================

class LiveMonitor:
    """
    Background monitor. Every `window_seconds` the packets collected so far
    are turned into flow records, classified with the existing model and
    appended to a rolling history.
    """

    def __init__(self, predictor, mode="demo", window_seconds=DEFAULT_WINDOW_SECONDS,
                 interface=None, host_filter=None, demo_scenario="auto",
                 flow_history=500, window_history=240, idle_timeout=180):
        if mode not in ("capture", "demo"):
            raise ValueError("mode must be 'capture' or 'demo'")
        self.predictor = predictor
        self.mode = mode
        self.window_seconds = float(window_seconds)
        self.interface = interface
        self.host_filter = (host_filter or "").strip() or None
        self.idle_timeout = idle_timeout

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._aggregator = FlowAggregator()
        self._demo = DemoTrafficGenerator(demo_scenario) if mode == "demo" else None
        self._sniffer = None
        self._worker = None

        self.flows = deque(maxlen=flow_history)
        self.windows = deque(maxlen=window_history)
        self.latest_flows = []
        self.running = False
        self.error = None
        self.notice = None
        self.started_at = None
        self.window_count = 0
        self.packets_seen = 0
        self.non_ip_ignored = 0
        self.malformed = 0
        self.prediction_errors = 0
        self.empty_windows_in_row = 0
        self._last_poll = time.time()

    # --------------------------------------------------------
    # start / stop
    # --------------------------------------------------------

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self.notice = None

        if self.mode == "capture":
            self._start_capture()

        self.running = True
        self.started_at = datetime.now()
        self._last_poll = time.time()
        self._worker = threading.Thread(target=self._run, name="live-ids-worker", daemon=True)
        self._worker.start()

    def _start_capture(self):
        available, message = capture_status()
        if not available:
            raise CaptureError(message)
        if not self.interface:
            raise CaptureError("Select a network interface first.")
        names = [item["name"] for item in list_interfaces()]
        if self.interface not in names:
            raise CaptureError(
                f"Interface '{self.interface}' was not found. Available: {', '.join(names)}"
            )

        from scapy.all import AsyncSniffer

        try:
            self._sniffer = AsyncSniffer(
                iface=self.interface,
                filter=build_bpf_filter(self.host_filter),
                prn=self._on_packet,
                store=False,
            )
            self._sniffer.start()
        except CaptureError:
            raise
        except Exception as error:
            raise CaptureError(explain_capture_error(error, self.interface)) from error

        # The sniffer opens the adapter in its own thread: give it a moment
        # and report any failure now instead of failing silently later.
        time.sleep(0.8)
        failure = getattr(self._sniffer, "exception", None)
        if failure is not None or not self._sniffer.running:
            self._stop_sniffer()
            raise CaptureError(explain_capture_error(
                failure or "the capture thread stopped immediately", self.interface
            ))

    def stop(self, reason=None):
        self._stop.set()
        self._stop_sniffer()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=3)
        self.running = False
        if reason:
            self.notice = reason

    def _stop_sniffer(self):
        sniffer, self._sniffer = self._sniffer, None
        if sniffer is not None:
            try:
                if sniffer.running:
                    sniffer.stop(join=True)
            except Exception:
                pass

    def clear_history(self):
        with self._lock:
            self.flows.clear()
            self.windows.clear()
            self.latest_flows = []

    # --------------------------------------------------------
    # packet callback (sniffer thread)
    # --------------------------------------------------------

    def _on_packet(self, packet):
        try:
            info = packet_to_info(packet)
        except Exception:
            self.malformed += 1
            return
        if info is None:
            self.non_ip_ignored += 1
            return
        with self._lock:
            self.packets_seen += 1
            self._aggregator.add(info)

    # --------------------------------------------------------
    # worker thread
    # --------------------------------------------------------

    def _run(self):
        next_tick = time.monotonic() + self.window_seconds
        while not self._stop.wait(max(next_tick - time.monotonic(), 0)):
            next_tick += self.window_seconds

            if time.time() - self._last_poll > self.idle_timeout:
                self._stop_sniffer()
                self.running = False
                self.notice = "Monitoring stopped automatically because the dashboard was closed."
                return

            if self.mode == "capture" and self._sniffer is not None:
                failure = getattr(self._sniffer, "exception", None)
                if failure is not None or not self._sniffer.running:
                    self.error = explain_capture_error(
                        failure or "the capture thread stopped", self.interface
                    )
                    self._stop_sniffer()
                    self.running = False
                    return

            try:
                self._process_window()
            except Exception as error:  # keep the monitor alive
                self.prediction_errors += 1
                self.error = f"Window could not be processed: {error}"

    def _process_window(self):
        window_end = time.time()

        with self._lock:
            if self.mode == "demo":
                packets, _ = self._demo.generate(window_end - self.window_seconds, self.window_seconds)
                for packet in packets:
                    self.packets_seen += 1
                    self._aggregator.add(packet)
            records = self._aggregator.flush()

        try:
            predictions, probabilities = self.predictor.predict(records)
        except Exception as error:
            self.prediction_errors += 1
            self.error = f"Model prediction failed: {error}"
            return

        stamp = datetime.fromtimestamp(window_end)
        self.window_count += 1
        self.empty_windows_in_row = 0 if records else self.empty_windows_in_row + 1

        rows = []
        for record, prediction, probability in zip(records, predictions, probabilities):
            rows.append({
                "Time": stamp,
                "Window": self.window_count,
                **record,
                "Prediction": LABELS[int(prediction)],
                "Malicious_Probability": float(probability),
            })
        rows.sort(key=lambda r: r["Malicious_Probability"], reverse=True)

        summary = summarise_window(records, self.window_seconds)
        flagged = int(sum(int(p) == MALICIOUS for p in predictions))
        window_row = {
            "Time": stamp,
            "Window": self.window_count,
            **summary,
            "Flagged_Flows": flagged,
            "Max_Malicious_Probability": float(max(probabilities)) if len(probabilities) else 0.0,
            "Verdict": "NO TRAFFIC" if not records else ("MALICIOUS" if flagged else "BENIGN"),
        }

        with self._lock:
            self.latest_flows = rows
            self.flows.extend(reversed(rows))
            self.windows.append(window_row)

    # --------------------------------------------------------
    # dashboard access
    # --------------------------------------------------------

    def snapshot(self):
        """Copy of the current state for the dashboard (also a keep-alive)."""
        self._last_poll = time.time()
        with self._lock:
            return {
                "running": self.running,
                "mode": self.mode,
                "error": self.error,
                "notice": self.notice,
                "started_at": self.started_at,
                "window_seconds": self.window_seconds,
                "interface": self.interface,
                "host_filter": self.host_filter,
                "window_count": self.window_count,
                "packets_seen": self.packets_seen,
                "non_ip_ignored": self.non_ip_ignored,
                "malformed": self.malformed,
                "dropped": self._aggregator.dropped_packets,
                "prediction_errors": self.prediction_errors,
                "empty_windows_in_row": self.empty_windows_in_row,
                "windows": list(self.windows),
                "flows": list(self.flows),
                "latest_flows": list(self.latest_flows),
            }


# ============================================================
# COMMAND LINE
# ============================================================

def _main():
    import argparse

    try:
        from .live_predictor import LivePredictor
    except ImportError:
        from live_predictor import LivePredictor

    parser = argparse.ArgumentParser(description="Live IDS helpers")
    parser.add_argument("--list-interfaces", action="store_true")
    parser.add_argument("--test-capture", metavar="INTERFACE")
    parser.add_argument("--test-demo", action="store_true")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--host", default=None, help="only traffic to/from this IP")
    args = parser.parse_args()

    if args.list_interfaces:
        available, message = capture_status()
        print(message)
        if available:
            for item in list_interfaces():
                print(f"  {item['name']!r:34} {item['description']:45} {item['ip']}")
        return

    if args.test_capture or args.test_demo:
        monitor = LiveMonitor(
            LivePredictor(),
            mode="capture" if args.test_capture else "demo",
            window_seconds=args.window,
            interface=args.test_capture,
            host_filter=args.host,
        )
        monitor.start()
        try:
            seen = 0
            end = time.time() + args.seconds
            while time.time() < end and monitor.running:
                time.sleep(0.5)
                state = monitor.snapshot()
                for window in state["windows"][seen:]:
                    print(
                        f"{window['Time']:%H:%M:%S}  packets={window['Packet_Count']:5d}  "
                        f"flows={window['Flows']:4d}  rate={window['Traffic_Rate']:8.1f} pkt/s  "
                        f"flagged={window['Flagged_Flows']:3d}  "
                        f"max_p={window['Max_Malicious_Probability']:.2f}  {window['Verdict']}"
                    )
                seen = len(state["windows"])
        finally:
            monitor.stop()
        state = monitor.snapshot()
        print(
            f"packets={state['packets_seen']} non_ip={state['non_ip_ignored']} "
            f"malformed={state['malformed']} errors={state['prediction_errors']} "
            f"error={state['error']}"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    _main()
