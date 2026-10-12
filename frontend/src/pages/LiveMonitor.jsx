import { useCallback, useEffect, useRef, useState } from "react";
import { Download, FlaskConical, Pause, Play, Square, Trash2 } from "lucide-react";
import { Badge, Button, Card, Callout, EmptyState, PageHeader, Section, Segmented, Select, Stat, Table } from "../components/ui";
import { api } from "../lib/api";
import LiveCharts from "../components/LiveCharts";
import { COLORS, num } from "../lib/format";

const time = (iso) => new Date(iso).getTime();
const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function ExportMenu({ disabled }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <Button variant="ghost" icon={Download} onClick={() => setOpen(!open)} disabled={disabled}>Export</Button>
      {open && !disabled && (
        <div className="absolute right-0 z-20 mt-2 w-64 overflow-hidden rounded-xl border border-line bg-white shadow-lg" onMouseLeave={() => setOpen(false)}>
          {[
            { href: "/api/live/export/windows.csv", label: "Windows (CSV)", hint: "one row per monitoring window" },
            { href: "/api/live/export/flows.csv", label: "Flows (CSV)", hint: "every classified flow with its 7 features" },
          ].map((item) => (
            <a key={item.href} href={item.href} download onClick={() => setOpen(false)} className="block px-4 py-2.5 hover:bg-slate-50">
              <div className="text-sm font-semibold text-ink">{item.label}</div>
              <div className="text-xs text-muted">{item.hint}</div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export default function LiveMonitor({ status }) {
  const live = status?.live;
  const [mode, setMode] = useState("demo");
  const [scenario, setScenario] = useState("auto");
  const [windowSeconds, setWindowSeconds] = useState(5);
  const [interfaces, setInterfaces] = useState(null);
  const [iface, setIface] = useState("");
  const [host, setHost] = useState("");
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [windows, setWindows] = useState([]);
  const poll = useRef(null);
  const history = useRef({ session: null, since: 0, windows: [] });

  // the server only sends windows newer than `since`; the full history is kept here
  const refresh = useCallback(async () => {
    try {
      const h = history.current;
      const next = await api.liveState(h.since);
      if (next.active) {
        if (next.started_at !== h.session || next.window_count < h.since) {
          const fetchedFrom = h.since;
          h.session = next.started_at;
          h.since = 0;
          h.windows = [];
          if (fetchedFrom > 0) Object.assign(next, await api.liveState(0));
        }
        const fresh = next.windows.filter((w) => w.Window > h.since);
        if (fresh.length) {
          h.windows = [...h.windows, ...fresh].slice(-2000);
          h.since = fresh[fresh.length - 1].Window;
        }
        setWindows(h.windows);
      }
      setState(next);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  // poll while monitoring is running
  useEffect(() => {
    if (live?.imports_ok) refresh();
  }, [live, refresh]);
  useEffect(() => {
    clearInterval(poll.current);
    if (state?.running) poll.current = setInterval(refresh, 1000);
    return () => clearInterval(poll.current);
  }, [state?.running, refresh]);

  useEffect(() => {
    if (mode === "capture" && interfaces === null && live?.imports_ok) {
      api.liveInterfaces().then((r) => {
        setInterfaces(r);
        if (r.interfaces?.length) setIface(r.interfaces[0].name);
      }).catch((e) => setInterfaces({ available: false, message: e.message, interfaces: [] }));
    }
  }, [mode, interfaces, live]);

  if (live && !live.imports_ok) {
    return (
      <>
        <PageHeader eyebrow="Application" title="Live IDS monitoring" />
        <Callout tone="danger" title="Live IDS modules could not be loaded">{live.error}</Callout>
      </>
    );
  }
  if (status && !status.models.rf10) {
    return (
      <>
        <PageHeader eyebrow="Application" title="Live IDS monitoring" />
        <EmptyState title="Model file not available" commands={["python src/test_lightweight_models.py"]}>
          Live monitoring uses models/random_forest_10_trees.joblib, which is git-ignored. Create it locally first.
        </EmptyState>
      </>
    );
  }

  const running = !!state?.running;
  const captureReady = mode === "demo" || (interfaces?.available && interfaces.interfaces.length > 0);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      await api.liveStart({
        mode,
        scenario,
        window_seconds: windowSeconds,
        interface: mode === "capture" ? iface : null,
        host_filter: mode === "capture" ? host || null : null,
      });
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function stop() {
    await api.liveStop().catch((e) => setError(e.message));
    refresh();
  }
  async function clear() {
    await api.liveClear().catch((e) => setError(e.message));
    history.current.windows = [];
    setWindows([]);
    refresh();
  }

  const latest = windows[windows.length - 1];
  const isDemo = state?.mode === "demo";

  return (
    <>
      <PageHeader eyebrow="Application" title="Live IDS monitoring">
        The trained 5G-NIDD model analysing network traffic in a controlled test environment: packets → flows → the same seven features →
        the existing model → a prediction. The model is never retrained during monitoring.
      </PageHeader>

      {/* Controls */}
      <Card className="mb-6">
        <div className="mb-4 flex flex-wrap items-center gap-3">
          {running ? (
            <Badge tone="green"><span className="pulse-dot h-2 w-2 rounded-full bg-emerald-500" /> Monitoring</Badge>
          ) : (
            <Badge tone="slate"><Pause className="h-3 w-3" /> Stopped</Badge>
          )}
          {running && isDemo && <Badge tone="amber"><FlaskConical className="h-3 w-3" /> Simulation — not real network traffic</Badge>}
        </div>

        <div className="flex flex-wrap items-end gap-5">
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">Mode</div>
            <Segmented
              value={mode}
              onChange={setMode}
              disabled={running}
              options={[{ value: "capture", label: "Live capture" }, { value: "demo", label: "Demo (simulation)" }]}
            />
          </div>

          {mode === "demo" && live?.scenarios && (
            <div className="w-80">
              <Select
                label="Simulated traffic"
                value={scenario}
                onChange={setScenario}
                disabled={running}
                options={Object.entries(live.scenarios).map(([value, label]) => ({ value, label }))}
              />
            </div>
          )}

          {mode === "capture" && interfaces?.available && interfaces.interfaces.length > 0 && (
            <>
              <div className="w-80">
                <Select
                  label="Network interface"
                  value={iface}
                  onChange={setIface}
                  disabled={running}
                  options={interfaces.interfaces.map((i) => ({ value: i.name, label: i.label }))}
                />
              </div>
              <label className="block w-64 text-sm">
                <span className="mb-1.5 block text-xs font-medium text-muted">Only this device IP (optional)</span>
                <input
                  value={host}
                  disabled={running}
                  onChange={(e) => setHost(e.target.value)}
                  placeholder="e.g. 192.168.137.25"
                  className="w-full rounded-xl border border-line px-3 py-2 text-sm shadow-sm outline-none focus:border-fiveg focus:ring-2 focus:ring-fiveg/20"
                />
              </label>
            </>
          )}

          <label className="block">
            <div className="mb-1.5 text-xs font-medium text-muted">Window: {windowSeconds} s</div>
            <input type="range" min={1} max={30} value={windowSeconds} disabled={running} onChange={(e) => setWindowSeconds(Number(e.target.value))} className="w-40 accent-[#2a78d6]" />
          </label>

          <div className="ml-auto flex gap-2">
            {running ? (
              <Button variant="danger" icon={Square} onClick={stop}>Stop</Button>
            ) : (
              <Button variant="blue" icon={Play} onClick={start} disabled={busy || !captureReady}>
                {busy ? "Starting…" : "Start monitoring"}
              </Button>
            )}
            <Button variant="ghost" icon={Trash2} onClick={clear} disabled={!state?.active}>Clear</Button>
            <ExportMenu disabled={!windows.length} />
          </div>
        </div>

        {mode === "capture" && interfaces && !interfaces.available && (
          <Callout tone="warning" className="mt-4" title="Live capture is not available">
            {interfaces.message} You can still present the dashboard with Demo mode.
          </Callout>
        )}
        {mode === "demo" && (
          <Callout tone="warning" className="mt-4" title="Simulation — not real network traffic">
            Demo mode creates synthetic packets in memory (nothing is sent on any network) and feeds them through the same feature extraction
            and trained model. Suspicious bursts imitate 5G-NIDD attack statistics; the benign/malicious decision is made by the model.
          </Callout>
        )}
      </Card>

      {error && <Callout tone="danger" className="mb-6">{error}</Callout>}
      {state?.error && <Callout tone="danger" className="mb-6">{state.error}</Callout>}
      {state?.notice && <Callout tone="info" className="mb-6">{state.notice}</Callout>}

      {!state?.active && (
        <Card className="py-10 text-center text-sm text-muted">Press <b className="text-ink">Start monitoring</b> to begin. Results appear after the first window.</Card>
      )}

      {state?.active && (
        <>
          <p className="mb-4 text-xs text-muted">
            {isDemo ? "Simulation — synthetic packets" : `Interface ${state.interface}${state.host_filter ? `, device ${state.host_filter}` : ""}`} ·
            window {state.window_seconds} s · {state.window_count} windows · {num(state.packets_seen)} IP packets · {num(state.non_ip_ignored)} non-IP frames
            ignored · {num(state.malformed)} malformed skipped
          </p>

          {state.mode === "capture" && state.empty_windows_in_row >= 2 && (
            <Callout tone="warning" className="mb-6" title="No packets received">
              Nothing arrived in the last {state.empty_windows_in_row} windows. Check that the interface carries the test traffic and that the device
              IP filter is correct, then generate some traffic (e.g. open a web page on the phone).
            </Callout>
          )}

          {!latest && <Card className="py-10 text-center text-sm text-muted">Collecting the first {state.window_seconds}-second window…</Card>}

          {latest && (
            <>
              <div className="mb-6 grid gap-5 lg:grid-cols-3">
                <Card className={`flex flex-col justify-center border-2 ${latest.Verdict === "MALICIOUS" ? "border-red-200 bg-red-50" : latest.Verdict === "BENIGN" ? "border-emerald-200 bg-emerald-50" : "border-line"}`}>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted">Latest window · {clock(time(latest.Time))}</div>
                  <div className={`mt-2 text-3xl font-bold ${latest.Verdict === "MALICIOUS" ? "text-red-600" : latest.Verdict === "BENIGN" ? "text-emerald-600" : "text-slate-500"}`}>
                    {latest.Verdict === "MALICIOUS" ? "🚨 " : latest.Verdict === "BENIGN" ? "✅ " : "⏳ "}{latest.Verdict}
                  </div>
                  <div className="mt-1 text-sm text-slate-600">
                    {latest.Flows ? `${latest.Flagged_Flows} of ${latest.Flows} flows classified malicious` : "No IP packets in this window"}
                  </div>
                  <div className="mt-3 text-xs text-muted">
                    Highest malicious probability: <b className="text-ink">{(latest.Max_Malicious_Probability * 100).toFixed(1)}%</b> (model score, not calibrated)
                  </div>
                </Card>
                <div className="grid grid-cols-2 gap-3 lg:col-span-2 sm:grid-cols-4">
                  <Stat label="Packets" value={num(latest.Packet_Count)} />
                  <Stat label="Traffic rate" value={`${latest.Traffic_Rate.toFixed(1)}`} hint="packets / second" />
                  <Stat label="Mean packet size" value={`${latest.Mean_Packet_Size.toFixed(0)} B`} />
                  <Stat label="TTL (most common)" value={latest.TTL === null ? "–" : latest.TTL.toFixed(0)} />
                  <Stat label="Flows" value={latest.Flows} />
                  <Stat label="TCP flows" value={latest.TCP} />
                  <Stat label="UDP flows" value={latest.UDP} />
                  <Stat label="ICMP flows" value={latest.ICMP} />
                </div>
              </div>

              <LiveCharts windows={windows} />

              <Section title="Live traffic history" subtitle="Most recent flow observations, newest first. Packets themselves are never stored.">
                <Table
                  dense
                  maxHeight={380}
                  columns={[
                    { key: "Time", label: "Time", render: (f) => clock(time(f.Time)) },
                    { key: "Source", label: "Source", className: "font-mono text-xs" },
                    { key: "Destination", label: "Destination", className: "font-mono text-xs" },
                    { key: "Protocol", label: "Proto" },
                    { key: "Packet_Count", label: "Packets", align: "right", render: (f) => num(f.Packet_Count) },
                    { key: "Rate", label: "Rate", align: "right", render: (f) => f.Rate.toFixed(2) },
                    { key: "Mean_Packet_Size", label: "Size (B)", align: "right", render: (f) => f.Mean_Packet_Size.toFixed(0) },
                    { key: "TTL", label: "TTL", align: "right", render: (f) => (f.TTL === null ? "–" : f.TTL.toFixed(0)) },
                    {
                      key: "Prediction",
                      label: "Prediction",
                      render: (f) => <Badge tone={f.Prediction === "MALICIOUS" ? "red" : "green"}>{f.Prediction}</Badge>,
                    },
                    {
                      key: "p",
                      label: "Malicious prob.",
                      render: (f) => (
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-20 rounded-full bg-slate-100">
                            <div className="h-full rounded-full" style={{ width: `${f.Malicious_Probability * 100}%`, background: f.Malicious_Probability >= 0.5 ? COLORS.malicious : COLORS.benign }} />
                          </div>
                          <span className="tabular-nums text-xs">{(f.Malicious_Probability * 100).toFixed(0)}%</span>
                        </div>
                      ),
                    },
                  ]}
                  rows={(state.flows ?? []).slice(0, 60)}
                />
              </Section>
            </>
          )}
        </>
      )}

      <div className="mt-8 grid gap-5 lg:grid-cols-2">
        <Card>
          <div className="font-semibold">How the seven features are computed live</div>
          <p className="mt-1 text-sm text-slate-600">
            5G-NIDD records are Argus bidirectional flow records (a status record about every 5 s), so live packets are grouped the same way: one flow
            in one window = one record = one prediction.
          </p>
          <div className="mt-3">
            <Table
              dense
              columns={[
                { key: "f", label: "Feature", className: "font-semibold" },
                { key: "s", label: "5G-NIDD source", className: "font-mono text-xs" },
                { key: "l", label: "Live computation", className: "whitespace-normal text-xs" },
              ]}
              rows={[
                { f: "Rate", s: "Rate", l: "(packets − 1) / (last − first packet time); 0 for one packet" },
                { f: "Packet_Count", s: "TotPkts", l: "packets of the flow in the window (both directions)" },
                { f: "Mean_Packet_Size", s: "TotBytes / TotPkts", l: "IP length + 14-byte Ethernet header, averaged" },
                { f: "TTL", s: "sTtl", l: "TTL / hop limit of the flow originator's packets" },
                { f: "TCP/UDP/ICMP", s: "one-hot of Proto", l: "1 for the flow's protocol, all 0 otherwise" },
              ]}
            />
          </div>
        </Card>
        <Card>
          <div className="font-semibold">📱 Controlled demo with an Android phone</div>
          <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-slate-600">
            <li>On the laptop: Settings → Network &amp; internet → <b>Mobile hotspot → On</b>.</li>
            <li>Connect the phone to the hotspot (it gets an address like <code>192.168.137.x</code>).</li>
            <li>Choose <b>Live capture</b> and the <b>Wi-Fi Direct Virtual Adapter</b> with IP <code>192.168.137.1</code>.</li>
            <li>Optionally enter the phone's IP so only its traffic is analysed.</li>
            <li>Press <b>Start monitoring</b>, then browse or stream on the phone — or open this dashboard on it at <code>http://192.168.137.1:8000</code>.</li>
          </ol>
          <p className="mt-3 text-xs text-muted">Only use devices and networks you own or control. Capture is passive: nothing is sent.</p>
        </Card>
      </div>

      <Callout tone="info" className="mt-6" title="Limitation">
        This model was validated only on 5G-NIDD and degraded sharply on CICIoT2023. Predictions on home Wi-Fi or phone traffic are cross-domain
        too, so read them as a pipeline demonstration, not a reliable security verdict.
      </Callout>
    </>
  );
}
