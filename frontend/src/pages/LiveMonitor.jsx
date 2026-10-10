import { useCallback, useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend as RLegend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FlaskConical, Pause, Play, Square, Trash2 } from "lucide-react";
import { Badge, Button, Card, Callout, EmptyState, PageHeader, Section, Segmented, Select, Stat, Table } from "../components/ui";
import { api } from "../lib/api";
import { COLORS, num } from "../lib/format";

const time = (iso) => new Date(iso).getTime();
const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function TimeTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-xl border border-line bg-white/95 px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-semibold">{clock(p.t)}</div>
      {p.source && <div className="text-slate-600">{p.source} → {p.destination} ({p.protocol})</div>}
      {p.prediction && <div>Prediction: <b>{p.prediction}</b></div>}
      {p.p !== undefined && <div>Malicious probability: <b>{(p.p * 100).toFixed(0)}%</b></div>}
      {p.rate !== undefined && <div>Traffic rate: <b>{p.rate.toFixed(1)} pkt/s</b></div>}
      {p.verdict && <div>Window verdict: <b>{p.verdict}</b></div>}
    </div>
  );
}

const verdictColor = (v) => (v === "MALICIOUS" ? COLORS.malicious : v === "BENIGN" ? COLORS.benign : COLORS.neutral);

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
  const poll = useRef(null);

  const refresh = useCallback(async () => {
    try {
      setState(await api.liveState());
    } catch (e) {
      setError(e.message);
    }
  }, []);

  // pick up a monitor that is already running, and poll while it runs
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
    refresh();
  }

  const windows = state?.windows ?? [];
  const latest = windows[windows.length - 1];
  const flowPoints = (state?.flows ?? []).map((f) => ({
    t: time(f.Time), p: f.Malicious_Probability, prediction: f.Prediction,
    source: f.Source, destination: f.Destination, protocol: f.Protocol,
  }));
  const windowPoints = windows.map((w) => ({ t: time(w.Time), p: w.Max_Malicious_Probability, rate: w.Traffic_Rate, verdict: w.Verdict }));
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

              <div className="mb-6 grid gap-5 lg:grid-cols-2">
                <Card>
                  <div className="text-sm font-semibold">Malicious probability over time</div>
                  <div className="text-xs text-muted">One point per flow; line = highest flow probability per window; dashed = 0.5 threshold</div>
                  <ResponsiveContainer width="100%" height={260}>
                    <ComposedChart margin={{ top: 14, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="#eef2f6" />
                      <XAxis type="number" dataKey="t" domain={["dataMin", "dataMax"]} tickFormatter={clock} tickLine={false} axisLine={{ stroke: "#e2e8f0" }} />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} tickLine={false} axisLine={false} width={48} />
                      <Tooltip content={<TimeTooltip />} />
                      <ReferenceLine y={0.5} stroke="#475569" strokeDasharray="5 4" />
                      <Line data={windowPoints} dataKey="p" name="Window peak" stroke="#94a3b8" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Scatter data={flowPoints.filter((f) => f.prediction === "BENIGN")} dataKey="p" name="Benign flow" fill={COLORS.benign} isAnimationActive={false} />
                      <Scatter data={flowPoints.filter((f) => f.prediction === "MALICIOUS")} dataKey="p" name="Malicious flow" fill={COLORS.malicious} shape="triangle" isAnimationActive={false} />
                      <RLegend iconSize={9} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </Card>
                <Card>
                  <div className="text-sm font-semibold">Traffic rate over time</div>
                  <div className="text-xs text-muted">Packets per second per window, marked by the window verdict</div>
                  <ResponsiveContainer width="100%" height={260}>
                    <ComposedChart data={windowPoints} margin={{ top: 14, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="#eef2f6" />
                      <XAxis type="number" dataKey="t" domain={["dataMin", "dataMax"]} tickFormatter={clock} tickLine={false} axisLine={{ stroke: "#e2e8f0" }} />
                      <YAxis tickLine={false} axisLine={false} width={48} />
                      <Tooltip content={<TimeTooltip />} />
                      <Line
                        dataKey="rate"
                        name="Traffic rate"
                        stroke="#94a3b8"
                        strokeWidth={2}
                        isAnimationActive={false}
                        dot={(props) => <circle key={props.index} cx={props.cx} cy={props.cy} r={4.5} fill={verdictColor(props.payload.verdict)} stroke="white" strokeWidth={1.5} />}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                  <div className="mt-1 flex gap-4 text-xs text-slate-600">
                    <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: COLORS.benign }} />Benign window</span>
                    <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: COLORS.malicious }} />Malicious window</span>
                  </div>
                </Card>
              </div>

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
