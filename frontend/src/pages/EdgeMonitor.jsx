import { useEffect, useRef, useState } from "react";
import { Play, RotateCcw, Square } from "lucide-react";
import { Button, Card, Callout, EmptyState, Legend, PageHeader, Segmented, Stat, Table } from "../components/ui";
import { api } from "../lib/api";

const OUTCOME = {
  TP: { color: "#dc2626", label: "Attack caught" },
  FN: { color: "#fdba74", label: "Attack missed" },
  FP: { color: "#a855f7", label: "False alarm" },
  TN: { color: "#bbf7d0", label: "Benign, no alert" },
};

const outcome = (f) => (f.prediction ? (f.target ? "TP" : "FP") : f.target ? "FN" : "TN");
const protocol = (f) => (f.TCP >= 0.5 ? "TCP" : f.UDP >= 0.5 ? "UDP" : f.ICMP >= 0.5 ? "ICMP" : "Other");
const SPEEDS = { fast: 15, normal: 60, slow: 180 };

export default function EdgeMonitor({ status }) {
  const [dataset, setDataset] = useState("5G-NIDD");
  const [count, setCount] = useState(150);
  const [speed, setSpeed] = useState("normal");
  const [feed, setFeed] = useState(null);
  const [shown, setShown] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const timer = useRef(null);

  useEffect(() => () => clearInterval(timer.current), []);

  if (status && !status.stream_samples) {
    return (
      <>
        <PageHeader eyebrow="Application" title="5G Edge IDS Monitor" />
        <EmptyState title="No recorded flows yet" commands={["python src/run_improved_experiments.py"]}>
          The monitor replays real flows saved in results/improved/stream_samples.csv, classified by the edge model in models/.
        </EmptyState>
      </>
    );
  }

  async function start() {
    clearInterval(timer.current);
    setError(null);
    setShown(0);
    try {
      const result = await api.stream(dataset, count);
      setFeed(result);
      setRunning(true);
      let i = 0;
      timer.current = setInterval(() => {
        i += 1;
        setShown(i);
        if (i >= result.flows.length) {
          clearInterval(timer.current);
          setRunning(false);
        }
      }, SPEEDS[speed]);
    } catch (e) {
      setError(e.message);
    }
  }

  function stop() {
    clearInterval(timer.current);
    setRunning(false);
  }

  const flows = feed ? feed.flows.slice(0, shown) : [];
  const counts = { TP: 0, FN: 0, FP: 0, TN: 0 };
  flows.forEach((f) => (counts[outcome(f)] += 1));
  const attacks = counts.TP + counts.FN;
  const latencies = flows.map((f) => f.latency_ms).sort((a, b) => a - b);
  const median = latencies.length ? latencies[Math.floor(latencies.length / 2)] : null;
  const alerts = flows
    .map((f, i) => ({ ...f, index: i + 1 }))
    .filter((f) => f.prediction === 1)
    .reverse()
    .slice(0, 10);
  const done = feed && shown >= feed.flows.length;

  return (
    <>
      <PageHeader eyebrow="Application" title="5G Edge IDS Monitor">
        How an operator would run the lightweight IDS at a 5G edge node: recorded real flows are replayed one at a time through the trained
        model, and every flow it labels malicious raises an alert. Switch the feed to CICIoT2023 to watch the same model in a different
        network.
      </PageHeader>

      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-5">
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">Traffic feed</div>
            <Segmented
              value={dataset}
              onChange={setDataset}
              disabled={running}
              options={[
                { value: "5G-NIDD", label: "5G-NIDD · 5G network" },
                { value: "CICIoT2023", label: "CICIoT2023 · different network" },
              ]}
            />
          </div>
          <label className="block">
            <div className="mb-1.5 text-xs font-medium text-muted">Flows: {count}</div>
            <input type="range" min={20} max={300} step={10} value={count} disabled={running} onChange={(e) => setCount(Number(e.target.value))} className="w-44 accent-[#2a78d6]" />
          </label>
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">Speed</div>
            <Segmented value={speed} onChange={setSpeed} disabled={running} options={[{ value: "slow", label: "Slow" }, { value: "normal", label: "Normal" }, { value: "fast", label: "Fast" }]} />
          </div>
          <div className="ml-auto flex gap-2">
            {running ? (
              <Button variant="ghost" icon={Square} onClick={stop}>Stop</Button>
            ) : (
              <Button variant="blue" icon={feed ? RotateCcw : Play} onClick={start}>{feed ? "Replay" : "Start replay"}</Button>
            )}
          </div>
        </div>
        {feed && <p className="mt-4 text-xs text-muted">Model: {feed.model}. Flows: real records from the {feed.dataset === "5G-NIDD" ? "5G-NIDD test split (never used for training)" : "CICIoT2023 dataset"}.</p>}
      </Card>

      {error && <Callout tone="danger" className="mb-6">{error}</Callout>}

      {feed && (
        <>
          <div className="mb-6 grid gap-4 sm:grid-cols-3 xl:grid-cols-5">
            <Stat label="Flows processed" value={`${shown} / ${feed.flows.length}`} />
            <Stat label="Alerts raised" value={counts.TP + counts.FP} />
            <Stat label="Attacks caught" value={`${counts.TP} / ${attacks}`} tone={attacks && counts.TP / attacks >= 0.5 ? "good" : "bad"} hint="Ground truth is shown for evaluation only" />
            <Stat label="False alarms" value={counts.FP} hint={`out of ${counts.FP + counts.TN} benign flows`} />
            <Stat label="Median latency" value={median === null ? "–" : `${median.toFixed(2)} ms`} hint="per flow, measured on the server" />
          </div>

          <div className="grid gap-5 lg:grid-cols-5">
            <Card className="lg:col-span-3">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="text-sm font-semibold">Flow stream</div>
                <Legend items={Object.values(OUTCOME)} />
              </div>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(14px,1fr))] gap-1">
                {feed.flows.map((f, i) => {
                  const o = outcome(f);
                  const visible = i < shown;
                  return (
                    <div
                      key={i}
                      title={visible ? `Flow ${i + 1}: ${OUTCOME[o].label} (${f.attack_type})` : `Flow ${i + 1}`}
                      className="aspect-square rounded-[3px] transition-colors duration-200"
                      style={{ background: visible ? OUTCOME[o].color : "#f1f5f9" }}
                    />
                  );
                })}
              </div>
              {done && (
                <Callout tone={attacks && counts.TP / attacks >= 0.5 ? "success" : "danger"} className="mt-4">
                  Caught {counts.TP} of {attacks} attacks ({attacks ? ((counts.TP / attacks) * 100).toFixed(1) : 0}%) with {counts.FP} false
                  alarm{counts.FP === 1 ? "" : "s"} out of {counts.FP + counts.TN} benign flows.
                </Callout>
              )}
            </Card>

            <Card className="lg:col-span-2" padded={false}>
              <div className="flex items-center justify-between px-5 pt-5">
                <div className="text-sm font-semibold">Alert log</div>
                {running && <span className="pulse-dot h-2 w-2 rounded-full bg-emerald-500" />}
              </div>
              <div className="p-3">
                {alerts.length ? (
                  <Table
                    dense
                    columns={[
                      { key: "index", label: "#", align: "right" },
                      { key: "p", label: "Proto", render: protocol },
                                            { key: "attack_type", label: "Truth", render: (f) => <span className="text-xs">{f.attack_type}</span> },
                      { key: "ok", label: "Result", render: (f) => (f.target ? <span className="text-xs font-semibold text-emerald-600">caught</span> : <span className="text-xs font-semibold text-purple-600">false alarm</span>) },
                    ]}
                    rows={alerts}
                  />
                ) : (
                  <p className="px-2 py-6 text-center text-sm text-muted">No alerts yet.</p>
                )}
              </div>
            </Card>
          </div>
        </>
      )}

      <p className="mt-6 text-xs text-muted">A replay of recorded flows for demonstration, not a live network capture, and not a production IDS.</p>
    </>
  );
}
