import { useEffect, useState } from "react";
import { ScanSearch, ShieldAlert, ShieldCheck } from "lucide-react";
import { Button, Card, Callout, EmptyState, PageHeader, Segmented, Select } from "../components/ui";
import { api } from "../lib/api";

const FIELDS = [
  { key: "Rate", label: "Rate", unit: "packets / second", step: "any" },
  { key: "Packet_Count", label: "Packet count", unit: "packets in the flow", step: "1" },
  { key: "Mean_Packet_Size", label: "Mean packet size", unit: "bytes per packet", step: "any" },
  { key: "TTL", label: "TTL", unit: "source time-to-live (0–255)", step: "1" },
];

const protocolOf = (s) => (s.TCP === 1 ? "TCP" : s.UDP === 1 ? "UDP" : s.ICMP === 1 ? "ICMP" : "Other");

export default function PredictionDemo({ data, status }) {
  const [ranges, setRanges] = useState({});
  const [values, setValues] = useState({ Rate: 0, Packet_Count: 2, Mean_Packet_Size: 42, TTL: 63 });
  const [protocol, setProtocol] = useState("UDP");
  const [sample, setSample] = useState("manual");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const samples = data.demo_samples;

  useEffect(() => {
    api.ranges().then((r) => {
      setRanges(r);
      // defaults = 5G-NIDD medians from feature_distribution_shift.csv
      setValues((v) => Object.fromEntries(Object.keys(v).map((k) => [k, r[k]?.median ?? v[k]])));
    }).catch(() => {});
  }, []);

  if (status && !status.models.rf10) {
    return (
      <>
        <PageHeader eyebrow="Application" title="Prediction demo" />
        <EmptyState title="Model file not available" commands={["python src/test_lightweight_models.py"]}>
          models/random_forest_10_trees.joblib is git-ignored because of its size. Create it locally to use the demo.
        </EmptyState>
      </>
    );
  }

  function loadSample(id) {
    setSample(id);
    setResult(null);
    if (id === "manual") return;
    const s = samples[Number(id)];
    setValues({ Rate: s.Rate, Packet_Count: s.Packet_Count, Mean_Packet_Size: s.Mean_Packet_Size, TTL: s.TTL });
    setProtocol(protocolOf(s));
  }

  async function classify() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.predict({ ...values, protocol }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const truth = sample !== "manual" ? samples[Number(sample)].Target : null;

  return (
    <>
      <PageHeader eyebrow="Application" title="Prediction demo — 10-tree Random Forest">
        Classify a single flow with the lightweight 10-tree Random Forest trained on 5G-NIDD. A demonstration of the trained model, not a
        production IDS.
      </PageHeader>

      <div className="grid gap-5 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          {samples ? (
            <Select
              label="Load a real record from the 5G-NIDD held-out test split"
              value={sample}
              onChange={loadSample}
              options={[
                { value: "manual", label: "Manual input" },
                ...samples.map((s, i) => ({ value: String(i), label: `Sample #${i} — true label: ${s.Target === 1 ? "Malicious" : "Benign"}` })),
              ]}
            />
          ) : (
            <p className="text-xs text-muted">
              Defaults are the 5G-NIDD medians. Run <code>python src/extract_demo_samples.py</code> to add real test records here.
            </p>
          )}

          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            {FIELDS.map((f) => {
              const r = ranges[f.key];
              const outside = r && (values[f.key] < r.min || values[f.key] > r.max);
              return (
                <label key={f.key} className="block">
                  <span className="text-sm font-medium">{f.label}</span>
                  <span className="ml-1.5 text-xs text-muted">{f.unit}</span>
                  <input
                    type="number"
                    min={0}
                    step={f.step}
                    value={values[f.key]}
                    onChange={(e) => {
                      setSample("manual");
                      setValues({ ...values, [f.key]: e.target.value === "" ? 0 : Number(e.target.value) });
                    }}
                    className={`mt-1.5 w-full rounded-xl border bg-white px-3 py-2 text-sm tabular-nums shadow-sm outline-none focus:ring-2 ${
                      outside ? "border-amber-400 focus:ring-amber-200" : "border-line focus:border-fiveg focus:ring-fiveg/20"
                    }`}
                  />
                  {r && <span className={`mt-1 block text-[11px] ${outside ? "text-amber-600" : "text-muted"}`}>5G-NIDD range {r.min.toLocaleString()} – {r.max.toLocaleString()}</span>}
                </label>
              );
            })}
          </div>

          <div className="mt-5">
            <div className="mb-1.5 text-sm font-medium">Protocol</div>
            <Segmented
              value={protocol}
              onChange={(p) => { setSample("manual"); setProtocol(p); }}
              options={["TCP", "UDP", "ICMP", "Other"].map((p) => ({ value: p, label: p }))}
            />
            <p className="mt-1.5 text-[11px] text-muted">One-hot, as in training: exactly one of TCP/UDP/ICMP is 1, or all are 0 for other protocols.</p>
          </div>

          <Button variant="blue" icon={ScanSearch} onClick={classify} disabled={busy} className="mt-6 w-full sm:w-auto">
            {busy ? "Classifying…" : "Classify flow"}
          </Button>
        </Card>

        <div className="lg:col-span-2">
          {error && <Callout tone="danger">{error}</Callout>}
          {!result && !error && (
            <Card className="flex h-full flex-col items-center justify-center text-center text-sm text-muted">
              <ScanSearch className="mb-3 h-8 w-8 text-slate-300" />
              Enter flow features and press <b className="mx-1 text-ink">Classify flow</b>.
            </Card>
          )}
          {result && (
            <Card className={`fade-in border-2 ${result.prediction ? "border-red-200" : "border-emerald-200"}`}>
              <div className={`flex items-center gap-3 rounded-xl px-4 py-4 ${result.prediction ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-700"}`}>
                {result.prediction ? <ShieldAlert className="h-8 w-8" /> : <ShieldCheck className="h-8 w-8" />}
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide opacity-80">Prediction</div>
                  <div className="text-2xl font-bold">{result.prediction ? "Malicious" : "Benign"}</div>
                </div>
              </div>
              <div className="mt-5">
                <div className="flex justify-between text-sm">
                  <span className="text-muted">Share of trees voting malicious</span>
                  <span className="font-bold tabular-nums">{(result.probability * 100).toFixed(0)}%</span>
                </div>
                <div className="relative mt-2 h-3 rounded-full bg-gradient-to-r from-emerald-200 via-slate-100 to-red-200">
                  <div className="absolute top-1/2 h-5 w-1.5 -translate-y-1/2 rounded-full bg-ink" style={{ left: `calc(${result.probability * 100}% - 3px)` }} />
                  <div className="absolute left-1/2 top-[-4px] h-5 w-px bg-slate-400" />
                </div>
                <div className="mt-1 flex justify-between text-[11px] text-muted"><span>benign</span><span>threshold 50%</span><span>malicious</span></div>
              </div>
              {truth !== null && (
                <div className="mt-5 rounded-xl bg-slate-50 px-4 py-3 text-sm">
                  True label of this record: <b>{truth === 1 ? "Malicious" : "Benign"}</b>{" "}
                  {truth === result.prediction ? <span className="text-emerald-600">— correct</span> : <span className="text-red-600">— incorrect</span>}
                </div>
              )}
              {result.outside_training_range.length > 0 && (
                <Callout tone="warning" className="mt-4">
                  Outside the range seen in 5G-NIDD: {result.outside_training_range.join(", ")}. The output is an extrapolation.
                </Callout>
              )}
            </Card>
          )}
          <p className="mt-3 text-xs leading-relaxed text-muted">
            This model was validated only on 5G-NIDD. Its predictions on traffic from other networks are unreliable — that is the main finding.
          </p>
        </div>
      </div>
    </>
  );
}
