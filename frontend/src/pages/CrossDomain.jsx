import { useMemo, useState } from "react";
import { Card, ChartCard, Callout, PageHeader, Section, Select, Stat, Table } from "../components/ui";
import { DATASET_SERIES, GroupedBars } from "../components/charts";
import { METRICS, num, pct, pp } from "../lib/format";

function ConfusionMatrix({ cm, title, tone }) {
  const rows = [
    ["Actual benign", cm.TN, cm.FP],
    ["Actual malicious", cm.FN, cm.TP],
  ];
  const base = tone === "fiveg" ? "42,120,214" : "235,104,52";
  return (
    <Card>
      <div className="mb-4 text-sm font-semibold">{title}</div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-sm">
        <div />
        <div className="pb-1 text-center text-xs font-medium text-muted">Predicted benign</div>
        <div className="pb-1 text-center text-xs font-medium text-muted">Predicted malicious</div>
        {rows.map(([label, a, b]) => {
          const total = a + b;
          return [
            <div key={`${label}l`} className="flex items-center pr-2 text-xs font-medium text-muted">{label}</div>,
            ...[a, b].map((v, i) => {
              const share = total ? v / total : 0;
              return (
                <div
                  key={`${label}${i}`}
                  className="flex flex-col items-center justify-center rounded-xl py-4"
                  style={{ background: `rgba(${base},${0.08 + share * 0.85})`, color: share > 0.55 ? "white" : "#0f172a" }}
                >
                  <span className="text-lg font-bold tabular-nums">{num(v)}</span>
                  <span className="text-[11px] opacity-80">{(share * 100).toFixed(1)}% of row</span>
                </div>
              );
            }),
          ];
        })}
      </div>
    </Card>
  );
}

export default function CrossDomain({ data }) {
  const options = useMemo(() => {
    const list = (data.algorithms ?? []).map((a) => ({ name: a.model, in: a.in_domain, cross: a.cross_domain }));
    const rf100 = data.rf_trees?.find((r) => r.trees === 100);
    if (rf100?.cross_domain) list.push({ name: "Random Forest (100 trees) — baseline", in: rf100.in_domain, cross: rf100.cross_domain });
    return list;
  }, [data]);
  const [selected, setSelected] = useState(options[0]?.name);
  const model = options.find((o) => o.name === selected);
  const c5 = data.confusion.fiveg;
  const cc = data.confusion.cic;

  return (
    <>
      <PageHeader eyebrow="Core experiment" title="Cross-domain evaluation: 5G-NIDD → CICIoT2023">
        Each model was trained only on 5G-NIDD, then applied to every processed CICIoT2023 record with no retraining,
        fine-tuning, threshold change or re-scaling. CICIoT2023 comes from a different, non-5G IoT network.
      </PageHeader>

      {model && (
        <Section
          title="In-domain vs cross-domain"
          action={
            <div className="w-72">
              <Select value={selected} onChange={setSelected} options={options.map((o) => ({ value: o.name, label: o.name }))} />
            </div>
          }
        >
          <div className="grid gap-5 lg:grid-cols-5">
            <div className="grid grid-cols-2 gap-3 lg:col-span-2">
              {METRICS.map((m) => (
                <Stat
                  key={m.key}
                  label={`${m.label} · CICIoT2023`}
                  value={pct(model.cross[m.key])}
                  tone="cic"
                  delta={`${pp((model.cross[m.key] - model.in[m.key]) * 100)} vs 5G-NIDD (${pct(model.in[m.key], 1)})`}
                  deltaTone={model.cross[m.key] < model.in[m.key] ? "bad" : "neutral"}
                />
              ))}
            </div>
            <ChartCard className="lg:col-span-3" title={model.name}>
              <GroupedBars
                height={260}
                series={DATASET_SERIES}
                labelDigits={2}
                data={METRICS.map((m) => ({ name: m.label, fiveg: model.in[m.key] * 100, cic: model.cross[m.key] * 100 }))}
              />
            </ChartCard>
          </div>
        </Section>
      )}

      {c5 && cc && (
        <>
          <Section title="Reading the metrics correctly: class balance" subtitle="The two test sets contain very different shares of malicious traffic, which changes what a “good” score looks like.">
            <div className="mb-5 grid gap-4 sm:grid-cols-3">
              <Stat label="Malicious share · 5G-NIDD test" value={pct(c5.malicious_share, 1)} tone="fiveg" />
              <Stat label="Malicious share · CICIoT2023" value={pct(cc.malicious_share, 1)} tone="cic" />
              <Stat label="Benign records · CICIoT2023" value={num(cc.benign)} hint={`out of ${num(cc.total)} records`} />
            </div>
            <Table
              columns={[
                { key: "name", label: "Reference / model", className: "font-semibold" },
                { key: "a5", label: "5G-NIDD accuracy", align: "right" },
                { key: "f5", label: "5G-NIDD F1", align: "right" },
                { key: "ac", label: "CICIoT2023 accuracy", align: "right" },
                { key: "pc", label: "CICIoT2023 precision", align: "right" },
                { key: "fc", label: "CICIoT2023 F1", align: "right" },
              ]}
              rows={[
                { name: "Trivial: label everything malicious", a5: pct(c5.trivial_accuracy), f5: pct(c5.trivial_f1), ac: pct(cc.trivial_accuracy), pc: pct(cc.trivial_precision), fc: pct(cc.trivial_f1) },
                { name: "Random Forest (100 trees) baseline", a5: pct(c5.accuracy), f5: pct(data.baseline.f1_score), ac: pct(cc.accuracy), pc: pct(data.cross.precision), fc: pct(data.cross.f1_score) },
              ]}
            />
            <div className="mt-5 grid gap-3 md:grid-cols-2">
              <Callout tone="warning" title="Accuracy is dominated by benign records">
                The model predicts nearly everything as benign, so the records it gets right on CICIoT2023 are mostly the {num(cc.benign)} benign ones.
              </Callout>
              <Callout tone="warning" title="Precision is not a strength here">
                With {pct(cc.malicious_share, 1)} of records malicious, a random pick is already that precise. Cross-domain precision is below this base rate.
              </Callout>
              <Callout tone="info" title="Use recall, balanced accuracy and MCC">
                Recall shows that almost all attacks are missed; balanced accuracy and MCC are not inflated by class imbalance.
              </Callout>
              <Callout tone="info" title="In-domain is moderate, not strong">
                On 5G-NIDD a trivial “always malicious” rule already scores {pct(c5.trivial_f1, 1)} F1, so the margin over chance is limited.
              </Callout>
            </div>
          </Section>

          <Section title="Confusion matrices — 100-tree Random Forest baseline" subtitle="Saved only for the baseline (baseline_5g_results.json and cross_domain_results.json). Colour = share of the actual class.">
            <div className="grid gap-5 lg:grid-cols-2">
              <ConfusionMatrix cm={c5} title="5G-NIDD held-out test (in-domain)" tone="fiveg" />
              <ConfusionMatrix cm={cc} title="CICIoT2023 (cross-domain)" tone="cic" />
            </div>
            <div className="mt-5 grid gap-5 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <Table
                  columns={[
                    { key: "m", label: "Derived from confusion matrix", className: "font-medium" },
                    { key: "a", label: "5G-NIDD", align: "right" },
                    { key: "b", label: "CICIoT2023", align: "right" },
                  ]}
                  rows={[
                    { m: "Share of records predicted malicious", a: pct(c5.predicted_malicious_share), b: pct(cc.predicted_malicious_share) },
                    { m: "Benign recall (specificity)", a: pct(c5.specificity), b: pct(cc.specificity) },
                    { m: "Malicious recall (detection rate)", a: pct(c5.recall), b: pct(cc.recall) },
                    { m: "Balanced accuracy", a: pct(c5.balanced_accuracy), b: pct(cc.balanced_accuracy) },
                  ]}
                />
              </div>
              <Card className="flex flex-col justify-center bg-gradient-to-br from-fiveg-soft to-cic-soft">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-600">The model's behaviour flips</div>
                <div className="mt-3 flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-fiveg">{pct(c5.predicted_malicious_share, 1)}</span>
                  <span className="text-sm text-slate-600">of 5G-NIDD flows flagged</span>
                </div>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-cic">{pct(cc.predicted_malicious_share, 1)}</span>
                  <span className="text-sm text-slate-600">of CICIoT2023 flows flagged</span>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-slate-600">
                  CICIoT2023 traffic lands in regions of the feature space that the 5G-trained model associates with benign traffic.
                </p>
              </Card>
            </div>
          </Section>
        </>
      )}

      <Callout tone="danger" title="Conclusion">
        Models that work on 5G-NIDD suffer a severe loss of detection performance when transferred to CICIoT2023 without retraining.
        This is the finding of the project, not a failure of the experiment.
      </Callout>
    </>
  );
}
