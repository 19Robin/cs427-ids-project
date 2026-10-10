import { useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, ChartCard, Callout, EmptyState, PageHeader, Section, Select, Stat, Table } from "../components/ui";
import { ChartTooltip, DATASET_SERIES, GroupedBars, HBars } from "../components/charts";
import { COLORS, axisModel, kb, num, pct, shortModel } from "../lib/format";

const IN_TEST = "5G-NIDD test (in-domain)";
const CROSS_TEST = "CICIoT2023 (cross-domain)";
const EDGE_MODEL = "Random Forest (10 trees, depth ≤ 12)";

const EXPERIMENTS = [
  { id: "Main", short: "Main", changes: "Nothing extra", tests: "The cross-domain result after the data fixes" },
  { id: "A1", short: "A1", changes: "CICIoT2023 protocol flags rounded to 0/1", tests: "Are the protocol features defined differently?" },
  { id: "A2", short: "A2", changes: "A1 + Packet_Count removed from both", tests: "The feature with the largest definition mismatch" },
  { id: "B", short: "B", changes: "CICIoT2023 limited to attacks comparable to 5G-NIDD", tests: "Do unseen attack types explain the drop?" },
  { id: "C", short: "C", changes: "Each dataset standardised with its own statistics", tests: "A simple fix for different value ranges (unlabelled target data)" },
  { id: "D", short: "D", changes: "1% of labelled CICIoT2023 added to training", tests: "How cheaply an operator could adapt the model — this is retraining" },
];

const expId = (name) => (name === "Main" ? "Main" : name.split(":")[0]);

function heat(value) {
  // 50% = chance (white) → red below, green above
  if (value === null || value === undefined) return {};
  const d = Math.max(-1, Math.min(1, (value - 0.5) / 0.5));
  const color = d >= 0 ? `rgba(16,185,129,${0.12 + d * 0.6})` : `rgba(239,68,68,${0.12 + -d * 0.6})`;
  return { background: color };
}

export default function Improved({ data, status }) {
  const imp = data.improved;
  const models = imp?.model_info?.map((m) => m.Model) ?? [];
  const [attackModel, setAttackModel] = useState(models.includes(EDGE_MODEL) ? EDGE_MODEL : models[0]);
  const [diagModel, setDiagModel] = useState(models.includes(EDGE_MODEL) ? EDGE_MODEL : models[0]);

  if (!imp) {
    return (
      <>
        <PageHeader eyebrow="Second round" title="Improved experiments: fixes, diagnosis and mitigation" />
        <EmptyState
          title="No improved results yet"
          commands={["python src/preprocess_data.py", "python src/run_improved_experiments.py", "git add results/improved"]}
        >
          The datasets are not in the repository, so run the experiment script locally. It writes results/improved/ and this page fills in
          automatically.
        </EmptyState>
      </>
    );
  }

  const q = imp.quality ?? {};
  const results = imp.results;
  const main = results.filter((r) => r.Experiment === "Main");
  const find = (rows, model, test) => rows.find((r) => r.Model === model && r.Test_Set === test);
  const mainChart = (key) =>
    models.map((m) => ({
      name: m,
      fiveg: (find(main, m, IN_TEST)?.[key] ?? null) * 100,
      cic: (find(main, m, CROSS_TEST)?.[key] ?? null) * 100,
    }));
  const cross = results.filter((r) => r.Test_Set !== IN_TEST);
  const experiments = [...new Set(cross.map((r) => r.Experiment))];
  const diagRows = experiments.map((e) => cross.find((r) => r.Experiment === e && r.Model === diagModel)).filter(Boolean);
  const noRetrain = diagRows.filter((r) => !expId(r.Experiment).startsWith("D"));
  const best = noRetrain.reduce((a, b) => (b.Balanced_Accuracy > a.Balanced_Accuracy ? b : a), noRetrain[0]);
  const mainRow = diagRows.find((r) => r.Experiment === "Main");
  const dRow = diagRows.find((r) => expId(r.Experiment) === "D");
  const info = imp.model_info;
  const perAttack = imp.per_attack?.filter((r) => r.Model === attackModel) ?? [];

  return (
    <>
      <PageHeader eyebrow="Second round" title="Improved experiments: fixes, diagnosis and mitigation">
        Applies the data preparation promised in the proposal (duplicate removal and class balancing), then tests <i>why</i> the
        cross-domain performance drops. Every model is trained on 5G-NIDD only, except experiment D, which is labelled as retraining.
      </PageHeader>

      {/* 1. Data quality */}
      {q.original_split_test_overlap_share !== undefined && (
        <Section title="1 · Data quality" subtitle="Duplicates are removed on the 7 features plus the label.">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat
              label="Original test rows also seen in training"
              value={pct(q.original_split_test_overlap_share, 1)}
              tone="bad"
              hint="Exact 7-feature vector appears in the training split: inflates in-domain scores"
            />
            <Stat
              label="5G-NIDD rows with conflicting labels"
              value={pct(q["5g_nidd_conflicting_label_share"], 1)}
              tone="bad"
              hint="Same features as a row with the opposite label: no model can get all of these right"
            />
            <Stat
              label="5G-NIDD duplicates removed"
              value={pct(1 - q["5g_nidd_rows_after_dedup"] / q["5g_nidd_rows_before_dedup"], 1)}
              hint={`${num(q["5g_nidd_rows_before_dedup"])} → ${num(q["5g_nidd_rows_after_dedup"])} rows`}
            />
            <Stat
              label="CICIoT2023 duplicates removed"
              value={pct(1 - q.ciciot2023_rows_after_dedup / q.ciciot2023_rows_before_dedup, 1)}
              hint={`${num(q.ciciot2023_rows_before_dedup)} → ${num(q.ciciot2023_rows_after_dedup)} rows`}
            />
          </div>
          <Callout tone="info" className="mt-4" title="What this means">
            With only seven features, many 5G-NIDD records are identical, and many identical records carry opposite labels. That explains why the
            original in-domain accuracy was capped around 74%. After deduplication, scores describe performance on <i>distinct</i> traffic
            patterns rather than on the most frequent ambiguous ones, so report both views.
          </Callout>
        </Section>
      )}

      {/* 2. Main */}
      <Section title="2 · Main experiment: deduplicated data + class balancing" subtitle="Balanced accuracy and MCC are not inflated by class imbalance: 50% balanced accuracy and MCC 0 mean no better than chance.">
        <div className="grid gap-5 lg:grid-cols-2">
          <ChartCard title="F1 score">
            <GroupedBars data={mainChart("F1")} series={DATASET_SERIES} height={300} xTickFormatter={axisModel} />
          </ChartCard>
          <ChartCard title="Balanced accuracy" footer="Dashed line = chance (50%).">
            <GroupedBars data={mainChart("Balanced_Accuracy")} series={DATASET_SERIES} height={300} xTickFormatter={axisModel} reference={{ value: 50 }} />
          </ChartCard>
        </div>
        <div className="mt-5">
          <Table
            dense
            columns={[
              { key: "m", label: "Model", className: "font-semibold" },
              ...["F1", "Recall", "Specificity", "Balanced_Accuracy"].map((k) => ({
                key: `i${k}`,
                label: `5G ${k.replace("_", " ")}`,
                align: "right",
                render: ({ m }) => pct(find(main, m, IN_TEST)?.[k], 1),
              })),
              { key: "imcc", label: "5G MCC", align: "right", render: ({ m }) => find(main, m, IN_TEST)?.MCC?.toFixed(3) },
              ...["F1", "Recall", "Specificity", "Balanced_Accuracy"].map((k) => ({
                key: `c${k}`,
                label: `CIC ${k.replace("_", " ")}`,
                align: "right",
                render: ({ m }) => <span className="text-cic">{pct(find(main, m, CROSS_TEST)?.[k], 1)}</span>,
              })),
              { key: "cmcc", label: "CIC MCC", align: "right", render: ({ m }) => <b className="text-cic">{find(main, m, CROSS_TEST)?.MCC?.toFixed(3)}</b> },
            ]}
            rows={models.map((m) => ({ m }))}
          />
        </div>
      </Section>

      {/* 3. Lightweight */}
      <Section title="3 · Lightweight enough for a 5G edge node?" subtitle={`Compared with the ${data.paper.dl_iid_size_kb} KB quantised DL-IID model of ${data.paper.citation}, the reference paper for this project.`}>
        <div className="grid gap-5 lg:grid-cols-2">
          <ChartCard title="Saved model size" subtitle="KB, log scale" footer="Dashed line: DL-IID (Khan et al.)">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={info.map((m) => ({ name: m.Model, value: m.Model_Size_KB }))} layout="vertical" margin={{ top: 4, right: 72, left: 4, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke="#eef2f6" />
                <XAxis type="number" scale="log" domain={[1, "auto"]} allowDataOverflow tickFormatter={(v) => (v >= 1024 ? `${Math.round(v / 1024)} MB` : `${v} KB`)} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={180} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <Tooltip cursor={{ fill: "rgba(148,163,184,0.12)" }} content={<ChartTooltip format={(v) => kb(v)} />} />
                <ReferenceLine x={data.paper.dl_iid_size_kb} stroke={COLORS.fiveg} strokeDasharray="5 4" strokeWidth={2} />
                <Bar dataKey="value" name="Size" fill="#64748b" radius={[0, 4, 4, 0]} maxBarSize={22}>
                  <LabelList dataKey="value" position="right" formatter={kb} style={{ fontSize: 11, fill: "#334155" }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
          <ChartCard title="Time to classify one flow" subtitle="Median, single thread, on the machine that ran the experiments">
            <HBars
              data={info.map((m) => ({ name: m.Model, value: m.Latency_ms_per_flow }))}
              color="#64748b"
              unit=" ms"
              tickWidth={180}
              labelFormat={(v) => `${v.toFixed(2)} ms`}
            />
          </ChartCard>
        </div>
      </Section>

      {/* 4. Per attack */}
      {imp.per_attack && (
        <Section
          title="4 · Detection per attack type, in-domain and cross-domain"
          subtitle="For attack types this is the detection rate. For the benign row it is the false-alarm rate."
          action={<div className="w-72"><Select value={attackModel} onChange={setAttackModel} options={models.map((m) => ({ value: m, label: m }))} /></div>}
        >
          <div className="grid gap-5 lg:grid-cols-2">
            {[
              { test: IN_TEST, color: COLORS.fiveg },
              { test: CROSS_TEST, color: COLORS.cic },
            ].map(({ test, color }) => {
              const rows = perAttack.filter((r) => r.Test_Set === test).sort((a, b) => b.Predicted_Malicious_Rate - a.Predicted_Malicious_Rate);
              return (
                <ChartCard key={test} title={test}>
                  {rows.length ? (
                    <HBars data={rows.map((r) => ({ name: r.Attack_Type, value: r.Predicted_Malicious_Rate * 100 }))} color={color} domain={[0, 100]} tickWidth={170} labelFormat={(v) => `${v.toFixed(1)}%`} />
                  ) : (
                    <p className="text-sm text-muted">No per-attack results for this test set.</p>
                  )}
                </ChartCard>
              );
            })}
          </div>
        </Section>
      )}

      {/* 5. Diagnosis */}
      <Section
        title="5 · Why does performance drop? Diagnosis experiments"
        subtitle="All bars are measured on CICIoT2023 (B: comparable-attack subset; D: the 99% not used for training)."
        action={<div className="w-72"><Select value={diagModel} onChange={setDiagModel} options={models.map((m) => ({ value: m, label: m }))} /></div>}
      >
        <div className="mb-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {EXPERIMENTS.map((e) => (
            <div key={e.id} className={`rounded-xl border px-4 py-3 ${e.id === "D" ? "border-amber-200 bg-amber-50" : "border-line bg-white"}`}>
              <div className="flex items-center gap-2">
                <span className={`flex h-6 min-w-6 items-center justify-center rounded-md px-1.5 text-xs font-bold ${e.id === "D" ? "bg-amber-500 text-white" : "bg-ink text-white"}`}>{e.short}</span>
                <span className="text-sm font-semibold">{e.changes}</span>
              </div>
              <div className="mt-1 text-xs text-muted">{e.tests}</div>
            </div>
          ))}
        </div>

        <div className="grid gap-5 lg:grid-cols-5">
          <ChartCard className="lg:col-span-3" title={`Balanced accuracy on CICIoT2023 — ${diagModel}`} footer="Dashed line = chance (50%). Experiment D (amber) involves retraining and is not a no-retraining result.">
            <HBars
              data={diagRows.map((r) => ({
                name: r.Experiment,
                value: r.Balanced_Accuracy * 100,
                color: expId(r.Experiment) === "D" ? "#eda100" : COLORS.cic,
              }))}
              domain={[0, 100]}
              tickWidth={230}
              reference={{ value: 50 }}
              labelFormat={(v) => `${v.toFixed(1)}%`}
            />
          </ChartCard>
          <Card className="lg:col-span-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted">Result for this model</div>
            {best && mainRow && (
              <p className="mt-3 text-sm leading-relaxed text-slate-700">
                Without retraining, the best balanced accuracy comes from <b>{best.Experiment}</b>:{" "}
                <b className="text-cic">{pct(best.Balanced_Accuracy, 1)}</b>, vs {pct(mainRow.Balanced_Accuracy, 1)} in the main experiment
                (recall {pct(best.Recall, 1)}, F1 {pct(best.F1, 1)}).
              </p>
            )}
            {dRow && (
              <p className="mt-3 text-sm leading-relaxed text-slate-700">
                With 1% labelled CICIoT2023 data (retraining), balanced accuracy reaches{" "}
                <b className="text-amber-600">{pct(dRow.Balanced_Accuracy, 1)}</b>.
              </p>
            )}
            <div className="mt-4 space-y-2 border-t border-line pt-4 text-xs leading-relaxed text-slate-600">
              <p><b>A1/A2 recover a lot</b> → part of the drop comes from the datasets <i>defining</i> features differently.</p>
              <p><b>B stays low</b> → the model misses even attacks it was trained on; unseen attack types are not the whole story.</p>
              <p><b>C helps</b> → part of the drop is a value-range difference that unlabelled data can correct.</p>
              <p><b>D</b> → how much a small labelled sample from the new network helps an operator.</p>
            </div>
          </Card>
        </div>

        <div className="mt-5">
          <div className="mb-2 text-sm font-semibold">All models × all experiments — balanced accuracy on CICIoT2023</div>
          <div className="overflow-x-auto rounded-xl border border-line bg-white">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-muted">
                <tr>
                  <th className="px-4 py-2.5 text-left font-semibold">Experiment</th>
                  {models.map((m) => (
                    <th key={m} className="px-3 py-2.5 text-right font-semibold">{shortModel(m)}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {experiments.map((e) => (
                  <tr key={e}>
                    <td className="whitespace-nowrap px-4 py-2 font-medium">{e}</td>
                    {models.map((m) => {
                      const r = cross.find((x) => x.Experiment === e && x.Model === m);
                      return (
                        <td key={m} className="px-3 py-2 text-right font-semibold tabular-nums" style={heat(r?.Balanced_Accuracy)}>
                          {pct(r?.Balanced_Accuracy, 1)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">Red = worse than chance, white ≈ chance (50%), green = better than chance.</p>
        </div>
      </Section>
    </>
  );
}
