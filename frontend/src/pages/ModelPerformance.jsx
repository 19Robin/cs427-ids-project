import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartCard, Callout, EmptyState, PageHeader, Section, Table } from "../components/ui";
import { ChartTooltip, HBars } from "../components/charts";
import { COLORS, METRICS, num, pct } from "../lib/format";

export default function ModelPerformance({ data }) {
  const rows = data.rf_trees;
  if (!rows) {
    return (
      <>
        <PageHeader title="Random Forest: effect of model size" />
        <EmptyState title="No Random Forest results" commands={["python src/test_lightweight_models.py", "python src/test_lightweight_cross_domain.py"]}>
          results/lightweight_model_comparison.csv was not found.
        </EmptyState>
      </>
    );
  }

  const metricData = rows.map((r) => ({
    name: `${r.trees}`,
    ...Object.fromEntries(METRICS.map((m) => [m.key, r.in_domain[m.key] * 100])),
  }));
  const f1Data = rows.map((r) => ({
    name: `${r.trees}`,
    fiveg: r.in_domain.f1 * 100,
    cic: r.cross_domain ? r.cross_domain.f1 * 100 : null,
  }));
  const ten = rows.find((r) => r.trees === 10);
  const hundred = rows.find((r) => r.trees === 100);

  return (
    <>
      <PageHeader eyebrow="Lightweight models" title="Random Forest: effect of model size">
        Random Forests with 10, 25, 50 and 100 trees, trained on the same 80/20 stratified 5G-NIDD split with the same seven
        features. The 100-tree model is the original baseline; the 10-tree model is the lightweight variant.
      </PageHeader>

      {ten && hundred && (
        <Callout tone="success" className="mb-8" title="10 trees are enough">
          The 10-tree forest matches the 100-tree baseline (F1 {pct(ten.in_domain.f1)} vs {pct(hundred.in_domain.f1)}) while being
          about {Math.round(hundred.size_mb / ten.size_mb)}× smaller and {Math.round(hundred.train_seconds / ten.train_seconds)}×
          faster to train. “Lightweight” is still relative: at {ten.size_mb.toFixed(1)} MB it is far larger than the 108 KB
          DL-IID model of Khan et al., because its trees have no depth limit.
        </Callout>
      )}

      <div className="mb-10 grid gap-5 lg:grid-cols-2">
        <ChartCard title="In-domain metrics by number of trees" subtitle="5G-NIDD held-out test set" footer="Performance is essentially flat between 10 and 100 trees.">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={metricData} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="#eef2f6" />
              <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: "#e2e8f0" }} label={{ value: "Trees", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} width={48} />
              <Tooltip content={<ChartTooltip format={(v) => `${v.toFixed(2)}%`} labelFormat={(l) => `${l} trees`} />} />
              <Legend iconType="circle" iconSize={8} />
              {METRICS.map((m) => (
                <Line key={m.key} dataKey={m.key} name={m.label} stroke={COLORS[m.key]} strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="F1: in-domain vs cross-domain" subtitle="Adding trees does not help the cross-domain result">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={f1Data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="#eef2f6" />
              <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: "#e2e8f0" }} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} width={48} />
              <Tooltip content={<ChartTooltip format={(v) => `${v.toFixed(2)}%`} labelFormat={(l) => `${l} trees`} />} />
              <Legend iconType="circle" iconSize={8} />
              <Line dataKey="fiveg" name="5G-NIDD (in-domain)" stroke={COLORS.fiveg} strokeWidth={2.5} dot={{ r: 4 }} />
              <Line dataKey="cic" name="CICIoT2023 (cross-domain)" stroke={COLORS.cic} strokeWidth={2.5} dot={{ r: 4, shape: "square" }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Saved model size" subtitle="MB on disk (joblib)">
          <HBars
            data={rows.map((r) => ({ name: `${r.trees} trees`, value: r.size_mb }))}
            unit=" MB"
            color={COLORS.fiveg}
            tickWidth={70}
            labelFormat={(v) => `${v.toFixed(1)} MB`}
          />
        </ChartCard>

        <ChartCard title="Training time" subtitle="Seconds on 972,540 training rows">
          <HBars
            data={rows.map((r) => ({ name: `${r.trees} trees`, value: r.train_seconds }))}
            unit=" s"
            color={COLORS.fiveg}
            tickWidth={70}
            labelFormat={(v) => `${v.toFixed(1)} s`}
          />
        </ChartCard>
      </div>

      <Section title="All Random Forest results">
        <Table
          columns={[
            { key: "trees", label: "Trees", align: "right" },
            ...METRICS.map((m) => ({ key: m.key, label: `5G-NIDD ${m.label}`, align: "right", render: (r) => pct(r.in_domain[m.key]) })),
            { key: "cf1", label: "CICIoT2023 F1", align: "right", render: (r) => <span className="font-semibold text-cic">{pct(r.cross_domain?.f1)}</span> },
            { key: "train", label: "Train (s)", align: "right", render: (r) => num(r.train_seconds, 2) },
            { key: "size", label: "Size (MB)", align: "right", render: (r) => num(r.size_mb, 2) },
          ]}
          rows={rows}
        />
      </Section>

      {data.feature_importance && (
        <Section title="Feature importance — 100-tree baseline" subtitle="Impurity-based importances. They describe what the model relies on in 5G-NIDD; they do not by themselves explain the cross-domain result.">
          <ChartCard>
            <HBars
              data={[...data.feature_importance].sort((a, b) => b.Importance - a.Importance).map((f) => ({ name: f.Feature, value: f.Importance * 100 }))}
              color={COLORS.fiveg}
              tickWidth={130}
              labelFormat={(v) => `${v.toFixed(1)}%`}
            />
          </ChartCard>
        </Section>
      )}
    </>
  );
}
