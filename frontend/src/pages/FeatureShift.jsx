import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, ChartCard, Callout, EmptyState, Legend, PageHeader, Section, Table } from "../components/ui";
import { ChartTooltip, DATASET_SERIES, GroupedBars } from "../components/charts";
import { COLORS, compact, num } from "../lib/format";

function MedianPanel({ row }) {
  const values = [
    { label: "5G-NIDD", value: row.fiveg.median, color: COLORS.fiveg },
    { label: "CICIoT2023", value: row.cic.median, color: COLORS.cic },
  ];
  const max = Math.max(...values.map((v) => v.value), 1e-9);
  return (
    <Card>
      <div className="text-sm font-semibold">{row.feature.replaceAll("_", " ")}</div>
      <div className="text-xs text-muted">median</div>
      <div className="mt-4 flex h-32 items-end gap-4">
        {values.map((v) => (
          <div key={v.label} className="flex flex-1 flex-col items-center justify-end">
            <div className="mb-1 text-sm font-bold tabular-nums">{compact(v.value)}</div>
            <div className="w-full max-w-14 rounded-t-md" style={{ height: `${Math.max((v.value / max) * 88, 2)}px`, background: v.color }} />
            <div className="mt-1.5 text-[11px] text-muted">{v.label}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function FeatureShift({ data }) {
  const rows = data.feature_shift;
  const protocols = data.protocol_shift;
  if (!rows) {
    return (
      <>
        <PageHeader title="Feature and protocol shift" />
        <EmptyState title="No feature-shift results" commands={["python src/analyse_feature_shift.py"]}>
          results/feature_distribution_shift.csv was not found.
        </EmptyState>
      </>
    );
  }
  const get = (f) => rows.find((r) => r.feature === f);
  const smdData = rows.map((r) => ({ name: r.feature, value: r.smd }));
  const limit = Math.ceil(Math.max(...rows.map((r) => Math.abs(r.smd ?? 0))) * 1.25);
  const rate = get("Rate");
  const pc = get("Packet_Count");
  const mps = get("Mean_Packet_Size");
  const ttl = get("TTL");

  return (
    <>
      <PageHeader eyebrow="Domain shift" title="Feature and protocol distribution shift">
        Domain shift means the traffic seen in training differs statistically from the traffic seen in deployment. The seven
        common features are compared between 5G-NIDD (training environment) and CICIoT2023 (deployment environment).
      </PageHeader>

      <div className="mb-10 grid gap-5 lg:grid-cols-2">
        <ChartCard
          title="Size of the shift per feature"
          subtitle="Standardised mean difference: (CICIoT2023 mean − 5G-NIDD mean) / pooled std"
          footer="A descriptive effect size derived from the mean and std columns, not a statistical test."
        >
          <div className="mb-2">
            <Legend items={[{ label: "Higher in CICIoT2023", color: COLORS.cic }, { label: "Higher in 5G-NIDD", color: COLORS.fiveg }]} />
          </div>
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={smdData} layout="vertical" margin={{ top: 4, right: 48, left: 4, bottom: 4 }}>
              <CartesianGrid horizontal={false} stroke="#eef2f6" />
              <XAxis type="number" domain={[-limit, limit]} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" width={120} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <Tooltip cursor={{ fill: "rgba(148,163,184,0.12)" }} content={<ChartTooltip format={(v) => v.toFixed(2)} />} />
              <ReferenceLine x={0} stroke="#475569" />
              <Bar dataKey="value" name="Std. mean difference" radius={4} maxBarSize={22}>
                {smdData.map((d) => <Cell key={d.name} fill={d.value > 0 ? COLORS.cic : COLORS.fiveg} />)}
                <LabelList dataKey="value" position="right" formatter={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))} style={{ fontSize: 11, fill: "#334155" }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <div>
          <div className="mb-3 text-sm font-semibold">Medians of the continuous features</div>
          <div className="grid grid-cols-2 gap-4">
            {["Rate", "Packet_Count", "Mean_Packet_Size", "TTL"].map((f) => get(f) && <MedianPanel key={f} row={get(f)} />)}
          </div>
          <p className="mt-2 text-xs text-muted">Each panel has its own scale because the features have different units.</p>
        </div>
      </div>

      <Section title="Observations from the data">
        <div className="grid gap-4 md:grid-cols-2">
          {rate && (
            <Card><b>Rate</b><p className="mt-1 text-sm text-slate-600">Median {num(rate.fiveg.median)} in 5G-NIDD vs {num(rate.cic.median)} in CICIoT2023; the mean is {(rate.cic.mean / rate.fiveg.mean).toFixed(1)}× higher. Most 5G-NIDD records have a rate of zero.</p></Card>
          )}
          {pc && (
            <Card><b>Packet_Count</b><p className="mt-1 text-sm text-slate-600">Median {num(pc.fiveg.median)} vs {num(pc.cic.median)} — the largest standardised shift ({pc.smd.toFixed(2)}). CICIoT2023 values are capped at {num(pc.cic.max)}, pointing to a different definition as well as different traffic.</p></Card>
          )}
          {mps && (
            <Card><b>Mean_Packet_Size</b><p className="mt-1 text-sm text-slate-600">Median {num(mps.fiveg.median)} vs {num(mps.cic.median)} bytes — the smallest standardised shift ({mps.smd.toFixed(2)}).</p></Card>
          )}
          {ttl && (
            <Card><b>TTL</b><p className="mt-1 text-sm text-slate-600">Medians almost identical ({num(ttl.fiveg.median)} vs {num(ttl.cic.median)}) but the spread differs strongly (std {ttl.fiveg.std.toFixed(1)} vs {ttl.cic.std.toFixed(1)}). TTL is the most important feature of the 5G-trained Random Forest.</p></Card>
          )}
        </div>
      </Section>

      {protocols && (
        <Section title="Protocol distribution shift" subtitle="The dominant protocol changes from UDP in 5G-NIDD to TCP in CICIoT2023.">
          <div className="grid gap-5 lg:grid-cols-5">
            <ChartCard className="lg:col-span-3" footer="5G-NIDD: share of flows using each protocol (0/1 per flow). CICIoT2023: mean of a fractional indicator, so not a strict share of records.">
              <GroupedBars height={280} series={DATASET_SERIES.map((s) => ({ ...s, label: s.label.split(" (")[0] }))} data={protocols.map((p) => ({ name: p.protocol, fiveg: p.fiveg * 100, cic: p.cic * 100 }))} />
            </ChartCard>
            <div className="lg:col-span-2">
              <Table
                columns={[
                  { key: "protocol", label: "Protocol", className: "font-semibold" },
                  { key: "a", label: "5G-NIDD", align: "right", render: (p) => `${(p.fiveg * 100).toFixed(1)}%` },
                  { key: "b", label: "CICIoT2023", align: "right", render: (p) => `${(p.cic * 100).toFixed(1)}%` },
                  { key: "d", label: "Change", align: "right", render: (p) => <b>{((p.cic - p.fiveg) * 100 > 0 ? "+" : "") + ((p.cic - p.fiveg) * 100).toFixed(1)} pp</b> },
                ]}
                rows={protocols}
              />
            </div>
          </div>
        </Section>
      )}

      <Section title="Summary statistics">
        <Table
          dense
          columns={[
            { key: "feature", label: "Feature", className: "font-semibold" },
            { key: "m5", label: "5G median", align: "right", render: (r) => compact(r.fiveg.median) },
            { key: "mc", label: "CIC median", align: "right", render: (r) => compact(r.cic.median) },
            { key: "a5", label: "5G mean", align: "right", render: (r) => compact(r.fiveg.mean) },
            { key: "ac", label: "CIC mean", align: "right", render: (r) => compact(r.cic.mean) },
            { key: "s5", label: "5G std", align: "right", render: (r) => compact(r.fiveg.std) },
            { key: "sc", label: "CIC std", align: "right", render: (r) => compact(r.cic.std) },
            { key: "r5", label: "5G range", align: "right", render: (r) => `${compact(r.fiveg.min)} – ${compact(r.fiveg.max)}` },
            { key: "rc", label: "CIC range", align: "right", render: (r) => `${compact(r.cic.min)} – ${compact(r.cic.max)}` },
            { key: "smd", label: "Std. mean diff.", align: "right", render: (r) => <b>{r.smd?.toFixed(2)}</b> },
          ]}
          rows={rows}
        />
      </Section>

      <Callout tone="info" title="Interpretation">
        Feature and protocol distribution differences provide evidence of domain shift between the two environments that may contribute
        to the cross-domain degradation. They are descriptive: on their own they do not prove which feature causes the failure. The
        Improved Experiments page tests this directly.
      </Callout>
    </>
  );
}
