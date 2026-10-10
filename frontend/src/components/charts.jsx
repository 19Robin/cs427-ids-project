import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { COLORS } from "../lib/format";

// Shared Recharts styling: thin bars, quiet grid, rounded data ends.

export function ChartTooltip({ active, payload, label, format = (v) => v, labelFormat }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line bg-white/95 px-3 py-2 text-xs shadow-lg backdrop-blur">
      {label !== undefined && (
        <div className="mb-1 font-semibold text-ink">{labelFormat ? labelFormat(label) : label}</div>
      )}
      {payload.map((p) => (
        <div key={p.dataKey ?? p.name} className="flex items-center gap-2 text-slate-600">
          <span className="h-2 w-2 rounded-sm" style={{ background: p.color ?? p.payload?.fill }} />
          <span>{p.name}</span>
          <span className="ml-auto pl-4 font-semibold tabular-nums text-ink">{format(p.value, p)}</span>
        </div>
      ))}
    </div>
  );
}

const pctTick = (v) => `${v}%`;

/**
 * Grouped vertical bars.
 * data: [{ name, [seriesKey]: value }]   series: [{ key, label, color }]
 */
export function GroupedBars({
  data,
  series,
  height = 300,
  unit = "%",
  domain = [0, 100],
  showLabels = true,
  labelDigits = 1,
  reference,
  xTickFormatter,
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 22, right: 8, left: 0, bottom: 4 }} barGap={3} barCategoryGap="22%">
        <CartesianGrid vertical={false} stroke="#eef2f6" />
        <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: "#e2e8f0" }} interval={0} tickFormatter={xTickFormatter} />
        <YAxis domain={domain} tickLine={false} axisLine={false} tickFormatter={unit === "%" ? pctTick : undefined} width={48} />
        <Tooltip
          cursor={{ fill: "rgba(148,163,184,0.12)" }}
          content={<ChartTooltip format={(v) => (v === null || v === undefined ? "–" : `${v.toFixed(2)}${unit}`)} />}
        />
        <Legend iconType="square" iconSize={10} wrapperStyle={{ paddingTop: 8 }} />
        {reference && (
          <ReferenceLine y={reference.value} stroke="#64748b" strokeDasharray="5 4" label={reference.label ? { value: reference.label, position: "insideTopRight", fontSize: 11, fill: "#64748b" } : undefined} />
        )}
        {series.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={44} animationDuration={600}>
            {showLabels && (
              <LabelList
                dataKey={s.key}
                position="top"
                formatter={(v) => (v === null || v === undefined ? "" : v.toFixed(labelDigits))}
                style={{ fontSize: 10, fill: "#475569" }}
              />
            )}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * Horizontal bars, one series. data: [{ name, value, color?, label? }]
 */
export function HBars({ data, height, unit = "%", domain, color = COLORS.cic, labelFormat, tickWidth = 150, reference }) {
  const h = height ?? Math.max(160, data.length * 30 + 40);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 64, left: 4, bottom: 4 }} barCategoryGap="22%">
        <CartesianGrid horizontal={false} stroke="#eef2f6" />
        <XAxis type="number" domain={domain ?? [0, "auto"]} tickLine={false} axisLine={false} tickFormatter={unit === "%" ? pctTick : undefined} />
        <YAxis type="category" dataKey="name" width={tickWidth} tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 11 }} />
        <Tooltip
          cursor={{ fill: "rgba(148,163,184,0.12)" }}
          content={<ChartTooltip format={(v, p) => (labelFormat ? labelFormat(v, p.payload) : `${v.toFixed(2)}${unit}`)} />}
        />
        {reference && <ReferenceLine x={reference.value} stroke={reference.color ?? "#64748b"} strokeDasharray="5 4" />}
        <Bar dataKey="value" name="Value" radius={[0, 4, 4, 0]} maxBarSize={22} animationDuration={600}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color ?? color} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            formatter={(v) => (labelFormat ? labelFormat(v) : `${v.toFixed(2)}${unit}`)}
            style={{ fontSize: 11, fill: "#334155" }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export const DATASET_SERIES = [
  { key: "fiveg", label: "5G-NIDD (in-domain)", color: COLORS.fiveg },
  { key: "cic", label: "CICIoT2023 (cross-domain)", color: COLORS.cic },
];
