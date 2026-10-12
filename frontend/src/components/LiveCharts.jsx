import { useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, Legend, Segmented } from "./ui";
import { COLORS } from "../lib/format";

const toMs = (iso) => new Date(iso).getTime();
const clock = (ms) =>
  new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

const RANGES = [
  { value: "120", label: "2 min" },
  { value: "300", label: "5 min" },
  { value: "all", label: "All" },
];

function LiveTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-xl border border-line bg-white/95 px-3 py-2 text-xs shadow-lg backdrop-blur">
      <div className="mb-1 font-semibold text-ink">{clock(p.t)}</div>
      {p.source && (
        <div className="text-slate-600">
          {p.source} → {p.destination} ({p.protocol})
        </div>
      )}
      {p.peak !== undefined && <div>Highest flow probability: <b>{(p.peak * 100).toFixed(0)}%</b></div>}
      {p.p !== undefined && <div>Malicious probability: <b>{(p.p * 100).toFixed(0)}%</b></div>}
      {p.rate !== undefined && <div>Traffic rate: <b>{p.rate.toFixed(1)} pkt/s</b></div>}
      {p.benign !== undefined && (
        <div>
          Flows: <b>{p.benign + p.malicious}</b> ({p.malicious} malicious)
        </div>
      )}
      {p.verdict && (
        <div>
          Verdict: <b style={{ color: p.verdict === "MALICIOUS" ? COLORS.malicious : COLORS.benign }}>{p.verdict}</b>
        </div>
      )}
    </div>
  );
}

function ChartShell({ title, subtitle, children, legend, className = "" }) {
  return (
    <Card className={className}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-ink">{title}</div>
          <div className="text-xs text-muted">{subtitle}</div>
        </div>
        {legend}
      </div>
      <div className="mt-3">{children}</div>
    </Card>
  );
}

const axisProps = {
  tickLine: false,
  axisLine: false,
  tick: { fontSize: 11, fill: "#94a3b8" },
};

export default function LiveCharts({ windows }) {
  const [range, setRange] = useState("120");

  const all = windows.map((w) => ({
    t: toMs(w.Time),
    peak: w.Max_Malicious_Probability ?? 0,
    rate: w.Traffic_Rate ?? 0,
    verdict: w.Verdict,
    malicious: w.Flagged_Flows ?? 0,
    benign: Math.max((w.Flows ?? 0) - (w.Flagged_Flows ?? 0), 0),
  }));
  const end = all.length ? all[all.length - 1].t : Date.now();
  const start = range === "all" || !all.length ? (all[0]?.t ?? end) : end - Number(range) * 1000;
  const data = all.filter((d) => d.t >= start);
  const domain = [Math.max(start, data[0]?.t ?? start), end];
  // five evenly spaced time ticks across the visible range
  const ticks = domain[1] > domain[0] ? [0, 1, 2, 3, 4].map((i) => domain[0] + ((domain[1] - domain[0]) * i) / 4) : [domain[0]];
  const xAxis = (
    <XAxis
      type="number"
      dataKey="t"
      domain={domain}
      ticks={ticks}
      tickFormatter={clock}
      {...axisProps}
      axisLine={{ stroke: "#e2e8f0" }}
    />
  );

  return (
    <div className="mb-6 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-ink">
          <span className="pulse-dot h-2 w-2 rounded-full bg-emerald-500" /> Live graphs
        </div>
        <Segmented options={RANGES} value={range} onChange={setRange} />
      </div>

      {/* Threat level */}
      <ChartShell
        title="Threat level"
        subtitle="Highest malicious probability among the flows of each window. Red markers = windows with flows labelled malicious (bigger = more flows)."
        legend={
          <Legend
            items={[
              { label: "Window threat level", color: "#475569" },
              { label: "Window with malicious flows", color: COLORS.malicious },
              { label: "Alert zone (≥ 50%)", color: "#fecaca" },
            ]}
          />
        }
      >
        <ResponsiveContainer width="100%" height={250}>
          <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="threatFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={COLORS.malicious} stopOpacity={0.45} />
                <stop offset="50%" stopColor="#f59e0b" stopOpacity={0.18} />
                <stop offset="100%" stopColor={COLORS.benign} stopOpacity={0.08} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="#f1f5f9" />
            <ReferenceArea y1={0.5} y2={1} fill="#fee2e2" fillOpacity={0.45} ifOverflow="extendDomain" />
            <ReferenceLine
              y={0.5}
              stroke="#ef4444"
              strokeDasharray="4 4"
              strokeOpacity={0.6}
              label={{ value: "decision threshold", position: "insideTopLeft", fontSize: 10, fill: "#b91c1c" }}
            />
            {xAxis}
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v) => `${v * 100}%`} width={44} {...axisProps} />
            <Tooltip content={<LiveTooltip />} cursor={{ stroke: "#cbd5e1", strokeDasharray: "3 3" }} />
            <Area
              type="monotone"
              dataKey="peak"
              stroke="#475569"
              strokeWidth={2.2}
              fill="url(#threatFill)"
              isAnimationActive={false}
              activeDot={{ r: 5, fill: "#0f172a", stroke: "white", strokeWidth: 2 }}
              dot={(props) =>
                props.payload.malicious > 0 ? (
                  <circle
                    key={props.index}
                    cx={props.cx}
                    cy={props.cy}
                    r={3.5 + Math.min(props.payload.malicious, 12) * 0.45}
                    fill={COLORS.malicious}
                    fillOpacity={0.9}
                    stroke="white"
                    strokeWidth={1.5}
                  />
                ) : (
                  <g key={props.index} />
                )
              }
            />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartShell>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Traffic rate */}
        <ChartShell
          title="Traffic rate"
          subtitle="Packets per second in each window"
          legend={<Legend items={[{ label: "Rate", color: COLORS.fiveg }, { label: "Malicious window", color: COLORS.malicious }]} />}
        >
          <ResponsiveContainer width="100%" height={210}>
            <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="rateFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={COLORS.fiveg} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={COLORS.fiveg} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} stroke="#f1f5f9" />
              {xAxis}
              <YAxis width={44} {...axisProps} />
              <Tooltip content={<LiveTooltip />} cursor={{ stroke: "#cbd5e1", strokeDasharray: "3 3" }} />
              <Area
                type="monotone"
                dataKey="rate"
                stroke={COLORS.fiveg}
                strokeWidth={2.2}
                fill="url(#rateFill)"
                isAnimationActive={false}
                activeDot={{ r: 5, fill: COLORS.fiveg, stroke: "white", strokeWidth: 2 }}
                dot={(props) =>
                  props.payload.verdict === "MALICIOUS" ? (
                    <circle key={props.index} cx={props.cx} cy={props.cy} r={4.5} fill={COLORS.malicious} stroke="white" strokeWidth={1.5} />
                  ) : (
                    <g key={props.index} />
                  )
                }
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartShell>

        {/* Flows per window */}
        <ChartShell
          title="Flows per window"
          subtitle="How many flows the model saw, and how many it flagged"
          legend={<Legend items={[{ label: "Benign", color: COLORS.benign }, { label: "Malicious", color: COLORS.malicious }]} />}
        >
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }} barCategoryGap="18%">
              <CartesianGrid vertical={false} stroke="#f1f5f9" />
              <XAxis dataKey="t" tickFormatter={clock} minTickGap={60} {...axisProps} axisLine={{ stroke: "#e2e8f0" }} />
              <YAxis allowDecimals={false} width={44} {...axisProps} />
              <Tooltip content={<LiveTooltip />} cursor={{ fill: "rgba(148,163,184,0.12)" }} />
              <Bar dataKey="benign" stackId="f" fill={COLORS.benign} fillOpacity={0.85} isAnimationActive={false} />
              <Bar dataKey="malicious" stackId="f" fill={COLORS.malicious} radius={[3, 3, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartShell>
      </div>
    </div>
  );
}
