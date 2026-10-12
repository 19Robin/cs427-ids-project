import { useRef, useState } from "react";
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
import { ChevronLeft, ChevronRight, ChevronsLeft, History, Radio } from "lucide-react";
import { Card, Legend, Segmented } from "./ui";
import { COLORS } from "../lib/format";

const toMs = (iso) => new Date(iso).getTime();
const clock = (ms) =>
  new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

const SPANS = [
  { value: "60", label: "1 min" },
  { value: "120", label: "2 min" },
  { value: "300", label: "5 min" },
];

function LiveTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-xl border border-line bg-white/95 px-3 py-2 text-xs shadow-lg backdrop-blur">
      <div className="mb-1 font-semibold text-ink">{clock(p.t)}</div>
      {p.peak !== undefined && <div>Highest flow probability: <b>{(p.peak * 100).toFixed(0)}%</b></div>}
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

function NavButton({ onClick, disabled, title, children }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-white text-slate-600 shadow-sm transition hover:bg-slate-50 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  );
}

export default function LiveCharts({ windows }) {
  const [span, setSpan] = useState("120");
  const [viewEnd, setViewEnd] = useState(null); // null = follow live data
  const drag = useRef(null);

  const all = windows.map((w) => ({
    t: toMs(w.Time),
    peak: w.Max_Malicious_Probability ?? 0,
    rate: w.Traffic_Rate ?? 0,
    verdict: w.Verdict,
    malicious: w.Flagged_Flows ?? 0,
    benign: Math.max((w.Flows ?? 0) - (w.Flagged_Flows ?? 0), 0),
  }));

  const spanMs = Number(span) * 1000;
  const first = all[0]?.t ?? Date.now();
  const latest = all[all.length - 1]?.t ?? first;
  const scrollable = latest - first > spanMs;
  const minEnd = first + spanMs;
  const clampEnd = (t) => Math.min(latest, Math.max(minEnd, t));
  const isLive = viewEnd === null || !scrollable;
  const end = isLive ? Math.max(latest, minEnd) : clampEnd(viewEnd);
  const start = end - spanMs;
  const data = all.filter((d) => d.t >= start && d.t <= end);
  const domain = [start, end];
  const ticks = [0, 1, 2, 3, 4].map((i) => start + (spanMs * i) / 4);

  const goTo = (t) => {
    const next = clampEnd(t);
    setViewEnd(next >= latest ? null : next);
  };
  const shift = (fraction) => goTo(end + spanMs * fraction);

  // drag the threat chart to scroll through time
  const onPointerDown = (e) => {
    if (!scrollable) return;
    drag.current = { x: e.clientX, end, width: e.currentTarget.getBoundingClientRect().width };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    goTo(drag.current.end - (dx / drag.current.width) * spanMs);
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const xAxis = (
    <XAxis type="number" dataKey="t" domain={domain} ticks={ticks} tickFormatter={clock} allowDataOverflow {...axisProps} axisLine={{ stroke: "#e2e8f0" }} />
  );

  return (
    <div className="mb-6 space-y-5">
      {/* Time navigation */}
      <Card className="!p-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink">
            {isLive ? (
              <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                <span className="pulse-dot h-2 w-2 rounded-full bg-emerald-500" /> LIVE
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-700">
                <History className="h-3.5 w-3.5" /> HISTORY
              </span>
            )}
            <span className="tabular-nums text-slate-600">
              {clock(start)} – {clock(end)}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <NavButton onClick={() => goTo(minEnd)} disabled={!scrollable || end <= minEnd} title="Go to the start of the session">
              <ChevronsLeft className="h-4 w-4" />
            </NavButton>
            <NavButton onClick={() => shift(-0.5)} disabled={!scrollable || end <= minEnd} title="Back half a window">
              <ChevronLeft className="h-4 w-4" />
            </NavButton>
            <NavButton onClick={() => shift(0.5)} disabled={isLive} title="Forward half a window">
              <ChevronRight className="h-4 w-4" />
            </NavButton>
            <button
              onClick={() => setViewEnd(null)}
              disabled={isLive}
              className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-emerald-600 px-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-default disabled:bg-emerald-100 disabled:text-emerald-700"
            >
              <Radio className="h-4 w-4" /> Live
            </button>
            <Segmented options={SPANS} value={span} onChange={setSpan} />
          </div>
        </div>

        {/* Overview of the whole session */}
        <div className="mt-3">
          <ResponsiveContainer width="100%" height={46}>
            <AreaChart data={all} margin={{ top: 2, right: 6, left: 6, bottom: 0 }}>
              <XAxis type="number" dataKey="t" domain={[first, Math.max(latest, minEnd)]} hide />
              <YAxis domain={[0, 1]} hide />
              <ReferenceArea x1={start} x2={end} fill={COLORS.fiveg} fillOpacity={0.12} stroke={COLORS.fiveg} strokeOpacity={0.5} ifOverflow="hidden" />
              <Area type="monotone" dataKey="peak" stroke="#94a3b8" strokeWidth={1.2} fill="#e2e8f0" isAnimationActive={false} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <input
            type="range"
            min={minEnd}
            max={Math.max(latest, minEnd)}
            step={1000}
            value={end}
            disabled={!scrollable}
            onChange={(e) => goTo(Number(e.target.value))}
            className="mt-1 w-full accent-[#2a78d6] disabled:opacity-40"
            aria-label="Scroll through the monitoring session"
          />
          <div className="flex justify-between text-[11px] text-muted">
            <span>{clock(first)}</span>
            <span>{scrollable ? "Drag the slider or the threat chart to look back" : `History builds up as monitoring runs (${all.length} windows)`}</span>
            <span>{clock(latest)}</span>
          </div>
        </div>
      </Card>

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
        <div
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          className={`select-none ${scrollable ? (drag.current ? "cursor-grabbing" : "cursor-grab") : ""}`}
          style={{ touchAction: "pan-y" }}
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
        </div>
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
