import { AlertTriangle, CheckCircle2, Info, OctagonAlert, Terminal } from "lucide-react";

export function PageHeader({ eyebrow, title, children }) {
  return (
    <header className="mb-8">
      {eyebrow && (
        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-fiveg">{eyebrow}</div>
      )}
      <h1 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">{title}</h1>
      {children && <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-slate-600">{children}</p>}
    </header>
  );
}

export function Section({ title, subtitle, children, action }) {
  return (
    <section className="mb-10">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-ink">{title}</h2>
          {subtitle && <p className="mt-1 max-w-3xl text-sm text-muted">{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Card({ children, className = "", padded = true }) {
  return (
    <div className={`rounded-2xl border border-line bg-white shadow-sm ${padded ? "p-5" : ""} ${className}`}>
      {children}
    </div>
  );
}

export function ChartCard({ title, subtitle, children, footer, className = "" }) {
  return (
    <Card className={className}>
      {title && <h3 className="text-sm font-semibold text-ink">{title}</h3>}
      {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      <div className="mt-4">{children}</div>
      {footer && <p className="mt-3 text-xs leading-relaxed text-muted">{footer}</p>}
    </Card>
  );
}

const TONES = {
  neutral: "text-ink",
  fiveg: "text-fiveg",
  cic: "text-cic",
  good: "text-emerald-600",
  bad: "text-red-600",
};

export function Stat({ label, value, hint, tone = "neutral", delta, deltaTone }) {
  return (
    <div className="rounded-2xl border border-line bg-white p-4 shadow-sm">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div className={`mt-1.5 text-2xl font-bold tracking-tight tabular-nums ${TONES[tone]}`}>{value}</div>
      {delta && (
        <div className={`mt-1 text-xs font-semibold tabular-nums ${deltaTone === "bad" ? "text-red-600" : deltaTone === "good" ? "text-emerald-600" : "text-muted"}`}>
          {delta}
        </div>
      )}
      {hint && <div className="mt-1 text-xs leading-snug text-muted">{hint}</div>}
    </div>
  );
}

const CALLOUTS = {
  info: { icon: Info, cls: "border-blue-200 bg-blue-50 text-blue-900", iconCls: "text-blue-500" },
  warning: { icon: AlertTriangle, cls: "border-amber-200 bg-amber-50 text-amber-900", iconCls: "text-amber-500" },
  success: { icon: CheckCircle2, cls: "border-emerald-200 bg-emerald-50 text-emerald-900", iconCls: "text-emerald-500" },
  danger: { icon: OctagonAlert, cls: "border-red-200 bg-red-50 text-red-900", iconCls: "text-red-500" },
};

export function Callout({ tone = "info", title, children, className = "" }) {
  const { icon: Icon, cls, iconCls } = CALLOUTS[tone];
  return (
    <div className={`flex gap-3 rounded-xl border px-4 py-3 text-sm leading-relaxed ${cls} ${className}`}>
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${iconCls}`} />
      <div>
        {title && <div className="font-semibold">{title}</div>}
        <div className={title ? "mt-0.5" : ""}>{children}</div>
      </div>
    </div>
  );
}

export function Badge({ children, tone = "slate" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700",
    fiveg: "bg-fiveg-soft text-[#1d5fae]",
    cic: "bg-cic-soft text-[#b44a1f]",
    green: "bg-emerald-100 text-emerald-800",
    red: "bg-red-100 text-red-800",
    amber: "bg-amber-100 text-amber-800",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function DatasetTag({ which }) {
  return which === "fiveg" ? <Badge tone="fiveg">● 5G-NIDD · in-domain</Badge> : <Badge tone="cic">● CICIoT2023 · cross-domain</Badge>;
}

export function Table({ columns, rows, dense = false, maxHeight }) {
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-white">
      <div className="overflow-x-auto" style={maxHeight ? { maxHeight, overflowY: "auto" } : undefined}>
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-slate-50 text-xs uppercase tracking-wide text-muted">
            <tr>
              {columns.map((c) => (
                <th key={c.key} className={`whitespace-nowrap px-4 py-2.5 font-semibold ${c.align === "right" ? "text-right" : ""}`}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {rows.map((row, i) => (
              <tr key={i} className="hover:bg-slate-50/70">
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={`whitespace-nowrap px-4 ${dense ? "py-1.5" : "py-2.5"} ${c.align === "right" ? "text-right tabular-nums" : ""} ${c.className ?? ""}`}
                  >
                    {c.render ? c.render(row) : row[c.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Segmented({ options, value, onChange, disabled }) {
  return (
    <div className="inline-flex flex-wrap rounded-xl border border-line bg-slate-100 p-1">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            disabled={disabled}
            onClick={() => onChange(o.value)}
            className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition ${
              active ? "bg-white text-ink shadow-sm" : "text-slate-600 hover:text-ink"
            } disabled:cursor-not-allowed disabled:opacity-60`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Select({ label, value, onChange, options, disabled }) {
  return (
    <label className="block text-sm">
      {label && <span className="mb-1.5 block text-xs font-medium text-muted">{label}</span>}
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-xl border border-line bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-fiveg focus:ring-2 focus:ring-fiveg/20 disabled:opacity-60"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Button({ children, onClick, variant = "primary", disabled, icon: Icon, className = "" }) {
  const variants = {
    primary: "bg-ink text-white hover:bg-slate-800",
    blue: "bg-fiveg text-white hover:bg-[#1f66bd]",
    danger: "bg-red-600 text-white hover:bg-red-700",
    ghost: "border border-line bg-white text-ink hover:bg-slate-50",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold shadow-sm transition disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
    >
      {Icon && <Icon className="h-4 w-4" />}
      {children}
    </button>
  );
}

export function EmptyState({ title, children, commands }) {
  return (
    <Card className="border-dashed text-center">
      <div className="mx-auto max-w-xl py-6">
        <Terminal className="mx-auto h-8 w-8 text-slate-400" />
        <h3 className="mt-3 font-semibold text-ink">{title}</h3>
        <p className="mt-1 text-sm text-muted">{children}</p>
        {commands && (
          <pre className="mt-4 overflow-x-auto rounded-xl bg-ink px-4 py-3 text-left font-mono text-xs leading-relaxed text-slate-100">
            {commands.join("\n")}
          </pre>
        )}
      </div>
    </Card>
  );
}

export function Legend({ items }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
