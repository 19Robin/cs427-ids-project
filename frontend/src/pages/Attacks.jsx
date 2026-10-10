import { Badge, ChartCard, EmptyState, Legend, PageHeader, Section, Stat, Table } from "../components/ui";
import { HBars } from "../components/charts";
import { num, pct } from "../lib/format";

const COMPARABLE_COLOR = "#a8431b";
const OTHER_COLOR = "#f3a07f";

function weighted(rows) {
  const samples = rows.reduce((s, r) => s + r.samples, 0);
  const detected = rows.reduce((s, r) => s + r.detected, 0);
  return { samples, detected, rate: samples ? detected / samples : 0 };
}

export default function Attacks({ data }) {
  const attacks = data.attacks;
  if (!attacks) {
    return (
      <>
        <PageHeader title="Attack-type analysis" />
        <EmptyState title="No attack-type results" commands={["python src/test_attack_types.py"]}>
          results/attack_type_results.csv was not found.
        </EmptyState>
      </>
    );
  }

  const total = weighted(attacks);
  const maxRate = Math.max(...attacks.map((a) => a.rate));
  const categories = Object.entries(
    attacks.reduce((acc, a) => ((acc[a.category] ??= []).push(a), acc), {})
  )
    .map(([name, rows]) => ({ name, ...weighted(rows) }))
    .sort((a, b) => b.rate - a.rate);
  const comparable = weighted(attacks.filter((a) => a.comparable));
  const other = weighted(attacks.filter((a) => !a.comparable));
  const sorted = [...attacks].sort((a, b) => b.rate - a.rate);
  const domainMax = Math.max(10, Math.ceil(maxRate * 100 * 1.25));

  return (
    <>
      <PageHeader eyebrow="Where does it fail?" title="Attack-type analysis — CICIoT2023">
        Detection rate = share of each CICIoT2023 attack type that the 5G-trained model labelled malicious. Produced by
        src/test_attack_types.py with the <b>100-tree Random Forest baseline</b> (per-type totals match its confusion matrix).
      </PageHeader>

      <div className="mb-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Attack types" value={attacks.length} />
        <Stat label="Attack records" value={num(total.samples)} />
        <Stat label="Overall detection rate" value={pct(total.rate)} tone="bad" />
        <Stat label="Best single attack type" value={pct(maxRate)} hint={sorted[0].attack} />
      </div>

      <div className="mb-10 grid gap-5 lg:grid-cols-5">
        <ChartCard className="lg:col-span-3" title="Detection rate by attack category" subtitle="Weighted by number of records" footer="High-volume floods (DDoS, DoS, Mirai) are almost never detected; small recon, spoofing and web attacks slightly more often.">
          <HBars
            data={categories.map((c) => ({ name: c.name, value: c.rate * 100, samples: c.samples }))}
            domain={[0, domainMax]}
            tickWidth={110}
            labelFormat={(v, p) => `${v.toFixed(2)}%${p ? ` · n=${num(p.samples)}` : ""}`}
          />
        </ChartCard>
        <ChartCard className="lg:col-span-2" title="Attacks the model was trained on are still missed" subtitle="CICIoT2023 attacks with a comparable family in 5G-NIDD vs the rest">
          <div className="space-y-5 pt-2">
            {[
              { label: "Comparable to a 5G-NIDD attack", value: comparable, color: COMPARABLE_COLOR, note: "ICMP/UDP/SYN/HTTP floods, slow-rate DoS, port scans" },
              { label: "No comparable 5G-NIDD attack", value: other, color: OTHER_COLOR, note: "Mirai, spoofing, web attacks, other floods" },
            ].map((g) => (
              <div key={g.label}>
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-medium">{g.label}</span>
                  <span className="text-xl font-bold tabular-nums">{pct(g.value.rate)}</span>
                </div>
                <div className="mt-1.5 h-2.5 rounded-full bg-slate-100">
                  <div className="h-full rounded-full" style={{ width: `${Math.max((g.value.rate * 100 * 100) / domainMax, 1)}%`, background: g.color }} />
                </div>
                <div className="mt-1 text-xs text-muted">{num(g.value.samples)} records · {g.note}</div>
              </div>
            ))}
          </div>
          <p className="mt-5 text-xs leading-relaxed text-muted">
            Bars are scaled to {domainMax}%. Even familiar attack families are missed, suggesting the same attack looks different in the
            common feature space when it comes from a different network.
          </p>
        </ChartCard>
      </div>

      <Section title="Detection rate by attack type" subtitle={`The axis is limited to ${domainMax}% so differences are visible; no type exceeds ${pct(maxRate)}.`}>
        <ChartCard>
          <div className="mb-3">
            <Legend
              items={[
                { label: "◆ Comparable attack family exists in 5G-NIDD", color: COMPARABLE_COLOR },
                { label: "No comparable attack in 5G-NIDD", color: OTHER_COLOR },
              ]}
            />
          </div>
          <HBars
            data={sorted.map((a) => ({
              name: a.comparable ? `◆ ${a.attack}` : a.attack,
              value: a.rate * 100,
              color: a.comparable ? COMPARABLE_COLOR : OTHER_COLOR,
            }))}
            domain={[0, domainMax]}
            tickWidth={190}
          />
        </ChartCard>
      </Section>

      <Section title="Full table">
        <Table
          maxHeight={420}
          dense
          columns={[
            { key: "attack", label: "CICIoT2023 attack", className: "font-medium" },
            { key: "category", label: "Category" },
            { key: "comparable", label: "Comparable 5G-NIDD attack", render: (a) => (a.comparable ? <Badge tone="fiveg">{a.comparable}</Badge> : <span className="text-muted">—</span>) },
            { key: "samples", label: "Records", align: "right", render: (a) => num(a.samples) },
            { key: "detected", label: "Detected", align: "right", render: (a) => num(a.detected) },
            { key: "rate", label: "Detection rate", align: "right", render: (a) => pct(a.rate, 3) },
          ]}
          rows={sorted}
        />
        <p className="mt-2 text-xs text-muted">
          Comparable-family mapping follows the attack types listed in the 5G-NIDD documentation (ICMP/UDP/SYN/HTTP flood, slow-rate DoS,
          SYN/TCP-connect/UDP scans).
        </p>
      </Section>
    </>
  );
}
