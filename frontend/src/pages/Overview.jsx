import {
  ArrowRight,
  ArrowDown,
  Cpu,
  Database,
  Gauge,
  Network,
  RadioTower,
  ShieldCheck,
  Smartphone,
  Shuffle,
} from "lucide-react";
import { Card, Callout, PageHeader, Section, Table } from "../components/ui";
import { compact, num, pct } from "../lib/format";

function PipelineStep({ icon: Icon, title, text, tone }) {
  const tones = {
    fiveg: "border-fiveg/30 bg-fiveg-soft text-[#1d5fae]",
    cic: "border-cic/30 bg-cic-soft text-[#b44a1f]",
    neutral: "border-line bg-white text-ink",
  };
  return (
    <div className={`flex min-w-0 flex-1 flex-col items-center rounded-xl border px-3 py-3 text-center ${tones[tone]}`}>
      <Icon className="h-5 w-5" />
      <div className="mt-1.5 text-xs font-semibold leading-tight">{title}</div>
      <div className="mt-0.5 text-[11px] leading-snug opacity-80">{text}</div>
    </div>
  );
}

function Arrow() {
  return (
    <>
      <ArrowRight className="hidden h-4 w-4 shrink-0 text-slate-400 md:block" />
      <ArrowDown className="h-4 w-4 shrink-0 self-center text-slate-400 md:hidden" />
    </>
  );
}

function Pipeline() {
  return (
    <Card>
      <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
        <PipelineStep icon={Smartphone} title="UEs / attackers" text="attached via 5G RAN" tone="fiveg" />
        <Arrow />
        <PipelineStep icon={RadioTower} title="5G network" text="5G-NIDD test network" tone="fiveg" />
        <Arrow />
        <PipelineStep icon={Network} title="Flow records" text="Argus, per flow" tone="fiveg" />
        <Arrow />
        <PipelineStep icon={Shuffle} title="7 common features" text="rate, packets, size, TTL, protocol" tone="neutral" />
        <Arrow />
        <PipelineStep icon={Cpu} title="Lightweight ML IDS" text="RF · DT · LR · XGBoost" tone="neutral" />
        <Arrow />
        <PipelineStep icon={ShieldCheck} title="Benign / Malicious" text="per flow" tone="neutral" />
      </div>
      <div className="mt-4 grid gap-3 border-t border-line pt-4 sm:grid-cols-2">
        <div className="flex items-center gap-3 rounded-xl bg-fiveg-soft/60 px-4 py-3 text-sm">
          <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-fiveg" />
          <span><b>In-domain test:</b> 20% held-out 5G-NIDD flows</span>
        </div>
        <div className="flex items-center gap-3 rounded-xl bg-cic-soft/70 px-4 py-3 text-sm">
          <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-cic" />
          <span><b>Cross-domain test:</b> CICIoT2023 IoT network, <b>no retraining</b></span>
        </div>
      </div>
    </Card>
  );
}

const MOBILE_POINTS = [
  {
    icon: RadioTower,
    title: "Real 5G traffic",
    text: "5G-NIDD was generated on a 5G test network: benign users and attackers connect through 5G base stations and launch DoS/DDoS floods and port scans.",
  },
  {
    icon: Smartphone,
    title: "Massive IoT on 5G",
    text: "5G connects huge numbers of low-cost devices (mMTC). Compromised devices can flood or scan the mobile network, so operators need network-level detection.",
  },
  {
    icon: Gauge,
    title: "Edge constraints",
    text: "An IDS close to the traffic (network edge / MEC) has limited memory and compute, so model size and per-flow latency matter as much as accuracy.",
  },
  {
    icon: Shuffle,
    title: "Changing environments",
    text: "Cells, operators, slices and IoT verticals all produce different traffic. A 5G-trained IDS will be used in environments it never saw in training.",
  },
];

const FEATURE_ROWS = [
  ["Rate", "Rate", "Rate", "5G-NIDD median 0; CICIoT2023 median ≈ 24,606"],
  ["Packet_Count", "TotPkts", "Number", "CICIoT2023 never exceeds 100; 5G-NIDD median 2"],
  ["Mean_Packet_Size", "TotBytes / TotPkts", "AVG", "CICIoT2023 max 9,398 B exceeds a 1,500 B MTU"],
  ["TTL", "sTtl", "Time_To_Live", "Medians 63 vs 64; spread differs (std 55.9 vs 14.4)"],
  ["TCP / UDP / ICMP", "one-hot of Proto", "TCP / UDP / ICMP", "0/1 in 5G-NIDD; fractional in CICIoT2023"],
].map(([feature, fiveg, cic, caveat]) => ({ feature, fiveg, cic, caveat }));

export default function Overview({ data }) {
  const { datasets, algorithms, improved } = data;
  const fiveg = datasets.fiveg;
  const cic = datasets.cic;
  const main = improved?.results?.filter((r) => r.Experiment === "Main") ?? [];
  const improvedIn = main.filter((r) => r.Test_Set.startsWith("5G-NIDD"));
  const improvedCross = main.filter((r) => r.Test_Set.startsWith("CICIoT2023"));
  const range = (rows, key) =>
    rows.length ? [Math.min(...rows.map((r) => r[key])), Math.max(...rows.map((r) => r[key]))] : null;
  const inF1 = range(improvedIn, "F1");
  const crossBal = range(improvedCross, "Balanced_Accuracy");

  return (
    <>
      {/* Hero */}
      <div className="relative mb-10 overflow-hidden rounded-3xl bg-ink px-6 py-10 text-white sm:px-10">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-fiveg/30 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 right-40 h-56 w-56 rounded-full bg-cic/25 blur-3xl" />
        <div className="relative">
          <div className="text-xs font-semibold uppercase tracking-wider text-[#7fb6f2]">CS427 Mobile Communications · Research project</div>
          <h1 className="mt-3 max-w-3xl text-3xl font-bold leading-tight tracking-tight sm:text-4xl">
            Cross-Domain Evaluation of a Lightweight ML Intrusion Detection System for 5G Network Traffic
          </h1>
          <p className="mt-4 max-w-3xl text-[15px] leading-relaxed text-slate-300">
            How well does an IDS trained on traffic from a <b className="text-white">5G mobile network</b> detect
            attacks when it is moved, <b className="text-white">without retraining</b>, to a different IoT network?
          </p>
          <div className="mt-6 flex flex-wrap gap-2 text-xs font-semibold">
            <span className="rounded-full bg-fiveg/25 px-3 py-1 text-[#bcd8f7]">Train: 5G-NIDD</span>
            <span className="rounded-full bg-cic/25 px-3 py-1 text-[#f8c9b5]">Test without retraining: CICIoT2023 (not 5G)</span>
          </div>
        </div>
      </div>

      {/* Headline numbers */}
      {algorithms && (
        <Section title="Headline result" subtitle="F1 score of each algorithm on held-out 5G-NIDD traffic and on CICIoT2023 (original experiment).">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {algorithms.map((a) => (
              <Card key={a.model}>
                <div className="text-sm font-semibold text-ink">{a.model}</div>
                <div className="mt-4 flex items-end justify-between gap-2">
                  <div>
                    <div className="text-[11px] font-medium uppercase tracking-wide text-muted">5G-NIDD</div>
                    <div className="text-2xl font-bold tabular-nums text-fiveg">{pct(a.in_domain.f1, 1)}</div>
                  </div>
                  <ArrowRight className="mb-2 h-4 w-4 text-slate-300" />
                  <div className="text-right">
                    <div className="text-[11px] font-medium uppercase tracking-wide text-muted">CICIoT2023</div>
                    <div className="text-2xl font-bold tabular-nums text-cic">{pct(a.cross_domain.f1, 1)}</div>
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-fiveg" style={{ width: `${a.in_domain.f1 * 100}%` }} />
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-cic" style={{ width: `${Math.max(a.cross_domain.f1 * 100, 0.6)}%` }} />
                </div>
                <div className="mt-3 text-xs font-semibold text-red-600">−{a.f1_drop_pp.toFixed(2)} percentage points</div>
              </Card>
            ))}
          </div>
          <Callout tone="warning" className="mt-4" title="Every algorithm fails the same way">
            Models reach about 80–83% F1 on 5G-NIDD but almost completely fail to detect attacks in CICIoT2023.
            Changing the algorithm does not fix the cross-domain problem.
            {inF1 && crossBal && (
              <>
                {" "}After removing duplicates and balancing classes, in-domain F1 rises to {pct(inF1[0], 0)}–{pct(inF1[1], 0)},
                while cross-domain balanced accuracy stays at {pct(crossBal[0], 0)}–{pct(crossBal[1], 0)} (50% = chance).
              </>
            )}
          </Callout>
        </Section>
      )}

      <Section title="Experimental pipeline" subtitle="One trained model, two test environments.">
        <Pipeline />
      </Section>

      <Section title="Why this is a Mobile Communications project">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {MOBILE_POINTS.map(({ icon: Icon, title, text }) => (
            <Card key={title}>
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-fiveg-soft text-fiveg">
                <Icon className="h-5 w-5" />
              </div>
              <div className="mt-3 font-semibold text-ink">{title}</div>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{text}</p>
            </Card>
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-muted">
          Scope: the seven common features are generic IP-flow features, not 5G radio or core signalling features. The 5G
          link comes from the training traffic and the deployment scenario. CICIoT2023 is not a 5G dataset.
        </p>
      </Section>

      <Section title="Datasets">
        <div className="grid gap-4 md:grid-cols-2">
          <Card className="border-t-4 border-t-fiveg">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-fiveg" />
              <div className="font-semibold">5G-NIDD — source / training domain</div>
            </div>
            <p className="mt-1 text-sm text-muted">5G network intrusion dataset. Used for training and in-domain testing.</p>
            {fiveg && (
              <dl className="mt-4 grid grid-cols-3 gap-3">
                <div><dt className="text-xs text-muted">Records</dt><dd className="text-xl font-bold tabular-nums">{compact(fiveg.rows)}</dd></div>
                <div><dt className="text-xs text-muted">Train / test</dt><dd className="text-xl font-bold tabular-nums">80 / 20</dd></div>
                <div><dt className="text-xs text-muted">Malicious</dt><dd className="text-xl font-bold tabular-nums text-fiveg">{pct(fiveg.malicious_share, 1)}</dd></div>
              </dl>
            )}
          </Card>
          <Card className="border-t-4 border-t-cic">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-cic" />
              <div className="font-semibold">CICIoT2023 — target / cross-domain</div>
            </div>
            <p className="mt-1 text-sm text-muted">IoT lab network with 105 devices (not 5G). Used only for testing.</p>
            {cic && (
              <dl className="mt-4 grid grid-cols-3 gap-3">
                <div><dt className="text-xs text-muted">Records</dt><dd className="text-xl font-bold tabular-nums">{compact(cic.rows)}</dd></div>
                <div><dt className="text-xs text-muted">Attack types</dt><dd className="text-xl font-bold tabular-nums">{num(cic.attack_types)}</dd></div>
                <div><dt className="text-xs text-muted">Malicious</dt><dd className="text-xl font-bold tabular-nums text-cic">{pct(cic.malicious_share, 1)}</dd></div>
              </dl>
            )}
          </Card>
        </div>
      </Section>

      <Section
        title="Common feature representation"
        subtitle="Both datasets are mapped onto the same seven features so a model trained on 5G-NIDD can be applied to CICIoT2023 unchanged."
      >
        <Table
          columns={[
            { key: "feature", label: "Feature", className: "font-semibold" },
            { key: "fiveg", label: "5G-NIDD source", className: "font-mono text-xs" },
            { key: "cic", label: "CICIoT2023 source", className: "font-mono text-xs" },
            { key: "caveat", label: "Mapping caveat (from the processed data)", className: "text-muted whitespace-normal" },
          ]}
          rows={FEATURE_ROWS}
        />
      </Section>
    </>
  );
}
