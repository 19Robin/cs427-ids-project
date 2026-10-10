import { ChartCard, Callout, EmptyState, PageHeader, Section, Stat, Table } from "../components/ui";
import { DATASET_SERIES, GroupedBars, HBars } from "../components/charts";
import { COLORS, METRICS, num, pct, sizeLabel } from "../lib/format";

export default function Algorithms({ data }) {
  const algos = data.algorithms;
  if (!algos) {
    return (
      <>
        <PageHeader title="Algorithm comparison" />
        <EmptyState title="No algorithm results" commands={["python src/test_other_models.py"]}>
          results/other_models_comparison.csv was not found.
        </EmptyState>
      </>
    );
  }

  const f1Data = algos.map((a) => ({ name: a.model, fiveg: a.in_domain.f1 * 100, cic: a.cross_domain.f1 * 100 }));
  const lr = algos.find((a) => a.model === "Logistic Regression");

  return (
    <>
      <PageHeader eyebrow="Is it the algorithm?" title="Machine learning algorithm comparison">
        Four algorithms trained on the same 5G-NIDD split with the same seven features, then tested on the same held-out
        5G-NIDD set and on all CICIoT2023 records. If the failure were caused by one algorithm, the others should do better.
      </PageHeader>

      <div className="mb-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {algos.map((a) => (
          <Stat
            key={a.model}
            label={a.model}
            value={`−${a.f1_drop_pp.toFixed(1)} pp`}
            tone="bad"
            hint={`F1 ${pct(a.in_domain.f1, 1)} → ${pct(a.cross_domain.f1, 2)}`}
          />
        ))}
      </div>

      <div className="mb-10 grid gap-5 lg:grid-cols-5">
        <ChartCard className="lg:col-span-3" title="F1 score: in-domain vs cross-domain" subtitle="Same trained model, two networks">
          <GroupedBars data={f1Data} series={DATASET_SERIES} height={320} xTickFormatter={(v) => v.replace(" (10 trees)", "")} />
        </ChartCard>
        <ChartCard
          className="lg:col-span-2"
          title="Why is cross-domain precision still high?"
          subtitle="Precision vs the share of malicious records"
        >
          <p className="text-sm leading-relaxed text-slate-600">
            CICIoT2023 is <b className="text-cic">{pct(data.confusion.cic?.malicious_share, 1)}</b> malicious, so picking records
            at random would already give about that precision. Every model's cross-domain precision is{" "}
            <b>below</b> this base rate, so the few flows it flags are no better than chance.
          </p>
          <div className="mt-4 space-y-2.5">
            {algos.map((a) => (
              <div key={a.model}>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-600">{a.model}</span>
                  <span className="font-semibold tabular-nums">{pct(a.cross_domain.precision, 1)}</span>
                </div>
                <div className="relative mt-1 h-2 rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-cic" style={{ width: `${a.cross_domain.precision * 100}%` }} />
                  <div className="absolute top-[-3px] h-[14px] w-0.5 bg-ink" style={{ left: `${data.confusion.cic?.malicious_share * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted">Black marker = base rate (share of malicious records).</p>
        </ChartCard>
      </div>

      <Section title="All four metrics, per algorithm">
        <div className="grid gap-5 md:grid-cols-2">
          {algos.map((a) => (
            <ChartCard key={a.model} title={a.model}>
              <GroupedBars
                height={240}
                series={DATASET_SERIES}
                data={METRICS.map((m) => ({ name: m.label, fiveg: a.in_domain[m.key] * 100, cic: a.cross_domain[m.key] * 100 }))}
              />
            </ChartCard>
          ))}
        </div>
      </Section>

      <Section title="Cost: model size and training time">
        <div className="grid gap-5 md:grid-cols-2">
          <ChartCard title="Saved model size" footer="Logistic Regression is stored as “0.0 MB” in the results file (rounded to 2 decimals), i.e. smaller than 0.01 MB.">
            <HBars data={algos.map((a) => ({ name: a.model, value: a.size_mb }))} color={COLORS.neutral} tickWidth={160} labelFormat={(v) => sizeLabel(v)} unit=" MB" />
          </ChartCard>
          <ChartCard title="Training time" subtitle="Seconds on 972,540 rows">
            <HBars data={algos.map((a) => ({ name: a.model, value: a.train_seconds }))} color={COLORS.neutral} tickWidth={160} labelFormat={(v) => `${v.toFixed(1)} s`} unit=" s" />
          </ChartCard>
        </div>
      </Section>

      <Section title="Full table">
        <Table
          columns={[
            { key: "model", label: "Model", className: "font-semibold" },
            { key: "size", label: "Size", align: "right", render: (a) => sizeLabel(a.size_mb) },
            { key: "train", label: "Train (s)", align: "right", render: (a) => num(a.train_seconds, 2) },
            ...METRICS.map((m) => ({ key: `i${m.key}`, label: `5G ${m.label}`, align: "right", render: (a) => pct(a.in_domain[m.key]) })),
            ...METRICS.map((m) => ({ key: `c${m.key}`, label: `CIC ${m.label}`, align: "right", render: (a) => <span className="text-cic">{pct(a.cross_domain[m.key])}</span> })),
            { key: "drop", label: "F1 drop", align: "right", render: (a) => <b className="text-red-600">{a.f1_drop_pp.toFixed(2)} pp</b> },
          ]}
          rows={algos}
        />
      </Section>

      {lr && (
        <Callout tone="info" title="A note on Logistic Regression">
          Its in-domain recall of {pct(lr.in_domain.recall)} means it labels almost every 5G-NIDD record malicious, so its in-domain F1 is
          close to that of a trivial “always malicious” rule. In this original run it was also trained on unscaled features (fixed in the
          Improved Experiments).
        </Callout>
      )}
    </>
  );
}
