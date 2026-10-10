import { useEffect, useState } from "react";
import {
  Activity,
  ArrowLeftRight,
  BarChart3,
  Crosshair,
  FlaskConical,
  LayoutDashboard,
  Menu,
  RadioTower,
  ScanSearch,
  Shuffle,
  TreePine,
  X,
} from "lucide-react";
import { api } from "./lib/api";
import Overview from "./pages/Overview";
import ModelPerformance from "./pages/ModelPerformance";
import Algorithms from "./pages/Algorithms";
import CrossDomain from "./pages/CrossDomain";
import Attacks from "./pages/Attacks";
import FeatureShift from "./pages/FeatureShift";
import Improved from "./pages/Improved";
import EdgeMonitor from "./pages/EdgeMonitor";
import PredictionDemo from "./pages/PredictionDemo";
import LiveMonitor from "./pages/LiveMonitor";

const NAV = [
  {
    group: "Research",
    items: [
      { id: "overview", label: "Overview", icon: LayoutDashboard, page: Overview },
      { id: "models", label: "Model Size & Trees", icon: TreePine, page: ModelPerformance },
      { id: "algorithms", label: "Algorithm Comparison", icon: BarChart3, page: Algorithms },
      { id: "cross-domain", label: "Cross-Domain Evaluation", icon: ArrowLeftRight, page: CrossDomain },
      { id: "attacks", label: "Attack Analysis", icon: Crosshair, page: Attacks },
      { id: "feature-shift", label: "Feature & Protocol Shift", icon: Shuffle, page: FeatureShift },
      { id: "improved", label: "Improved Experiments", icon: FlaskConical, page: Improved },
    ],
  },
  {
    group: "Application",
    items: [
      { id: "edge-monitor", label: "5G Edge IDS Monitor", icon: RadioTower, page: EdgeMonitor },
      { id: "live", label: "Live IDS Monitoring", icon: Activity, page: LiveMonitor },
      { id: "predict", label: "Prediction Demo", icon: ScanSearch, page: PredictionDemo },
    ],
  },
];

const ALL = NAV.flatMap((g) => g.items);

function usePageFromHash() {
  const read = () => {
    const id = window.location.hash.replace("#/", "").replace("#", "");
    return ALL.some((p) => p.id === id) ? id : "overview";
  };
  const [page, setPage] = useState(read);
  useEffect(() => {
    const onChange = () => {
      setPage(read());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return page;
}

function Sidebar({ page, onNavigate }) {
  return (
    <div className="flex h-full flex-col bg-ink text-slate-300">
      <div className="px-5 pb-5 pt-6">
        <div className="flex items-center gap-3">
          <img src="/favicon.svg" alt="" className="h-9 w-9" />
          <div>
            <div className="text-sm font-bold leading-tight text-white">5G IDS Lab</div>
            <div className="text-[11px] text-slate-400">CS427 Mobile Communications</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-3">
        {NAV.map((group) => (
          <div key={group.group} className="mb-5">
            <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              {group.group}
            </div>
            {group.items.map((item) => {
              const Icon = item.icon;
              const active = item.id === page;
              return (
                <a
                  key={item.id}
                  href={`#/${item.id}`}
                  onClick={onNavigate}
                  className={`mb-0.5 flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                    active ? "bg-white/10 font-semibold text-white" : "hover:bg-white/5 hover:text-white"
                  }`}
                >
                  <Icon className={`h-4 w-4 ${active ? "text-[#5aa2f0]" : "text-slate-500"}`} />
                  {item.label}
                </a>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="border-t border-white/10 px-5 py-4 text-[11px] leading-relaxed text-slate-400">
        <div className="mb-2 flex flex-col gap-1">
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-sm bg-fiveg" /> 5G-NIDD · train &amp; in-domain
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-sm bg-cic" /> CICIoT2023 · cross-domain
          </span>
        </div>
        Samuela Robin &amp; Katea Yavunitu
      </div>
    </div>
  );
}

export default function App() {
  const page = usePageFromHash();
  const [data, setData] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    Promise.all([api.results(), api.status()])
      .then(([results, s]) => {
        setData(results);
        setStatus(s);
      })
      .catch((e) => setError(e.message));
  }, []);

  const current = ALL.find((p) => p.id === page);
  const Page = current.page;

  return (
    <div className="flex min-h-full">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 hidden w-64 lg:block">
        <Sidebar page={page} />
      </aside>

      {/* Mobile top bar + drawer */}
      <div className="fixed inset-x-0 top-0 z-30 flex items-center justify-between bg-ink px-4 py-3 lg:hidden">
        <div className="flex items-center gap-2 text-sm font-bold text-white">
          <img src="/favicon.svg" alt="" className="h-7 w-7" /> 5G IDS Lab
        </div>
        <button onClick={() => setMenuOpen(true)} className="text-white" aria-label="Open menu">
          <Menu className="h-6 w-6" />
        </button>
      </div>
      {menuOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setMenuOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-72">
            <button onClick={() => setMenuOpen(false)} className="absolute right-3 top-5 z-10 text-slate-400" aria-label="Close menu">
              <X className="h-5 w-5" />
            </button>
            <Sidebar page={page} onNavigate={() => setMenuOpen(false)} />
          </div>
        </div>
      )}

      <main className="min-w-0 flex-1 lg:pl-64">
        <div className="mx-auto max-w-7xl px-4 pb-16 pt-20 sm:px-8 lg:pt-10">
          {error && (
            <div className="rounded-2xl border border-red-200 bg-red-50 p-6 text-sm text-red-900">
              <div className="font-semibold">Could not reach the API.</div>
              <div className="mt-1">{error}</div>
              <div className="mt-3 text-red-800">
                Start the backend from the project root with{" "}
                <code className="rounded bg-red-100 px-1.5 py-0.5">uvicorn api.main:app --port 8000</code>
              </div>
            </div>
          )}
          {!error && !data && (
            <div className="flex h-64 items-center justify-center text-sm text-muted">Loading results…</div>
          )}
          {data && (
            <div key={page} className="fade-in">
              <Page data={data} status={status} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
