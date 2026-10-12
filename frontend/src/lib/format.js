export const COLORS = {
  fiveg: "#2a78d6",
  cic: "#eb6834",
  neutral: "#94a3b8",
  benign: "#10b981",
  malicious: "#ef4444",
  // metric colours
  accuracy: "#4a3aa7",
  precision: "#1baf7a",
  recall: "#e87ba4",
  f1: "#eda100",
};

export const METRICS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
];

export function pct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function compact(value) {
  if (value === null || value === undefined) return "–";
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(abs >= 1e4 ? 0 : 1)}k`;
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

export function sizeLabel(mb) {
  if (mb === null || mb === undefined) return "–";
  // other_models_comparison.csv rounds to 2 decimals: 0.0 means < 0.01 MB
  if (mb < 0.01) return "< 0.01 MB";
  return `${mb.toFixed(2)} MB`;
}

export function kb(value) {
  if (value === null || value === undefined) return "–";
  if (value >= 1024) return `${(value / 1024).toFixed(2)} MB`;
  return `${value.toFixed(1)} KB`;
}

export function pp(value, digits = 2) {
  if (value === null || value === undefined) return "–";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)} pp`;
}

// Short model names
export function shortModel(name) {
  return name
    .replace("Random Forest", "RF")
    .replace("Logistic Regression", "Log. Reg.")
    .replace("Decision Tree", "Dec. Tree");
}

// Shorter model names for chart axes
export function axisModel(name) {
  if (name.includes("depth")) return "RF d≤12";
  if (name.startsWith("Random Forest")) return "RF";
  if (name.startsWith("Decision")) return "Dec. Tree";
  if (name.startsWith("Logistic")) return "Log. Reg.";
  return name;
}
