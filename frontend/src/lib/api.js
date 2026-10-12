async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const message = body?.detail
      ? typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
      : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}

export const api = {
  results: () => request("/api/results"),
  status: () => request("/api/status"),
  ranges: () => request("/api/predict/ranges"),
  predict: (payload) => request("/api/predict", { method: "POST", body: JSON.stringify(payload) }),
  stream: (dataset, n) => request(`/api/stream?dataset=${encodeURIComponent(dataset)}&n=${n}`),
  liveInterfaces: () => request("/api/live/interfaces"),
  liveStart: (payload) => request("/api/live/start", { method: "POST", body: JSON.stringify(payload) }),
  liveStop: () => request("/api/live/stop", { method: "POST" }),
  liveClear: () => request("/api/live/clear", { method: "POST" }),
  liveState: () => request("/api/live/state"),
};
