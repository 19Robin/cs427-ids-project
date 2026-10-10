import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// During development (npm run dev) API calls are forwarded to the FastAPI
// server started with: uvicorn api.main:app --port 8000
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { chunkSizeWarningLimit: 1000 },
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
