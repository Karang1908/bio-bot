import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the Python studio server (uvicorn on :8765).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 2000 },
});
