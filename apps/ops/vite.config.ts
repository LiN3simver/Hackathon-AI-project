import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    proxy: {
      "/api": loadEnv(mode, "../..", "GUARDIAN_").GUARDIAN_API_PROXY_TARGET ?? "http://127.0.0.1:8000",
    },
  },
}));
