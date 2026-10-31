import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig(({ mode }) => ({
  // Phones only allow camera access on HTTPS; a local self-signed certificate is enough for LAN demos.
  plugins: [react(), basicSsl()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": loadEnv(mode, "../..", "GUARDIAN_").GUARDIAN_API_PROXY_TARGET ?? "http://127.0.0.1:8000" },
  },
}));
