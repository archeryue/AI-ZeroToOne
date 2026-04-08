import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: 0,
  use: {
    baseURL: "http://localhost:3000",
    // Use pre-installed Chromium
    launchOptions: {
      executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    },
    headless: true,
  },
  webServer: [
    {
      // Backend (FastAPI)
      command: "cd ../backend && python -m uvicorn main:app --host 0.0.0.0 --port 8001",
      port: 8001,
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      // Frontend (Next.js)
      command: "npm run dev -- -p 3000",
      port: 3000,
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
