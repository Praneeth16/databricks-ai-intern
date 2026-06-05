import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E config (issue #19).
 *
 * Pure frontend tests — backend ``/api/*`` calls are mocked via
 * ``page.route()`` in each spec, so no Databricks workspace creds and
 * no backend process are required to run the suite.
 *
 * The ``webServer`` block auto-starts ``vite dev`` before the run and
 * tears it down afterwards. Set ``E2E_VITE_PORT`` to override (e.g. CI
 * already has port 5173 occupied).
 */
const VITE_PORT = Number(process.env.E2E_VITE_PORT ?? 5173);

export default defineConfig({
  testDir: './tests/e2e',
  // Local-only for now — no CI gate yet (filed as follow-up). Keep
  // retries low so a flake doesn't hide a real regression.
  retries: 0,
  // One worker keeps the localStorage seeding deterministic per test
  // until we add per-test storage isolation.
  workers: 1,
  reporter: 'list',
  timeout: 30_000,
  use: {
    baseURL: `http://localhost:${VITE_PORT}`,
    trace: 'on-first-retry',
    headless: true,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: `http://localhost:${VITE_PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
