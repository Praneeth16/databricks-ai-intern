# frontend

Vite + React + TypeScript + MUI. Talks to the FastAPI backend at
`/api/*` and the Apps proxy at `/auth/*`.

## Dev

```bash
npm install
npm run dev       # :5173
npm run build     # tsc -b && vite build → frontend/dist
npm run lint
```

## End-to-end tests

Playwright + Chromium. Backend is mocked at the network layer
(`page.route('**/api/**', ...)`), so the suite runs offline — no
Databricks workspace creds, no FastAPI process required.

```bash
npm run test:e2e         # runs ./tests/e2e/*.spec.ts headless
npx playwright test --ui # interactive runner
npx playwright test --headed --debug   # pause on first action
```

First-time setup needs the browser binary:

```bash
npx playwright install chromium
```

### Adding a new spec

Mirror `tests/e2e/yolo-control.spec.ts`:

1. Seed the persisted store via `page.addInitScript(...)` BEFORE
   `page.goto('/')` so zustand hydrates with your fixture.
2. Mock every `/api/*` endpoint the component hits with `page.route()`.
   The catch-all (`**/api/**` returning `{}`) keeps unhandled paths
   from breaking the boot.
3. Assert on user-visible state (roles + text) rather than CSS
   selectors so the tests survive style refactors.

CI gate intentionally NOT wired yet — local-only until the suite has
been green for a week of normal dev. Track that follow-up under #19.
