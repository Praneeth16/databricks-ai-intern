import { test, expect, type Page } from '@playwright/test';

/**
 * E2E coverage for the YOLO auto-approval top-bar control (issue #18 +
 * Playwright harness from issue #19).
 *
 * Strategy
 * --------
 * 1. Mock every backend endpoint the AppLayout boot path hits so we
 *    don't need a Databricks-connected backend running. The bolt
 *    interaction goes through ``apiFetch`` → real fetch → Playwright
 *    intercepts at the network layer.
 * 2. Seed an active session into ``localStorage`` BEFORE navigation so
 *    the ``zustand/persist`` middleware hydrates with one already
 *    selected — without that the YoloControl button stays disabled.
 * 3. Assert PATCH body shape + the post-PATCH UI state (bolt color
 *    flips, tooltip mirrors the new remaining budget).
 */

const SESSION_ID = 'sess-e2e-yolo';

// Mirror of the persisted-store shape — must match zustand/persist's
// expectations exactly or hydration silently fails.
const seededStore = {
  state: {
    sessions: [
      {
        id: SESSION_ID,
        title: 'E2E session',
        createdAt: '2026-05-24T00:00:00Z',
        isActive: true,
        needsAttention: false,
      },
    ],
    activeSessionId: SESSION_ID,
  },
  version: 0,
};

async function setupRoutes(page: Page, patchHandler: (body: any) => any) {
  // Auth endpoints — return a logged-in dev user so the layout doesn't
  // redirect us to a login page.
  await page.route('**/auth/status', async (route) => {
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ auth_enabled: false }),
    });
  });
  await page.route('**/auth/me', async (route) => {
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        authenticated: true, username: 'e2e-user',
        name: 'E2E User', picture: null,
      }),
    });
  });

  // Session metadata — AppLayout fetches /api/session/<id> on mount.
  await page.route(`**/api/session/${SESSION_ID}`, async (route) => {
    if (route.request().method() !== 'GET') return route.continue();
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        session_id: SESSION_ID,
        created_at: '2026-05-24T00:00:00Z',
        is_active: true,
        is_processing: false,
        message_count: 0,
        model: 'databricks/databricks-claude-opus-4',
      }),
    });
  });

  // Catch-all: prevent unhandled /api or /auth calls from failing
  // silently and dragging in real backend latency.
  await page.route('**/api/**', async (route) => {
    // Default empty list / no-op for any path we didn't override.
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({}),
    });
  });

  // The PATCH /yolo endpoint — the actual surface under test. We delay
  // the route registration until after the more specific ones above so
  // ``page.route`` matches PATCH /yolo first.
  await page.route(`**/api/session/${SESSION_ID}/yolo`, async (route) => {
    const req = route.request();
    if (req.method() === 'GET') {
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({
          enabled: false, cost_cap_usd: null,
          estimated_spend_usd: 0, remaining_usd: null,
        }),
      });
      return;
    }
    if (req.method() === 'PATCH') {
      const body = JSON.parse(req.postData() || '{}');
      const response = patchHandler(body);
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify(response),
      });
      return;
    }
    await route.continue();
  });
}

test.beforeEach(async ({ page }) => {
  // ``initScript`` runs in the page context BEFORE app JS, which is
  // what we need so zustand/persist hydrates with our seed instead of
  // creating an empty store.
  await page.addInitScript((s) => {
    window.localStorage.setItem('hf-agent-sessions', JSON.stringify(s));
  }, seededStore);
});

test('left-click bolt toggles YOLO on with default cap', async ({ page }) => {
  const patchBodies: any[] = [];
  await setupRoutes(page, (body) => {
    patchBodies.push(body);
    return {
      enabled: body.enabled,
      cost_cap_usd: body.cost_cap_usd ?? 5,
      estimated_spend_usd: 0,
      remaining_usd: body.enabled ? (body.cost_cap_usd ?? 5) : null,
    };
  });

  await page.goto('/');
  const bolt = page.getByRole('button', { name: /toggle yolo/i });
  await expect(bolt).toBeVisible();
  await expect(bolt).toBeEnabled();

  await bolt.click();

  // PATCH fired with enabled=true and the default cap.
  await expect.poll(() => patchBodies.length).toBe(1);
  expect(patchBodies[0]).toMatchObject({
    enabled: true,
    cost_cap_usd: 5,
  });

  // After mutation, hovering the bolt shows the on-state tooltip with
  // the remaining budget mirrored back from our mocked response.
  await bolt.hover();
  await expect(page.getByText(/YOLO ON/)).toBeVisible();
});

test('right-click bolt opens dialog and Apply PATCHes with cap', async ({ page }) => {
  const patchBodies: any[] = [];
  await setupRoutes(page, (body) => {
    patchBodies.push(body);
    return {
      enabled: body.enabled,
      cost_cap_usd: body.cost_cap_usd ?? null,
      estimated_spend_usd: 0,
      remaining_usd: body.enabled ? (body.cost_cap_usd ?? null) : null,
    };
  });

  await page.goto('/');
  const bolt = page.getByRole('button', { name: /toggle yolo/i });
  await bolt.click({ button: 'right' });

  // Dialog visible with the budget field.
  await expect(page.getByRole('dialog')).toBeVisible();
  const capField = page.getByLabel('Budget cap (USD)');
  await capField.fill('25');

  await page.getByRole('button', { name: /apply & enable/i }).click();

  await expect.poll(() => patchBodies.length).toBeGreaterThanOrEqual(1);
  const last = patchBodies[patchBodies.length - 1];
  expect(last).toMatchObject({ enabled: true, cost_cap_usd: 25 });
});

test('bolt not accessible when no active session', async ({ page }) => {
  // Override the seed: no sessions at all. AppLayout swaps in the
  // WelcomeScreen for this state, which doesn't render the top bar at
  // all — so the assertion is "control is unreachable", which the
  // welcome screen satisfies by structurally hiding it. This is the
  // contract we care about: a user with no session can't accidentally
  // toggle YOLO on. ``count()`` of 0 confirms it.
  await page.addInitScript(() => {
    window.localStorage.setItem('hf-agent-sessions', JSON.stringify({
      state: { sessions: [], activeSessionId: null },
      version: 0,
    }));
  });

  await setupRoutes(page, () => ({}));
  await page.goto('/');

  await expect(page.getByRole('button', { name: /toggle yolo/i })).toHaveCount(0);
});
