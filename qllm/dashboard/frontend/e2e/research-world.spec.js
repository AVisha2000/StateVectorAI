import { test, expect } from "@playwright/test";
import { mockApi } from "./fixtures.js";
import { RESEARCH_WORLD_SNAPSHOT } from "../src/lib/researchWorld.js";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test("the world keeps illustrative scenes separate from snapshot agents", async ({
  page,
}) => {
  const external = [];
  page.on("request", (request) => {
    if (!request.url().startsWith("http://localhost:4174/"))
      external.push(request.url());
  });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A world of curiosity." }),
  ).toBeVisible();
  await expect(
    page.getByText(/^Interactive preview/),
  ).toBeVisible();
  await expect(page.locator(".rw-studio-card")).toHaveCount(7);
  await expect(page.locator(".rw-researcher-list button")).toHaveCount(6);
  await expect(
    page.getByText(/6 sample agents.*No live research connected/),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "AI safety Testing a robot", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Sage", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Concept studio", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Open workspace" }),
  ).toHaveAttribute("href", "/portal");
  await expect.poll(() => external).toEqual([]);
});

test("each researcher opens their own brief, including the second agent in a discipline", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /NE Nova Ember/ }).click();
  await expect(
    page.getByRole("heading", { name: "Nova Ember", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Check model assumptions" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Meet Nova", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Coffee with Nova." }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Meet Nova", exact: true }),
  ).toBeFocused();
});

test("camera controls change both axes, pause motion, and restore the initial view", async ({
  page,
}) => {
  await page.goto("/");
  const canvas = page.locator(".planet-canvas");
  await page.getByRole("button", { name: "Pause motion", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-agent-motion", "paused");
  const before = await canvas.getAttribute("data-camera");
  await page.getByRole("button", { name: "Rotate globe up" }).click();
  await expect
    .poll(async () => (await canvas.getAttribute("data-camera")).split(",")[1])
    .not.toBe(before.split(",")[1]);
  await page.getByRole("button", { name: "Rotate globe left" }).click();
  await expect
    .poll(async () => (await canvas.getAttribute("data-camera")).split(",")[0])
    .not.toBe(before.split(",")[0]);
  await page.getByRole("button", { name: "Reset view" }).click();
  await expect(canvas).toHaveAttribute("data-camera", "0.00,0.65,11.90");
});

test("meeting supports drafts, shared reading, local calculation and visitor movement", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await page
    .getByLabel("Your question", { exact: true })
    .fill("Keep this question for our next visit");
  await page.getByRole("button", { name: "Close research session" }).click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await expect(page.getByLabel("Your question", { exact: true })).toHaveValue(
    "Keep this question for our next visit",
  );
  await page
    .getByRole("button", { name: "Enter first-person", exact: true })
    .click();
  const roomCanvas = page.locator(".meeting-game-canvas");
  const before = await roomCanvas.getAttribute("data-visitor");
  await roomCanvas.locator("canvas").press("w");
  await expect
    .poll(() => roomCanvas.getAttribute("data-visitor"))
    .not.toBe(before);
  const moved = await roomCanvas.getAttribute("data-visitor");
  await page
    .getByLabel("Your question", { exact: true })
    .fill("wasd e t are text while typing");
  await expect(roomCanvas).toHaveAttribute("data-visitor", moved);
  await expect(page.getByLabel("Your question", { exact: true })).toHaveValue(
    "wasd e t are text while typing",
  );
  await page
    .getByLabel("Your question", { exact: true })
    .fill("Keep this question for our next visit");
  await page.getByRole("button", { name: "Show me a working note" }).click();
  await expect(page.getByLabel("Your question", { exact: true })).toHaveValue(
    "Keep this question for our next visit",
  );
  await expect(
    page.getByText(
      "Illustrative working note. This is not a published paper or a research result.",
    ),
  ).toBeVisible();
  await expect(page.getByLabel("Research conversation")).toBeVisible();
  await page
    .getByRole("button", { name: "What if we tried…", exact: true })
    .click();
  await expect(page.getByLabel("Your idea")).toHaveValue(
    "What if we tried a different approach?",
  );
  await page.getByText("Keep a research idea", { exact: true }).click();
  await page.getByRole("button", { name: "Save experiment draft" }).click();
  await expect(
    page.getByText(
      "Draft saved for this visit. No experiment has been started.",
    ),
  ).toBeVisible();
  await page.getByRole("button", { name: "Launch rocket" }).click();
  await page
    .getByText("Model, calculation and limits", { exact: true })
    .click();
  await expect(
    page.getByText(/121 sample values computed locally/),
  ).toBeVisible();
});

test("a carried note survives movement, reading, typing, return and studio re-entry", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await page
    .getByLabel("Room interactions")
    .getByRole("button", { name: "Take working note", exact: true })
    .click();
  const room = page.locator(".meeting-game-canvas"),
    canvas = room.locator("canvas");
  await expect(room).toHaveAttribute("data-note-held", "true");
  await expect(page.getByLabel("Working note reader")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Enter first-person", exact: true })
    .click();
  await canvas.press("q");
  await expect(
    page.getByText("Move closer to the coffee table to put the note down."),
  ).toBeVisible();
  await canvas.press("r");
  await expect(page.getByLabel("Working note reader")).toBeFocused();
  const visitor = await room.getAttribute("data-visitor");
  await page
    .getByLabel("Your question", { exact: true })
    .fill("r q wasd remain ordinary letters");
  await expect(room).toHaveAttribute("data-note-held", "true");
  await expect(room).toHaveAttribute("data-visitor", visitor);
  await page.getByRole("button", { name: "Lower note R", exact: true }).click();
  await canvas.press("q");
  for (let i = 0; i < 6; i++)
    await page
      .getByRole("button", { name: "Walk forward", exact: true })
      .click();
  await expect(
    page.getByRole("button", { name: "Put note on table" }),
  ).toBeEnabled();
  await expect(
    page.getByText("Move closer to the coffee table to put the note down."),
  ).toHaveCount(0);
  await canvas.press("q");
  await expect(room).toHaveAttribute("data-note-held", "false");
  await expect(room).toHaveAttribute("data-note-settled", "true");
  await expect(room).toHaveAttribute("data-note-position", "0.000,0.473,0.370");
  await page.getByLabel("Room controls", { exact: true }).click();
  await page
    .getByLabel("Room interactions")
    .getByRole("button", { name: "Take working note", exact: true })
    .click();
  await page.getByRole("button", { name: "Leave studio", exact: true }).click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await expect(room).toHaveAttribute("data-note-held", "true");
  await expect(page.getByLabel("Your question", { exact: true })).toHaveValue(
    "r q wasd remain ordinary letters",
  );
});

test("equipment inspection restores overview and never teleports a walking visitor", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  const room = page.locator(".meeting-game-canvas");
  await expect(room).toHaveAttribute("data-camera-owner", "overview-orbit");
  const before = await room.getAttribute("data-camera");
  const open = async () => {
    const controls = page.getByLabel("Room controls", { exact: true });
    if (await controls.count()) await controls.click();
    await page
      .getByLabel("Room interactions", { exact: true })
      .getByRole("button", { name: "Try an experiment", exact: true })
      .click();
  };
  await open();
  await expect(room).toHaveAttribute("data-camera-owner", "inspection-orbit");
  await expect(
    page.getByRole("button", { name: "Show conversation", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("slider", { name: "Launch angle", exact: true }),
  ).toBeFocused();
  await page
    .getByRole("button", { name: "Put material away", exact: true })
    .click();
  await expect(room).toHaveAttribute("data-camera-owner", "overview-orbit");
  await expect(room).toHaveAttribute("data-camera", before);
  await page
    .getByRole("button", { name: "Enter first-person", exact: true })
    .click();
  const prior = await room.getAttribute("data-visitor");
  await room.locator("canvas").press("w");
  await expect.poll(() => room.getAttribute("data-visitor")).not.toBe(prior);
  const walked = await room.getAttribute("data-visitor");
  await open();
  await expect(room).toHaveAttribute("data-camera-owner", "pointer-look");
  await expect(room).toHaveAttribute("data-visitor", walked);
});

test("unavailable or malformed snapshots retain a fixture with a manual retry", async ({
  page,
}) => {
  let requests = 0;
  await page.route("**/api/research-world", (route) => {
    requests += 1;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(requests === 1 ? {} : RESEARCH_WORLD_SNAPSHOT),
    });
  });
  await page.goto("/");
  await expect(
    page.getByText("Snapshot unavailable · showing sample data."),
  ).toBeVisible();
  await page.waitForTimeout(150);
  expect(requests).toBe(1);
  await page.getByRole("button", { name: "Retry snapshot" }).click();
  await expect(
    page.getByText(/6 sample agents.*No live research connected/),
  ).toBeVisible();
  expect(requests).toBe(2);
});

test("shared board retains words and ink and quoted prompts do not erase a draft", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await page
    .getByLabel("Your question", { exact: true })
    .fill("Keep this thought");
  await page
    .getByRole("button", { name: "Let’s use the board", exact: true })
    .click();
  await page
    .getByLabel("The assumption", { exact: true })
    .fill("Ignore air resistance");
  const sketch = page.locator(".shared-sketch > svg");
  await sketch.scrollIntoViewIfNeeded();
  const rect = await sketch.boundingBox();
  await page.mouse.move(rect.x + 20, rect.y + 25);
  await page.mouse.down();
  await page.mouse.move(rect.x + 160, rect.y + 100, { steps: 10 });
  await page.mouse.up();
  await expect(sketch.locator("polyline")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Close research session", exact: true })
    .click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await expect(page.getByLabel("The assumption", { exact: true })).toHaveValue(
    "Ignore air resistance",
  );
  await expect(page.locator(".shared-sketch polyline")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Show me a working note", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Ask about this passage", exact: true })
    .nth(1)
    .click();
  await page
    .getByRole("button", { name: "Discuss this passage", exact: true })
    .click();
  await expect(page.getByLabel("Your question", { exact: true })).toHaveValue(
    "Keep this thought",
  );
  await expect(page.getByLabel("Research conversation")).toContainText(
    "Which control stays fixed?",
  );
});

test("denied pointer lock retains first-person keyboard and drag controls", async ({
  page,
}) => {
  await page.addInitScript(() => {
    HTMLCanvasElement.prototype.requestPointerLock = () =>
      Promise.reject(new Error("test denied"));
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await page
    .getByRole("button", { name: "Enter first-person", exact: true })
    .click();
  await expect(
    page.getByText(
      "Mouse capture was not allowed. Drag to look; WASD still moves.",
    ),
  ).toBeVisible();
  const host = page.locator(".meeting-game-canvas");
  await host.locator("canvas").press("w");
  await expect(host).not.toHaveAttribute("data-visitor", "0.83,1.65");
  await host.locator("canvas").press("t");
  await expect(page.getByLabel("Your question", { exact: true })).toBeFocused();
  const position = await host.getAttribute("data-visitor");
  await page
    .getByLabel("Your question", { exact: true })
    .pressSequentially("wasd e t");
  await expect(host).toHaveAttribute("data-visitor", position);
  await expect(page.getByLabel("Your question", { exact: true })).toHaveValue(
    "wasd e t",
  );
});

test("server content replaces bundled agent content", async ({ page }) => {
  const snapshot = {
    ...RESEARCH_WORLD_SNAPSHOT,
    agents: RESEARCH_WORLD_SNAPSHOT.agents.map((agent, i) =>
      i === 0 ? { ...agent, title: "Endpoint snapshot title" } : agent,
    ),
  };
  await page.route("**/api/research-world", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(snapshot),
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: /LV Lyra Vale/ }).click();
  await expect(
    page.getByRole("heading", { name: "Endpoint snapshot title" }),
  ).toBeVisible();
});

test("launch bench keeps measured paths separate from aim and compares prior flights", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await page
    .getByRole("button", { name: "What if we tried…", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Pause room motion", exact: true })
    .click();
  await page.getByRole("button", { name: "30°", exact: true }).click();
  await page
    .getByRole("button", { name: "Launch rocket", exact: true })
    .click();
  await expect(
    page.getByText("Landed at 35.3 m", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "60°", exact: true }).click();
  await expect(page.locator(".launch-result")).toContainText(
    "Aim is now 60°; launch to apply.",
  );
  await page
    .getByRole("button", { name: "Launch rocket", exact: true })
    .click();
  await expect(page.locator(".launch-discovery")).toContainText(
    "Same landing as 30°.",
  );
  await expect(page.locator(".launch-discovery")).toContainText("10.2");
  await expect(page.locator(".meeting-game-canvas")).toHaveAttribute(
    "data-flight-progress",
    "1",
  );
  await page
    .getByRole("slider", { name: "Launch angle", exact: true })
    .press("ArrowRight");
  await expect(page.locator(".launch-result")).toContainText(
    "Aim is now 61°; launch to apply.",
  );
});

for (const bench of [
  {
    studio: "Chemistry At the reaction bench",
    agent: "Iris",
    region: "Hands-on reaction bench",
    run: "Run reaction",
    difference: "8.0 percentage points less reactant.",
  },
  {
    studio: "Biology Exploring living systems",
    agent: "Fern",
    region: "Hands-on growth bench",
    run: "Grow colonies",
    difference: "40.3 percentage points more population.",
  },
])
  test(`${bench.region} shares a selected worker sample and preserves drafts`, async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByRole("button", { name: bench.studio, exact: true }).click();
    await page
      .getByRole("button", { name: `Meet ${bench.agent}`, exact: true })
      .click();
    await page
      .getByLabel("Room interactions", { exact: true })
      .getByRole("button", { name: "Try an experiment", exact: true })
      .click();
    await page.getByRole("button", { name: "Fast 1.2", exact: true }).click();
    await page.getByRole("button", { name: bench.run, exact: true }).click();
    const surface = page.getByRole("region", {
      name: bench.region,
      exact: true,
    });
    await expect(surface).toHaveAttribute("data-sample", "60");
    await expect(surface).toContainText(bench.difference);
    await page.getByRole("button", { name: "Slow 0.2", exact: true }).click();
    await expect(surface).toContainText(
      "Rate 0.2 is selected; run again to apply it.",
    );
    await expect(surface).toContainText(bench.difference);
    await page
      .getByRole("slider", { name: "Model time", exact: true })
      .press("Home");
    await page
      .getByRole("slider", { name: "Model time", exact: true })
      .press("ArrowRight");
    await expect(surface).toHaveAttribute("data-sample", "1");
    await expect(page.locator(".meeting-game-canvas")).toHaveAttribute(
      "data-kinetic-sample",
      "1",
    );
    await page
      .getByRole("button", { name: "Show conversation", exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Your question", exact: true })
      .fill("Keep this unfinished thought");
    await page
      .getByRole("button", { name: "Discuss this moment", exact: true })
      .click();
    await expect(page.locator(".meeting-quote-context")).toContainText(
      "t=0.08",
    );
    await expect(
      page.getByRole("textbox", { name: "Your question", exact: true }),
    ).toHaveValue("Keep this unfinished thought");
    await page
      .getByRole("button", { name: "Discuss this observation", exact: true })
      .click();
    await expect(
      page.getByRole("log", { name: "Research conversation", exact: true }),
    ).toContainText("scripted prompt about the browser toy");
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await page
      .getByRole("button", { name: "Pause room motion", exact: true })
      .click();
    await page.getByRole("button", { name: "Play time", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Room paused", exact: true }),
    ).toBeVisible();
    await expect(surface).toHaveAttribute("data-sample", "0");
    await page
      .getByRole("button", { name: "Resume room motion", exact: true })
      .click();
    await expect(surface).toHaveAttribute("data-sample", "120", {
      timeout: 10000,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    const layout = await page.evaluate(() => ({
      width: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    expect(layout.scroll).toBe(layout.width);
  });

test("collected results survive another run, pinning, typing and studio return", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", {
      name: "Chemistry At the reaction bench",
      exact: true,
    })
    .click();
  await page.getByRole("button", { name: "Meet Iris", exact: true }).click();
  const actions = page.getByLabel("Room interactions", { exact: true });
  await actions
    .getByRole("button", { name: "Try an experiment", exact: true })
    .click();
  await page.getByRole("button", { name: "Fast 1.2", exact: true }).click();
  await page.getByRole("button", { name: "Run reaction", exact: true }).click();
  await expect(
    page.getByRole("slider", { name: "Model time", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("slider", { name: "Model time", exact: true })
    .press("Home");
  await page
    .getByRole("slider", { name: "Model time", exact: true })
    .press("ArrowRight");
  await page
    .getByRole("button", { name: "Take result card", exact: true })
    .click();
  const canvas = page.getByLabel(/^Research room game/);
  await expect(canvas).toBeFocused();
  await canvas.press("r");
  const first = page.getByRole("article", {
    name: "Result card 1",
    exact: true,
  });
  await expect(first).toContainText("90.5%");
  await expect(first).toContainText("Time / seconds: 0.08");
  const draft = page.getByRole("textbox", {
    name: "Your question",
    exact: true,
  });
  await draft.fill("rq keep this unfinished question");
  await draft.press("r");
  await draft.press("q");
  await expect(first).toBeVisible();
  await first
    .getByRole("button", { name: "Discuss this result", exact: true })
    .click();
  await expect(page.locator(".meeting-quote-context")).toContainText("90.5%");
  await expect(draft).toHaveValue("rq keep this unfinished questionrq");
  await page.getByLabel("Collected result reader", { exact: true }).press("r");
  await actions
    .getByRole("button", { name: "Pin result to board", exact: true })
    .click();
  await expect(first).toContainText("Pinned to board");
  await expect(page.locator("[data-pinned-results]")).toHaveAttribute(
    "data-pinned-results",
    "1",
  );
  await actions
    .getByRole("button", { name: "Try an experiment", exact: true })
    .click();
  await page.getByRole("button", { name: "Slow 0.2", exact: true }).click();
  await page.getByRole("button", { name: "Run reaction", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Take result card", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Take result card", exact: true })
    .click();
  await actions
    .getByRole("button", { name: "Pin result to board", exact: true })
    .click();
  await expect(
    page.getByRole("article", { name: "Result card 2", exact: true }),
  ).toContainText("Rate constant: 0.2");
  await expect(first).toContainText("90.5%");
  await page.getByRole("button", { name: "Leave studio", exact: true }).click();
  await page.getByRole("button", { name: "Meet Iris", exact: true }).click();
  await expect(first).toContainText("Pinned to board");
  await expect(
    page.getByRole("article", { name: "Result card 2", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  const layout = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(layout.scroll).toBe(layout.width);
});

test("room launcher separates direct aim from a fired result and restores control", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("/");
  await page
    .getByRole("button", {
      name: "Physics Modelling a trajectory",
      exact: true,
    })
    .click();
  await page.getByRole("button", { name: "Meet Lyra", exact: true }).click();
  await page.getByRole("button", { name: "Use launcher", exact: true }).click();
  const room = page.locator("[data-camera-owner]");
  await expect(room).toHaveAttribute("data-camera-owner", "launcher-aim");
  const camera = await room.getAttribute("data-camera");
  const canvas = page.getByLabel(/^Launcher aiming/);
  const box = await canvas.boundingBox();
  const x = box.x + box.width * 0.65;
  const y = box.y + box.height * 0.45;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x, y + 60, { steps: 6 });
  await page.mouse.up();
  await expect(
    page.getByLabel("Room launch angle 30 degrees", { exact: true }),
  ).toBeVisible();
  await expect(room).toHaveAttribute("data-camera", camera);
  await canvas.press("Space");
  await canvas.press("Shift+ArrowUp");
  await expect(
    page.getByLabel("Room launch angle 35 degrees", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "In-room launcher controls" }),
  ).toContainText("30° · 35.31 m");
  await page
    .getByRole("button", { name: "Take result card", exact: true })
    .click();
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
  await page.getByLabel(/^Research room game/).press("r");
  await expect(
    page.getByRole("article", { name: "Result card 1", exact: true }),
  ).toContainText("35.31 m");
  const draft = page.getByRole("textbox", {
    name: "Your question",
    exact: true,
  });
  await draft.fill("Keep my unfinished thought");
  await page.getByRole("button", { name: "Use launcher", exact: true }).click();
  await page.getByLabel(/^Launcher aiming/).press("Escape");
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
  await page
    .getByRole("button", { name: "Show conversation", exact: true })
    .click();
  await expect(draft).toHaveValue("Keep my unfinished thought");
  await page.getByRole("button", { name: "Use launcher", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .getByRole("button", { name: "Raise launch angle", exact: true })
    .click();
  await expect(
    page.getByLabel("Room launch angle 36 degrees", { exact: true }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});

test("live snapshots never add fictional workers", async ({ page }) => {
  await page.route("**/api/research-world", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...RESEARCH_WORLD_SNAPSHOT, mode: "live" }),
    }),
  );
  await page.goto("/");
  await expect(page.locator(".rw-studio-card")).toHaveCount(4);
  await expect(page.locator(".rw-researcher-list button")).toHaveCount(6);
  await expect(page.getByText("Concept studio", { exact: true })).toHaveCount(
    0,
  );
});

test("narrow selection has no page overflow or controls obscuring the brief", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page
    .getByRole("button", {
      name: "Physics Modelling a trajectory",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("button", { name: "Meet Lyra", exact: true }),
  ).toBeVisible();
  const layout = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    width: document.documentElement.clientWidth,
    note: document
      .querySelector(".rw-field-note-selected")
      .getBoundingClientRect().bottom,
    controls: document.querySelector(".rw-world-bottom").getBoundingClientRect()
      .top,
  }));
  expect(layout.scroll).toBe(layout.width);
  expect(layout.controls).toBeGreaterThan(layout.note);
  await page.getByLabel("Search research studios").fill("no-such-studio");
  await expect(page.getByText(/No studios match/)).toBeVisible();
});

test("reduced motion freezes the world without disabling interaction", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".planet-canvas")).toHaveAttribute(
    "data-agent-motion",
    "paused",
  );
  await expect(page.getByText(/Reduced motion/)).toBeVisible();
  await page
    .getByRole("button", {
      name: "Physics Modelling a trajectory",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("heading", { name: "Lyra Vale", exact: true }),
  ).toBeVisible();
});

test("blocked scene code leaves every researcher accessible", async ({
  page,
}) => {
  await page.route("**/assets/ResearchPlanet-*.js", (route) => route.abort());
  await page.goto("/");
  await expect(
    page.getByText("The 3D view is unavailable. Explore every studio below."),
  ).toBeVisible();
  await page.getByRole("button", { name: /ER Echo Rowan/ }).click();
  await expect(
    page.getByRole("heading", { name: "Echo Rowan", exact: true }),
  ).toBeVisible();
});

test("WebGL initialization failure retains the text interface", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      return String(type).startsWith("webgl")
        ? null
        : original.call(this, type, ...args);
    };
  });
  await page.goto("/");
  await expect(
    page.getByText(
      "The 3D view is unavailable on this device. Every studio is still accessible from the list below.",
    ),
  ).toBeVisible();
  await expect(page.locator(".rw-researcher-list button")).toHaveCount(6);
});

test("portal links and direct refresh stay on the existing portal routes", async ({
  page,
}) => {
  await page.goto("/portal");
  await expect(page.getByRole("link", { name: "Workboard" })).toBeVisible();
  await page.goto("/portal/decisions/decision-controls");
  await expect(
    page.getByRole("heading", { name: "Human decision" }),
  ).toBeVisible();
  await page.goto("/portal/research/work-controls");
  await expect(
    page.getByRole("heading", { name: "Map matched controls" }),
  ).toBeVisible();
  await page.goto("/launch");
  await expect(page).toHaveURL(/\/portal\/bench$/);
});
