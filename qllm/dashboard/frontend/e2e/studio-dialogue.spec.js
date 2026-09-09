import { test, expect } from "@playwright/test";
import { mockApi } from "./fixtures.js";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Physics Modelling a trajectory", exact: true }).click();
  await page.getByRole("button", { name: "Meet Lyra", exact: true }).click();
  await page.getByRole("button", { name: "Enter first-person", exact: true }).click();
  await expect(page.getByRole("button", { name: "Show conversation", exact: true })).toBeVisible();
});

test("talk, send, history and Escape preserve the visitor and unfinished reply", async ({ page }) => {
  const scene = page.locator(".meeting-game-canvas");
  const canvas = scene.locator("canvas");
  const visitor = await scene.getAttribute("data-visitor");
  const camera = await scene.getAttribute("data-camera");
  const look = await scene.getAttribute("data-look");
  await canvas.press("t");
  await expect(scene).toHaveAttribute("data-camera-owner", "conversation-portrait");
  await expect(page.getByLabel("In-room dialogue", { exact: true })).toBeVisible();
  await expect(page.locator(".meeting-scene-wrap")).toHaveAttribute("inert", "");
  const composer = page.getByLabel("Your question", { exact: true });
  await expect(composer).toBeFocused();
  await composer.fill("Which assumption should we examine first?");
  await composer.press("Enter");
  await expect(page.getByLabel("Research conversation", { exact: true })).toContainText("I’ve kept your question");
  await expect(page.locator(".rw-message")).toHaveCount(2);
  await composer.fill("Unfinished thought: wasd");
  await composer.press("Escape");
  await expect(canvas).toBeFocused();
  await expect(page.locator(".meeting-scene-wrap")).not.toHaveAttribute("inert");
  await expect(scene).toHaveAttribute("data-visitor", visitor);
  await expect(scene).toHaveAttribute("data-camera", camera);
  await expect(scene).toHaveAttribute("data-look", look);
  await expect(scene).toHaveAttribute("data-camera-owner", "pointer-look");
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  await expect(composer).toHaveValue("Unfinished thought: wasd");
  await page.getByRole("button", { name: "Full conversation", exact: true }).click();
  const log = page.getByLabel("Research conversation", { exact: true });
  await expect(log).toBeFocused();
  await expect(log).toContainText("Come on in");
  await expect(page.locator(".rw-message")).toHaveCount(4);
  await expect(composer).toHaveValue("Unfinished thought: wasd");
  await expect(page.locator(".meeting-scene-wrap")).toHaveAttribute("inert", "");
  await expect(scene).toHaveAttribute("data-camera-owner", "conversation-portrait");
  await log.press("Escape");
  await expect(canvas).toBeFocused();
  await expect(scene).toHaveAttribute("data-visitor", visitor);
  await expect(page.getByRole("dialog")).toBeVisible();
});

test("portrait resize and seated conversation return preserve the original posture and view", async ({ page }) => {
  await page.getByLabel("Room controls", { exact: true }).click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  const scene = page.locator(".meeting-game-canvas");
  await expect(scene).toHaveAttribute("data-posture", "seated");
  const camera = await scene.getAttribute("data-camera"), look = await scene.getAttribute("data-look");
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  await expect(scene).toHaveAttribute("data-camera-owner", "conversation-portrait");
  const portrait = await scene.getAttribute("data-camera");
  await page.setViewportSize({ width: 320, height: 568 });
  await expect(scene).toHaveAttribute("data-camera", portrait);
  await page.getByLabel("Your question", { exact: true }).fill("wasd stays a draft");
  await page.getByLabel("Your question", { exact: true }).press("Escape");
  await expect(scene).toHaveAttribute("data-camera-owner", "seated-look");
  await expect(scene).toHaveAttribute("data-posture", "seated");
  await expect(scene).toHaveAttribute("data-camera", camera);
  await expect(scene).toHaveAttribute("data-look", look);
});

test("320px dialogue keeps a real scene and composer visible without overlapping", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  await expect(page.getByLabel("Your question", { exact: true })).toBeFocused();
  const bounds = await page.evaluate(() => {
    const box = (selector) => {
      const r = document.querySelector(selector).getBoundingClientRect();
      return { top: r.top, bottom: r.bottom, height: r.height };
    };
    return { height: innerHeight, width: innerWidth, scroll: document.documentElement.scrollWidth,
      scene: box(".meeting-game-canvas"), dialogue: box(".rw-conversation"), composer: box(".rw-compose") };
  });
  expect(bounds.width).toBe(bounds.scroll);
  expect(bounds.scene.height).toBeGreaterThanOrEqual(180);
  expect(bounds.scene.top).toBeGreaterThanOrEqual(0);
  expect(bounds.scene.bottom).toBeLessThanOrEqual(bounds.dialogue.top + .5);
  expect(bounds.composer.bottom).toBeLessThanOrEqual(bounds.height + .5);
});

test("dialogue shortcuts preserve paper handoff and expose complete experiment controls", async ({ page }) => {
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  await page.getByRole("button", { name: "Show me a working note", exact: true }).click();
  await expect(page.locator(".meeting-game-canvas")).toHaveAttribute("data-note-offered", "true");
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  await page.getByRole("button", { name: "Try a browser experiment", exact: true }).click();
  await expect(page.getByLabel("Conversation workspace", { exact: true })).toBeVisible();
  await expect(page.locator(".conversation-trial")).toBeVisible();
  await expect(page.locator('.conversation-trial input')).toBeFocused();
  await expect(page.locator(".meeting-scene-wrap")).toHaveAttribute("inert", "");
  await page.locator('.conversation-trial input').press("Escape");
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
  await expect(page.getByRole("dialog")).toBeVisible();
});

test("deliberate Escape steps back, exits cleanly, and the same visit can reopen", async ({ page }) => {
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  const composer = page.getByLabel("Your question", { exact: true });
  await composer.fill("Keep this question for later");
  await composer.press("Escape");
  const canvas = page.getByLabel(/^Research room game/);
  await expect(canvas).toBeFocused();
  await expect(page.getByRole("dialog")).toBeVisible();
  await canvas.press("Escape");
  await expect(page.locator("dialog")).toHaveCount(0);
  await page.getByRole("button", { name: "Meet Lyra", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(composer).toHaveValue("Keep this question for later");
});

test("repeated result-reader returns cannot leave an invisible mounted studio", async ({ page }) => {
  await page.getByRole("button", { name: "Show conversation", exact: true }).click();
  await page.getByRole("button", { name: "Try a browser experiment", exact: true }).click();
  await page.getByRole("button", { name: "Run browser experiment", exact: true }).click();
  await page.getByRole("button", { name: "Take result card", exact: true }).click();
  const canvas = page.getByLabel(/^Research room game/);
  for (let visit = 0; visit < 2; visit++) {
    await page.getByRole("button", { name: "Show conversation", exact: true }).click();
    await page.getByRole("button", { name: "Read carried result", exact: true }).click();
    const reader = page.getByLabel("Collected result reader", { exact: true });
    await expect(reader).toBeFocused();
    await expect(reader).toContainText("40.77 m");
    await reader.press("Escape");
    await expect(canvas).toBeFocused();
    await expect(page.getByRole("dialog")).toBeVisible();
  }
});
