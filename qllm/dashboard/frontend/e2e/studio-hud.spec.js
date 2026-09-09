import { test, expect } from "@playwright/test";
import { mockApi } from "./fixtures.js";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Physics Modelling a trajectory", exact: true }).click();
  await page.getByRole("button", { name: "Meet Lyra", exact: true }).click();
  await page.getByRole("button", { name: "Enter first-person", exact: true }).click();
});

test("focus corners mark usable targets at canvas center and stay out of conversation and aiming", async ({ page }) => {
  const reticle = page.locator(".meeting-reticle");
  const canvas = page.getByLabel(/^Research room game/);
  await expect(reticle).toBeVisible();
  await expect(reticle).not.toHaveClass(/has-target/);
  await expect(reticle).toHaveAttribute("aria-hidden", "true");
  await expect(reticle).toHaveCSS("pointer-events", "none");
  expect(await reticle.evaluate(el => getComputedStyle(el, "::before").content)).toBe("none");

  await page.getByLabel("Room controls", { exact: true }).click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await expect(reticle).toHaveClass(/has-target/);
  await expect(page.getByRole("button", { name: "Talk to researcher", exact: true })).toBeVisible();
  for (const width of [1280, 320]) {
    await page.setViewportSize({ width, height: 740 });
    const focus = await reticle.evaluate(el => {
      const dot = el.firstElementChild.getBoundingClientRect();
      const frame = el.parentElement.querySelector("canvas").getBoundingClientRect();
      const corners = getComputedStyle(el, "::before");
      return {
        x: dot.x + dot.width / 2 - frame.x - frame.width / 2,
        y: dot.y + dot.height / 2 - frame.y - frame.height / 2,
        content: corners.content, width: corners.width, height: corners.height,
        animation: corners.animationName, pointer: corners.pointerEvents,
        overflow: document.documentElement.scrollWidth > innerWidth,
      };
    });
    expect(Math.abs(focus.x)).toBeLessThan(.5);
    expect(Math.abs(focus.y)).toBeLessThan(.5);
    expect(focus).toMatchObject({ content: '""', width: "28px", height: "28px",
      animation: "none", pointer: "none", overflow: false });
  }

  await canvas.press("e");
  await expect(page.getByLabel("Your question", { exact: true })).toBeFocused();
  await expect(reticle).toBeHidden();
  await page.getByLabel("Your question", { exact: true }).press("Escape");
  await expect(reticle).toBeVisible();
  await expect(reticle).toHaveClass(/has-target/);
  await page.getByLabel("Room controls", { exact: true }).click();
  await page.getByRole("button", { name: "Use launcher", exact: true }).click();
  await expect(reticle).toBeHidden();
  await page.getByLabel(/^Launcher aiming/).press("Escape");
  await expect(reticle).toBeVisible();
  // The existing launcher stands the visitor before aiming; return keeps that pose.
  await expect(reticle).not.toHaveClass(/has-target/);
  expect(await reticle.evaluate(el => getComputedStyle(el, "::before").content)).toBe("none");
  await canvas.press("g");
  await expect(reticle).toHaveClass(/has-target/);
});

test("held arrow walks continuously, stops on outside release, and returns keyboard focus to the game", async ({ page }) => {
  const scene = page.locator(".meeting-game-canvas");
  const start = (await scene.getAttribute("data-visitor")).split(",").map(Number);
  const button = page.getByRole("button", { name: "Walk left", exact: true });
  const box = await button.boundingBox();
  await page.mouse.move(box.x + box.width/2, box.y + box.height/2);
  await page.mouse.down();
  await expect(scene).toHaveAttribute("data-walk-pointers", "1");
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
  await page.waitForTimeout(500);
  await page.mouse.move(5, 5);
  await page.mouse.up();
  await expect(scene).toHaveAttribute("data-walk-pointers", "0");
  const stopped = await scene.getAttribute("data-visitor");
  const end = stopped.split(",").map(Number);
  expect(Math.hypot(end[0]-start[0], end[1]-start[1])).toBeGreaterThan(.1);
  await page.waitForTimeout(300);
  await expect(scene).toHaveAttribute("data-visitor", stopped);
  await button.press("Enter");
  await expect(button).toBeFocused();
  await expect(scene).not.toHaveAttribute("data-visitor", stopped);
  await expect(scene).toHaveAttribute("data-walk-pointers", "0");
});

test("conversation interrupts a held arrow without resuming motion on return", async ({ page }) => {
  const scene = page.locator(".meeting-game-canvas");
  const box = await page.getByRole("button", { name: "Walk forward", exact: true }).boundingBox();
  await page.mouse.move(box.x + box.width/2, box.y + box.height/2);
  await page.mouse.down();
  await expect(scene).toHaveAttribute("data-walk-pointers", "1");
  await page.keyboard.press("t");
  await expect(page.getByLabel("Your question", { exact: true })).toBeFocused();
  await expect(scene).toHaveAttribute("data-walk-pointers", "0");
  const stopped = await scene.getAttribute("data-visitor");
  await page.mouse.up();
  await page.getByLabel("Your question", { exact: true }).press("Escape");
  await page.waitForTimeout(300);
  await expect(scene).toHaveAttribute("data-visitor", stopped);
  await expect(scene).toHaveAttribute("data-walk-pointers", "0");
});

test("Controls is keyboard-dismissible without leaving or losing room state", async ({ page }) => {
  const controls = page.getByLabel("Room controls", { exact: true });
  await controls.press("Enter");
  await expect(page.getByRole("button", { name: "Take a seat", exact: true })).toBeVisible();
  await controls.press("Escape");
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(controls).toBeFocused();
  await expect(page.locator(".meeting-room-controls")).not.toHaveAttribute("open");
  await controls.click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await expect(page.locator(".meeting-game-canvas")).toHaveAttribute("data-posture", "seated");
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
  await expect(page.locator(".meeting-room-controls")).not.toHaveAttribute("open");
});

test("320px carried-item and movement rails leave the actual canvas unobstructed", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  const controls = page.getByLabel("Room controls", { exact: true });
  await controls.click();
  await page.getByRole("button", { name: "Take a seat", exact: true }).click();
  await controls.click();
  await page.getByLabel("Room interactions", { exact: true }).getByRole("button", { name: "Take working note", exact: true }).click();
  const scene = page.locator(".meeting-game-canvas");
  await expect(scene).toHaveAttribute("data-note-settled", "true");
  const bounds = await page.evaluate(() => {
    const rect = (selector) => { const r = document.querySelector(selector).getBoundingClientRect(); return { top: r.top, bottom: r.bottom, height: r.height }; };
    return { width: innerWidth, scroll: document.documentElement.scrollWidth,
      canvas: rect(".meeting-game-canvas"), controls: rect(".meeting-room-controls"),
      item: rect(".meeting-note-pocket"), movement: rect(".meeting-control-hint") };
  });
  expect(bounds.scroll).toBe(bounds.width);
  expect(bounds.canvas.height).toBeGreaterThan(300);
  expect(bounds.controls.bottom).toBeLessThanOrEqual(bounds.canvas.top);
  expect(bounds.item.bottom).toBeLessThanOrEqual(bounds.canvas.top);
  expect(bounds.movement.top + .1).toBeGreaterThanOrEqual(bounds.canvas.bottom);
  await page.getByRole("button", { name: "Read note R", exact: true }).click();
  await expect(page.getByLabel("Working note reader")).toContainText("Compare boundary conditions");
  await page.getByRole("button", { name: "Ask about this passage" }).last().click();
  await expect(page.getByLabel("Your question", { exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Lower note R", exact: true }).click();
  await page.getByLabel(/^Research room game/).press("q");
  await expect(scene).toHaveAttribute("data-note-held", "false");
});

test("launcher returns from Controls to the same first-person visitor", async ({ page }) => {
  const scene = page.locator(".meeting-game-canvas");
  const visitor = await scene.getAttribute("data-visitor");
  await page.getByLabel("Room controls", { exact: true }).click();
  await page.getByRole("button", { name: "Use launcher", exact: true }).click();
  await expect(scene).toHaveAttribute("data-launcher-active", "true");
  await page.getByLabel(/^Launcher aiming/).press("Escape");
  await expect(scene).toHaveAttribute("data-launcher-active", "false");
  await expect(scene).toHaveAttribute("data-visitor", visitor);
  await expect(page.getByLabel("Room controls", { exact: true })).toBeVisible();
  await expect(page.getByLabel(/^Research room game/)).toBeFocused();
});
