import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import { transformWithEsbuild } from "vite";

// Exercise the actual JSX component's event handlers without a WebGL/browser
// dependency. Native focus/layout/disclosure behavior is covered by browser QA.
const source = await readFile(new URL("./RoomControls.jsx", import.meta.url), "utf8");
const { code } = await transformWithEsbuild(source, "RoomControls.jsx", { loader: "jsx", format: "cjs", jsx: "automatic" });
const require = createRequire(import.meta.url);
function fixture() {
  const calls = [];
  const inside = {}, outside = {}, document = { activeElement: inside };
  const details = { open: true, contains: (node) => node === inside };
  const refs = [{ current: details }, { current: { focus: () => calls.push("summary") } }];
  const module = { exports: {} };
  runInNewContext(code, { module, exports: module.exports, document,
    require: (id) => id === "react" ? { useRef: () => refs.shift() } : require(id),
  });
  const apiRef = { current: {
    release: () => calls.push("release"),
    focus: () => { calls.push("canvas"); document.activeElement = outside; },
  } };
  const tree = module.exports.default({ apiRef, children: "camera", actions: "actions", instructions: "keyboard help" });
  return { tree, panel: tree.props.children[1], details, document, inside, outside, calls };
}

test("controls use a initially closed native disclosure with complete accessible content", () => {
  const { tree, panel } = fixture();
  assert.equal(tree.type, "details");
  assert.equal(tree.props.open, undefined);
  assert.equal(tree.props.children[0].type, "summary");
  assert.equal(tree.props.children[0].props["aria-label"], "Room controls");
  assert.equal(panel.props.children[0], "camera");
  assert.equal(panel.props.children[1].props.children, "keyboard help");
  assert.equal(panel.props.children[3], "actions");
});

test("opening releases movement; Escape consumes only the disclosure and restores summary focus", () => {
  const { tree, details, calls } = fixture();
  tree.props.onToggle({ currentTarget: details });
  const event = { key: "Escape", preventDefault: () => calls.push("prevent"), stopPropagation: () => calls.push("stop") };
  tree.props.onKeyDown(event);
  assert.equal(details.open, false);
  assert.deepEqual(calls, ["release", "prevent", "stop", "summary"]);
  tree.props.onKeyDown(event);
  tree.props.onToggle({ currentTarget: details });
  assert.equal(calls.length, 4, "closed disclosure must not consume the next Escape");
});

test("focus can traverse controls; leaving closes without stealing the destination", () => {
  const { tree, details, inside, outside, calls } = fixture();
  tree.props.onBlur({ currentTarget: details, relatedTarget: inside });
  assert.equal(details.open, true);
  tree.props.onBlur({ currentTarget: details, relatedTarget: outside });
  assert.equal(details.open, false);
  assert.deepEqual(calls, []);
});

test("actions close controls but do not override an already selected reader or compose focus", () => {
  const f = fixture(), event = { target: { closest: () => ({}) } };
  f.panel.props.onClick(event);
  assert.equal(f.details.open, false);
  assert.deepEqual(f.calls, ["canvas"]);
  const g = fixture();
  g.document.activeElement = g.outside;
  g.panel.props.onClick(event);
  assert.equal(g.details.open, false);
  assert.deepEqual(g.calls, []);
  const h = fixture();
  h.panel.props.onClick({ target: { closest: () => null } });
  assert.equal(h.details.open, true);
});
