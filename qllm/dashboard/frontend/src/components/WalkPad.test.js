import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import { transformWithEsbuild } from "vite";

const source = await readFile(new URL("./WalkPad.jsx", import.meta.url), "utf8");
const { code } = await transformWithEsbuild(source, "WalkPad.jsx", { loader: "jsx", format: "cjs", jsx: "automatic" });
const require = createRequire(import.meta.url);
function fixture() {
  const calls = [], cleanups = [], module = { exports: {} };
  runInNewContext(code, { module, exports: module.exports, require: id => id === "react"
    ? { useRef: value => ({ current: value }), useEffect: fn => cleanups.push(fn()) } : require(id) });
  const apiRef = { current: {
    beginWalk: (...args) => calls.push(["begin", ...args]), endWalk: id => calls.push(["end", id]),
    move: (...args) => calls.push(["step", ...args]),
  } };
  const tree = module.exports.default({ apiRef });
  const captures = new Set();
  const button = {
    setPointerCapture: id => captures.add(id), hasPointerCapture: id => captures.has(id),
    releasePointerCapture: id => captures.delete(id),
  };
  const event = (id = 3) => ({ pointerId: id, button: 0, currentTarget: button,
    preventDefault: () => calls.push(["prevent"]) });
  return { tree, props: tree.props.children[0].props, calls, captures, event, cleanups, button, apiRef };
}
test("movement pad retains four named semantic single-step controls", () => {
  const f = fixture();
  assert.equal(f.tree.props.role, "group");
  assert.match(f.tree.props["aria-label"], /Hold an arrow/);
  assert.equal(f.tree.props.children.length, 4);
  for (const child of f.tree.props.children) {
    assert.equal(child.type, "button"); assert.equal(child.props.type, "button");
    assert.match(child.props["aria-label"], /^Walk /);
  }
  f.props.onClick({ detail: 0 });
  assert.deepEqual(f.calls, [["step", 1, 0]]);
});
test("pointer down starts once; release outside stops and compatibility click adds no step", () => {
  const f = fixture();
  f.props.onPointerDown(f.event());
  assert.equal(f.captures.has(3), true);
  f.props.onPointerUp(f.event());
  f.props.onClick({ detail: 1 });
  assert.deepEqual(f.calls, [["prevent"], ["begin", 3, 1, 0], ["end", 3]]);
  assert.equal(f.captures.size, 0);
});
test("cancel, capture loss and unmount release their specific pointer", () => {
  for (const name of ["onPointerCancel", "onLostPointerCapture"]) {
    const f = fixture(); f.props.onPointerDown(f.event()); f.props[name](f.event());
    assert.deepEqual(f.calls.at(-1), ["end", 3]); assert.equal(f.captures.size, 0);
  }
  const f = fixture(); f.props.onPointerDown(f.event(4));
  f.tree.props.children[1].props.onPointerDown(f.event(8));
  f.cleanups[0]();
  assert.deepEqual(f.calls.slice(-2), [["end", 4], ["end", 8]]);
  assert.equal(f.captures.size, 0);
});
test("unavailable pointer capture falls back to one step; secondary buttons do nothing", () => {
  const f = fixture();
  f.props.onPointerDown({ ...f.event(), button: 2 }); assert.equal(f.calls.length, 0);
  f.button.setPointerCapture = () => { throw new Error("not supported"); };
  f.props.onPointerDown(f.event()); f.props.onClick({ detail: 1 });
  assert.deepEqual(f.calls, [["prevent"], ["step", 1, 0]]);
  f.apiRef.current = null;
  f.cleanups[0]();
});
