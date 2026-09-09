import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import { transformWithEsbuild } from "vite";

const source = await readFile(new URL("./StudioDialog.jsx", import.meta.url), "utf8");
const { code } = await transformWithEsbuild(source, "StudioDialog.jsx", { loader: "jsx", format: "cjs", jsx: "automatic" });
const require = createRequire(import.meta.url);
function fixture() {
  const calls = [], effects = [], document = { body: { style: { overflow: "auto" } } };
  const dialog = { open: false, isConnected: true,
    showModal() { this.open = true; calls.push("show"); },
    close() { this.open = false; calls.push("native-close"); },
  };
  const module = { exports: {} };
  runInNewContext(code, { module, exports: module.exports, document,
    require: id => id === "react" ? { useEffect: fn => effects.push(fn) } : require(id),
  });
  const tree = module.exports.default({ dialogRef: { current: dialog },
    onDismissRequest: () => calls.push("step-back"), onClose: () => calls.push("leave"),
    children: "room", className: "studio", "aria-labelledby": "session-title",
  });
  const event = (overrides = {}) => ({ target: dialog, currentTarget: dialog,
    key: "Escape", defaultPrevented: false, repeat: false, cancelable: true,
    nativeEvent: { isComposing: false },
    preventDefault() { calls.push("prevent"); }, stopPropagation() { calls.push("stop"); }, ...overrides });
  return { tree, dialog, calls, effects, document, event };
}

test("studio shell preserves native modal semantics and restores prior scroll state", () => {
  const f = fixture();
  assert.equal(f.tree.type, "dialog");
  assert.equal(f.tree.props["aria-labelledby"], "session-title");
  assert.equal(f.tree.props.children, "room");
  const cleanup = f.effects[0]();
  assert.equal(f.document.body.style.overflow, "hidden");
  assert.equal(f.dialog.open, true);
  cleanup();
  assert.equal(f.dialog.open, false);
  assert.equal(f.document.body.style.overflow, "auto");
});

test("Escape steps back once and suppresses the competing native close default", () => {
  const f = fixture();
  f.tree.props.onKeyDown(f.event());
  assert.deepEqual(f.calls, ["prevent", "stop", "step-back"]);
  f.calls.length = 0;
  f.tree.props.onKeyDown(f.event({ repeat: true }));
  assert.deepEqual(f.calls, ["prevent", "stop"], "holding Escape must not dismiss successive layers");
});

test("child-consumed Escape, IME and other keys retain their own ownership", () => {
  const f = fixture();
  f.tree.props.onKeyDown(f.event({ defaultPrevented: true }));
  f.tree.props.onKeyDown(f.event({ nativeEvent: { isComposing: true } }));
  f.tree.props.onKeyDown(f.event({ key: "t" }));
  assert.deepEqual(f.calls, []);
});

test("cancellable native requests step back; forced closure waits for authoritative close", () => {
  const f = fixture();
  f.tree.props.onCancel(f.event());
  assert.deepEqual(f.calls, ["prevent", "step-back"]);
  f.calls.length = 0;
  f.tree.props.onCancel(f.event({ cancelable: false }));
  assert.deepEqual(f.calls, [], "do not mutate a material when the browser will close the whole visit");
  f.tree.props.onClose(f.event());
  assert.deepEqual(f.calls, ["leave"], "native closure must reconcile the mounted visit");
});

test("reopened StrictMode and disconnected cleanup close events cannot leave a new visit", () => {
  const f = fixture();
  const cleanup = f.effects[0]();
  cleanup();
  f.effects[0]();
  f.calls.length = 0;
  f.tree.props.onClose(f.event());
  assert.deepEqual(f.calls, []);
  f.dialog.open = false;
  f.dialog.isConnected = false;
  f.tree.props.onClose(f.event());
  assert.deepEqual(f.calls, []);
});

test("a descendant dialog cannot trigger the studio's cancel or close lifecycle", () => {
  const f = fixture();
  f.tree.props.onCancel(f.event({ target: {} }));
  f.tree.props.onClose(f.event({ target: {} }));
  assert.deepEqual(f.calls, []);
});
