// Pointer holds are independent of physical keyboard keys. Releasing a thumb
// must never release W; duplicate directions must never double walking speed.
export function createWalkInput() {
  const pointers = new Map();
  return {
    press(id, forward, right) {
      if (!Number.isInteger(id) || id < 0 || ![forward, right].every(v => [-1, 0, 1].includes(v)) || (!forward && !right)) return false;
      pointers.set(id, { forward, right });
      return true;
    },
    release(id) { pointers.delete(id); },
    clear() { pointers.clear(); },
    axes(keys) {
      let up = keys.has("w") || keys.has("arrowup"), down = keys.has("s") || keys.has("arrowdown");
      let left = keys.has("a") || keys.has("arrowleft"), right = keys.has("d") || keys.has("arrowright");
      for (const pointer of pointers.values()) {
        up ||= pointer.forward === 1; down ||= pointer.forward === -1;
        left ||= pointer.right === -1; right ||= pointer.right === 1;
      }
      return { forward: Number(up) - Number(down), right: Number(right) - Number(left) };
    },
    get size() { return pointers.size; },
  };
}
