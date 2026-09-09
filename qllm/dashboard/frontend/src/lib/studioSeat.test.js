import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { createCoffeeSeat } from "./studioSeat.js";
import { STUDIO_LAYOUT, VISITOR_SPAWN } from "./studioLayout.js";
import { walkClearance, stepVisitor } from "./studioWalk.js";
import { canReturnNote } from "./carriedNote.js";
import { canPinResult } from "./resultCards.js";

test("seating is presentation-only and stand restores the exact saved walking look", () => {
  const seat = createCoffeeSeat(), player = { ...VISITOR_SPAWN };
  const initial = { ...player };
  assert.equal(seat.seated, false);
  assert.equal(seat.position(player), player);
  for (let visit = 0; visit < 5; visit++) {
    seat.sit(.75, -.07);
    assert.equal(seat.seated, true);
    assert.equal(seat.position(player), STUDIO_LAYOUT.visitorChair);
    assert.deepEqual(player, initial);
    assert.ok(walkClearance(player) >= 0);
    assert.deepEqual(seat.stand(), { yaw: .75, pitch: -.07 });
    assert.equal(seat.position(player), player);
    assert.equal(seat.height, .8);
    assert.equal(seat.fov, 64);
  }
  assert.notDeepEqual(stepVisitor(player, .75, 1, 0, .05), player);
});

test("repeated sit does not replace return look and repeated stand is harmless", () => {
  const seat = createCoffeeSeat();
  seat.sit(-2.1, .3);
  assert.equal(seat.sit(1, -.5), null);
  assert.deepEqual(seat.stand(), { yaw: -2.1, pitch: .3 });
  assert.equal(seat.stand(), null);
});

test("seated lens centres the robot face at narrow and wide aspects", () => {
  const seat = createCoffeeSeat(), look = seat.sit(.75, -.07);
  const target = new T.Vector3(STUDIO_LAYOUT.researcher.x, .78, STUDIO_LAYOUT.researcher.z);
  for (const aspect of [.45, 1, 1.78, 2.4]) {
    const camera = new T.PerspectiveCamera(seat.fov, aspect, .05, 35);
    const at = seat.position(VISITOR_SPAWN);
    camera.position.set(at.x, seat.height, at.z);
    camera.rotation.set(look.pitch, look.yaw, 0, "YXZ");
    camera.updateMatrixWorld();
    const ndc = target.clone().project(camera);
    assert.ok(ndc.toArray().every(Number.isFinite));
    assert.ok(Math.abs(ndc.x) < 1e-12 && Math.abs(ndc.y) < 1e-12);
    assert.ok(ndc.z > -1 && ndc.z < 1);
  }
});

test("seated interactions use the chair position: note in reach, board still distant", () => {
  const seat = createCoffeeSeat();
  assert.equal(canReturnNote(seat.position(VISITOR_SPAWN), true), false);
  seat.sit(.75, -.07);
  assert.equal(canReturnNote(seat.position(VISITOR_SPAWN), true), true);
  assert.equal(canPinResult(seat.position(VISITOR_SPAWN), true), false);
  seat.stand();
  assert.equal(canReturnNote(seat.position(VISITOR_SPAWN), true), false);
});
