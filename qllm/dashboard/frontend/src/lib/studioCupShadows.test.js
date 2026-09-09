import test from "node:test";
import assert from "node:assert/strict";
import * as T from "three";
import { createMeetingScene } from "../components/meetingScene.js";
import { disposeScene } from "../components/researchPlanetScene.js";
import { STUDIO_TYPES } from "./researchStudios.js";
import { ROBOT_FIELDS } from "./researchRobot.js";
import { createStudioLighting } from "./studioLighting.js";

test("small cups cast their outer surfaces without changing their visible sides or other casters", t => {
  const previous = globalThis.document;
  globalThis.document = { createElement: () => ({ getContext: () => ({
    clearRect() {}, fillRect() {}, strokeRect() {}, fillText() {}, beginPath() {}, arc() {}, fill() {},
    measureText: text => ({ width: text.length * 10 }),
  }) }) };
  t.after(() => { if (previous === undefined) delete globalThis.document; else globalThis.document = previous; });
  const scene = new T.Scene(), lighting = createStudioLighting(scene);
  t.after(() => lighting.dispose());
  const toLight = lighting.key.position.clone().sub(lighting.key.target.position).normalize();

  for (const field of ROBOT_FIELDS) {
    const room = createMeetingScene(STUDIO_TYPES.find(studio => studio.id === field));
    try {
      room.group.updateMatrixWorld(true);
      const cups = room.group.getObjectsByProperty("name", "Coffee cup");
      assert.equal(cups.length, 2, field);
      const table = room.group.getObjectByName("Furniture table").children[0];
      const top = new T.Box3().setFromObject(table).max.y;
      assert.equal(table.material.shadowSide, null, "table keeps the renderer's existing policy");
      for (const cup of cups) {
        const meshes = cup.children.filter(part => part.isMesh);
        assert.equal(meshes.length, 3, "body, coffee surface and handle");
        for (const mesh of meshes) {
          assert.equal(mesh.material.side, T.FrontSide, "visible culling is unchanged");
          assert.equal(mesh.material.shadowSide, T.FrontSide, "only the shadow caster side changes");
          assert.ok(mesh.castShadow && mesh.receiveShadow);
        }

        // CPU geometric diagnostic of actual WebGL caster-side selection, not a
        // raster/PCF simulation. Biased tabletop lookups can lie inside the cup:
        // its front surface precedes them, while default back faces can follow.
        const occludes = (point, useOldPolicy = false) => {
          const ray = new T.Raycaster(point.clone().addScaledVector(toLight, 2), toLight.clone().negate(), 0, 2);
          const original = meshes.map(mesh => mesh.material.side);
          try {
            meshes.forEach(mesh => { mesh.material.side = useOldPolicy ? T.BackSide : mesh.material.shadowSide; });
            return ray.intersectObject(cup, true).length > 0;
          } finally {
            meshes.forEach((mesh, i) => { mesh.material.side = original[i]; });
          }
        };
        let rejectedByOldPolicy = 0;
        for (let ix = -3; ix <= 3; ix++) {
          for (let iz = -3; iz <= 3; iz++) {
            const lookup = new T.Vector3(cup.position.x + ix * .01, top + lighting.key.shadow.normalBias, cup.position.z + iz * .01);
            assert.ok(occludes(lookup), `${field}: outer cup blocks a biased contact lookup`);
            if (!occludes(lookup, true)) rejectedByOldPolicy++;
          }
        }
        assert.ok(rejectedByOldPolicy > 0, "negative control must reproduce back-face light leakage");
      }
    } finally {
      disposeScene(room.group);
    }
  }
});
