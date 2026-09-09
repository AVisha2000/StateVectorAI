import * as T from "three";

// One static authored key, one hemisphere fill, one existing scene environment.
// The room owns this mode policy; materials and the camera do not write lights.
export function createStudioLighting(scene) {
  const fill = new T.HemisphereLight("#e4f0e9", "#456558", 1.4);
  const key = new T.DirectionalLight("#ffe6c7", 2.2);
  key.position.set(-3, 6, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  // Frame projection reaches +/-3.71 in light X and +3.67 in Y. A 4-unit
  // half-extent leaves >.28 units for bias/filter support; 8/1024 units per texel.
  Object.assign(key.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: .5, far: 15 });
  key.shadow.camera.updateProjectionMatrix();
  key.shadow.normalBias = .02;
  scene.add(fill, key);
  let current = null;
  return {
    fill, key,
    present(firstPerson) {
      if (current === firstPerson) return;
      current = firstPerson;
      fill.intensity = firstPerson ? .65 : 1.4;
      key.intensity = firstPerson ? 2.6 : 2.2;
      scene.environmentIntensity = firstPerson ? .35 : .55;
    },
    dispose() {
      key.shadow.dispose();
      scene.remove(fill, key);
    },
  };
}
