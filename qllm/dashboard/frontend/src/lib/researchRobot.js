import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export const ROBOT_FIELDS = Object.freeze(["physics", "mathematics", "chemistry", "biology", "ai-safety", "machine-learning"]);
const templates = new Map();

// One CPU template per visited discipline. Rendered instances own their geometry
// and materials, so disposing a studio cannot invalidate another studio's robot.
export function loadResearchRobot(id) {
  if (!ROBOT_FIELDS.includes(id)) return Promise.resolve(null);
  if (!templates.has(id)) {
    const request = fetch(`/models/research-robots/${id}-robot.glb`, { signal: AbortSignal.timeout(12000) }).then(async (response) => {
      if (!response.ok) throw new Error("Robot asset unavailable");
      return new GLTFLoader().parseAsync(await response.arrayBuffer(), "");
    }).then(({ scene }) => {
      if (!scene.getObjectByName("gesture_arm")) throw new Error("Robot shoulder rig missing");
      const head = scene.getObjectByName("attention_head");
      if (!head || !["eye_left", "eye_right"].every((name) => scene.getObjectByName(name)?.parent === head))
        throw new Error("Robot face rig missing");
      return scene;
    }).catch((error) => { templates.delete(id); throw error; });
    templates.set(id, request);
  }
  return templates.get(id);
}

export function cloneResearchRobot(template) {
  if (!template) return null;
  const model = template.clone(true);
  const geometry = new Map(), materials = new Map();
  const cloneMaterial = (source) => {
    if (!materials.has(source)) materials.set(source, source.clone());
    return materials.get(source);
  };
  model.traverse((object) => {
    if (!object.isMesh) return;
    if (!geometry.has(object.geometry)) geometry.set(object.geometry, object.geometry.clone());
    object.geometry = geometry.get(object.geometry);
    object.material = Array.isArray(object.material) ? object.material.map(cloneMaterial) : cloneMaterial(object.material);
    object.castShadow = true;
    object.receiveShadow = true;
  });
  model.userData.arm = model.getObjectByName("gesture_arm");
  model.userData.head = model.getObjectByName("attention_head");
  model.userData.eyes = ["eye_left", "eye_right"].map((name) => model.getObjectByName(name)).filter(Boolean);
  model.userData.robotAsset = true;
  return model;
}
