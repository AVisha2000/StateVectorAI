// Deliberately small, illustrative models. These are not research evidence.
import { projectile } from "./projectileToy.js";
export const DEMO_MODELS = {
  physics: {
    title: "A projectile, two launch angles",
    parameter: "Launch angle",
    unit: "°",
    min: 20,
    max: 70,
    step: 1,
    initial: 45,
    axis: "Horizontal distance / metres",
    code: "y = x*tan(angle) - 9.81*x*x / (2*20*20*cos(angle)**2)",
  },
  mathematics: {
    title: "A damped wave",
    parameter: "Frequency",
    unit: "",
    min: 1,
    max: 5,
    step: 0.25,
    initial: 2,
    axis: "x",
    code: "y = sin(frequency*x) / (1 + x)",
  },
  chemistry: {
    title: "A first-order reaction model",
    parameter: "Rate constant",
    unit: "s⁻¹",
    min: 0.1,
    max: 1.5,
    step: 0.1,
    initial: 0.5,
    axis: "Time / seconds",
    code: "concentration = exp(-rate*time)",
  },
  biology: {
    title: "A logistic growth model",
    parameter: "Growth rate",
    unit: "",
    min: 0.1,
    max: 1.5,
    step: 0.1,
    initial: 0.5,
    axis: "Time / arbitrary units",
    code: "population = 1 / (1 + 9*exp(-growth*time))",
  },
  "ai-safety": {
    title: "A toy gate on synthetic scores",
    parameter: "Decision threshold",
    unit: "",
    min: 0.1,
    max: 0.9,
    step: 0.05,
    initial: 0.5,
    axis: "Synthetic score (not model data)",
    code: "allowed = syntheticScore < threshold ? 1 : 0",
  },
  "machine-learning": {
    title: "Gradient descent on a quadratic",
    parameter: "Step size",
    unit: "",
    min: 0.05,
    max: 0.9,
    step: 0.05,
    initial: 0.2,
    axis: "Iteration",
    code: "x = (1 - stepSize)**iteration; loss = x*x / 2",
  },
};
export function simulateStudio(id, parameter) {
  const model = DEMO_MODELS[id];
  if (
    !model ||
    !Number.isFinite(parameter) ||
    parameter < model.min ||
    parameter > model.max
  )
    throw new Error("Choose a parameter within the displayed range.");
  const sample = (p) =>
    Array.from({ length: 121 }, (_, i) => {
      let x = i / 12,
        y;
      if (id === "physics") {
        const position = projectile(p, i / 120);
        x = position.x;
        y = position.y;
      } else if (id === "mathematics") y = Math.sin(p * x) / (1 + x);
      else if (id === "chemistry") y = Math.exp(-p * x);
      else if (id === "biology") y = 1 / (1 + 9 * Math.exp(-p * x));
      else if (id === "ai-safety") {
        x = i / 120;
        y = x < p ? 1 : 0;
      } else {
        x = i / 6;
        y = (1 - p) ** (2 * x) / 2;
      }
      return { x, y };
    });
  return {
    id,
    parameter,
    baselineParameter: model.initial,
    baseline: sample(model.initial),
    points: sample(parameter),
    provenance:
      "Built-in browser illustration with synthetic inputs. No research agent or external service was used.",
  };
}
