import { DEMO_MODELS } from "./studioSimulation.js";
import { captureResultCard } from "./resultCards.js";

// One session owner serves the in-room tool and detailed bench. The fired input
// is captured at start; later aiming never changes an in-flight calculation.
export function createStudioCalculation({ createWorker, onState, onResult }) {
  let worker = null,
    disposed = false;
  return {
    start(id, parameter, context = null) {
      if (disposed || worker) return false;
      const model = DEMO_MODELS[id];
      if (
        !model ||
        !Number.isFinite(parameter) ||
        parameter < model.min ||
        parameter > model.max
      )
        return false;
      onState({ busy: true, error: "", context });
      let current;
      const finish = (result, error = "") => {
        if (disposed || current !== worker) return;
        worker = null;
        current.terminate();
        if (result) onResult(result, context);
        onState({ busy: false, error, context });
      };
      try {
        current = createWorker();
        worker = current;
        current.onmessage = ({ data }) => {
          if (data?.error)
            finish(
              null,
              "The browser calculation failed. Your previous result is unchanged. Try again.",
            );
          else if (
            data?.result?.id !== id ||
            data.result.parameter !== parameter ||
            !captureResultCard({ result: data.result })
          )
            finish(
              null,
              "The browser returned an invalid or different calculation. Your previous result is unchanged. Try again.",
            );
          else finish(data.result);
        };
        current.onerror = () =>
          finish(
            null,
            "The browser calculation could not run. Your previous result and draft are unchanged. Try again.",
          );
        current.postMessage({ id, parameter });
      } catch {
        current?.terminate();
        worker = null;
        onState({
          busy: false,
          context,
          error:
            "This browser could not start the calculation. Your draft is unchanged. Try again or reopen the studio.",
        });
      }
      return true;
    },
    dispose() {
      disposed = true;
      worker?.terminate();
      worker = null;
    },
  };
}
