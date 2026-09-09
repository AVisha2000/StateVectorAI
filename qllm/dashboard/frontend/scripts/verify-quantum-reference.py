"""CPU-only independent QuTiP check of the actual browser model.

No runtime dependency: pass --qutip-path for an isolated --target installation.
Run from the repository root with the checked-in Python environment.
"""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qutip-path", type=Path)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if args.qutip_path:
        sys.path.insert(0, str(args.qutip_path.resolve()))
    import numpy as np
    import qutip as qt

    rng = random.Random(20260905)
    cases = [[]]
    for axis in "xyz":
        for angle in (0, np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi):
            cases.append([{"axis": axis, "angle": angle}])
    for _ in range(200):
        cases.append([
            {"axis": rng.choice("xyz"), "angle": rng.choice((-1, 1)) * rng.randint(1, 6) * np.pi / 12}
            for _ in range(rng.randint(1, 128))
        ])
    # Evaluate fractional replay states as well as complete sequences.
    requests = [{"pulses": pulses, "fraction": fraction}
                for pulses in cases for fraction in (0, len(pulses) * 0.37, len(pulses))]
    javascript = """
      import {simulatePulses, targetOverlap, QUANTUM_TARGETS} from './src/lib/quantumPilot.js';
      let text = ''; for await (const part of process.stdin) text += part;
      console.log(JSON.stringify(JSON.parse(text).map(({pulses, fraction}) => {
        const {state} = simulatePulses(pulses, fraction);
        return {state, overlaps: QUANTUM_TARGETS.map(t => targetOverlap(state, t.vector))};
      })));
    """
    frontend = Path(__file__).resolve().parents[1]
    output = subprocess.run(
        [args.node, "--input-type=module", "-e", javascript], cwd=frontend,
        input=json.dumps(requests), text=True, capture_output=True, check=True,
    )
    actual = json.loads(output.stdout)
    pauli = dict(zip("xyz", (qt.sigmax(), qt.sigmay(), qt.sigmaz())))
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    targets = [(zero + one).unit(), one, (zero + 1j * one).unit()]
    max_state_error = max_overlap_error = 0.0
    for request, result in zip(requests, actual, strict=True):
        state = zero
        for index, pulse in enumerate(request["pulses"]):
            portion = max(0, min(1, request["fraction"] - index))
            if portion:
                unitary = (-0.5j * pulse["angle"] * portion * pauli[pulse["axis"]]).expm()
                state = unitary * state
        expected = [float(qt.expect(operator, state)) for operator in pauli.values()]
        overlaps = [abs(target.overlap(state)) ** 2 for target in targets]
        max_state_error = max(max_state_error, float(np.max(np.abs(np.array(expected) - result["state"]))))
        max_overlap_error = max(max_overlap_error, float(np.max(np.abs(np.array(overlaps) - result["overlaps"]))))
    assert max_state_error < 1e-10, max_state_error
    assert max_overlap_error < 1e-10, max_overlap_error
    print(json.dumps({
        "reference": "QuTiP " + qt.__version__, "seed": 20260905,
        "sequences": len(cases), "state_checks": len(requests),
        "max_bloch_error": max_state_error, "max_overlap_error": max_overlap_error,
        "tolerance": 1e-10, "status": "pass",
    }, indent=2))


if __name__ == "__main__":
    main()
