# StateVector — Session Handoff (2026-07-14)

Audience: the next coding agent continuing this work. This document is
self-contained: it explains what was built, exactly how it is designed and
implemented, the conventions you must follow, and the remaining work queue with
enough detail to execute without re-deriving decisions. Read it fully before
touching code. When in doubt, the repository's `AGENTS.md` chain is the
baseline authority and this document is the session-specific context on top.

---

## 0. Non-negotiable rules (read first)

These are enforced by tests, review, and the repository's research protocol.
Violating any of them is a defect even if the code "works".

**Research integrity (frontend and backend):**
1. Never compute or display a composite "advantage score". Per-dimension
   values only.
2. `claim_level`, `claim_status`, and `replication_status` are canonical
   ledger fields — render them verbatim, never derive or reclassify them in
   React. The dashboard's derived `assessment_*` fields are separate and must
   stay visibly separate.
3. Null results ("classical holds", "no advantage found") are first-class:
   never dimmed by default, never colored green, never hidden. Emphasis in any
   visualization must be **additive-only** — hover/selection may raise
   something, but nothing may ever render *below* its resting style.
4. Diagnostics (gradient variance, entanglement, expressibility, SNR) are
   mechanism observations — label them as diagnostics, never as advantage.
5. Wall-clock time from simulators is "simulator cost" — never present it as
   QPU cost.
6. Colors come from the validated CVD-safe tokens only: `--q` (quantum
   magenta), `--c` (classical blue), `--accent` (indigo), `--null` (neutral),
   `--good`/`--warn`/`--crit`, plus their `-soft` variants. No new hues.

**Human gates (require the user's explicit, named approval — never proceed):**
- GPU/QPU runs, long sweeps, high-memory simulation, cluster/cloud spend.
- Enabling any paid provider / entering credentials (the "D4" decision:
  research-service LLM/embedding providers stay disabled until the user names
  provider, credential handling, and a daily cost budget). Three untracked
  files exist for this (`.env.example`, `qllm/research_llm.py`,
  `tests/test_research_llm.py`) — leave them untouched.
- Strengthening any scientific conclusion, promoting a claim level, or
  editing claim-bearing text in `RESULTS.md`.
- Deleting/rewriting experiment artifacts, databases, or user work.
- The queued 12-seed GPU confirmation study (GPU_QUEUE.md item 6) is NOT
  authorized.

**Ownership boundary:** the frontend track owns `qllm/dashboard/frontend/**`
and the UI-log section of `docs/BUILD_COORDINATION.md`. Backend code
(`qllm/**` outside the frontend, `scripts/**`, backend tests) is owned by the
backend track. If you need a backend change, post the ask in the UI log of
`docs/BUILD_COORDINATION.md` — do not reach across without the user's explicit
instruction.

---

## 1. Repository topology and git workflow

Two worktrees, one repo:

| Worktree | Branch | Purpose |
| --- | --- | --- |
| `StateVectorAI` | `main` | integration; merges happen HERE |
| `StateVectorAI-ui-redesign` | `ui-redesign` | frontend feature work |

**Per-increment workflow (follow exactly):**

```bash
# 1. work + commit on ui-redesign (in the ui-redesign worktree)
cd StateVectorAI-ui-redesign
git add <only the files you intentionally changed>   # see CRLF warning below
git commit -m "feat(...): ..."
git push origin ui-redesign

# 2. merge to main FROM THE MAIN WORKTREE (never from ui-redesign)
cd ../StateVectorAI
git fetch origin
git merge --no-ff origin/ui-redesign -m "Merge ui-redesign: <summary>"
git push origin main

# 3. resync the feature branch
cd ../StateVectorAI-ui-redesign
git fetch origin && git rebase origin/main
git push -f origin ui-redesign
```

**Footgun #1 (this burned us):** never run
`git reset --hard origin/main && git merge ui-redesign` inside the
*ui-redesign* worktree. That resets the ui-redesign branch onto main and then
merges it into itself — a silent no-op that ships nothing. Always merge from
the main worktree, merging the remote ref `origin/ui-redesign`.

**Footgun #2 (CRLF churn):** `core.autocrlf=true` on this machine makes
`git status` show dozens of files as modified after any rebase — with
symmetric N-insertions/N-deletions diffs (pure line-ending smudge). These are
NOT real changes. Never `git add -A`. Always stage files by explicit path and
sanity-check `git diff --cached --stat` before committing (a real change has
asymmetric counts).

**Coordination doc:** `docs/BUILD_COORDINATION.md` lives on `main` and is the
only file both tracks may commit directly to `main`. Append UI-log entries
newest-last, dated (`2026-07-14 · ui: ...`). Edit only your own section.

---

## 2. Frontend architecture and conventions

Stack: React 18 + Vite + react-router 6 + TanStack Query. Recharts for
charts; **all graphs/diagrams are hand-authored SVG** (an earlier attempt to
use cytoscape was removed — external canvas graph libs are banned here).

**The one pattern that governs everything:** pure logic lives in
`src/lib/*.js` as framework-free functions with `node --test` unit tests;
presentation lives in `.jsx` files that only interpolate lib output. A `.jsx`
file must not contain geometry math, claim classification, or data shaping —
if you find yourself computing in JSX, extract a lib function and test it.

Key files:

| File | Role |
| --- | --- |
| `src/api.js` | thin fetch wrappers; one entry per endpoint |
| `src/lib/hooks.js` | TanStack Query hooks; `quiet404QueryOptions` + `isNotYetBuilt(error)` for graceful degradation on older backends |
| `src/lib/*.js` + `*.test.js` | pure view-model logic (verdictView, studyView, scalingView, curves, diagnostics, atlasModel, atlasMapLayout, circuitModel, …) |
| `src/surfaces/*.jsx` | route-level pages (Overview, Runs, RunDetail, Bench, Designer, Atlas, Studies, Verdicts, Scaling, …) |
| `src/components/**` | reusable presentation (charts.jsx, atlas/*, CircuitSvg, ModelGraphSvg) |
| `src/appShell.js` | NAV_GROUPS / routes metadata |
| `src/styles.css` | ALL styling; CSS custom-property design tokens, light + dark themes |
| `e2e/fixtures.js` | deterministic mock backend for Playwright (shapes mirror `qllm/dashboard/openapi.json`) |
| `e2e/*.spec.js` | functional E2E (Playwright, route-mocked, hermetic) |
| `e2e/visual.spec.js` | visual-regression snapshots tagged `@visual` (excluded from default run; win32 baselines committed) |

**Testing stack + verification commands** (run from
`qllm/dashboard/frontend`; current counts in parentheses — a regression below
these numbers means you broke something):

```powershell
npm test          # node --test unit suite            (102 passing)
npm run build     # vite production build             (must be clean)
npx playwright test --grep-invert @visual   # functional E2E (70 passing)
npx playwright test                         # + visual        (84 total)
```

**Visual-baseline regeneration recipe** (needed whenever a snapshotted surface
changes visibly). The Playwright web server reuses an existing `:4174`
listener, which can serve a STALE build — kill it first:

```bash
# kill any lingering preview server on :4174 (it may serve a stale dist)
netstat -ano | grep :4174 | grep LISTENING   # note PID, then taskkill //PID <pid> //F
rm -f e2e/visual.spec.js-snapshots/<surface>-*.png
npx playwright test visual --grep "<Surface>" --update-snapshots
```

Then eyeball both themes before committing. `visual.spec.js`'s `setup()` now
calls `page.emulateMedia({ reducedMotion: 'reduce' })` — all UI transitions
are gated behind `@media (prefers-reduced-motion: no-preference)`, so
snapshots always capture the deterministic end state. Keep new animations
inside that media query.

**mockApi pattern** (`e2e/fixtures.js`): a path→body table served through
`page.route('**/api/**')`. Pass overrides to change a route; `null` ⇒ 404 (for
degradation tests). Method-specific POST handlers are special-cased —
`POST /designer/circuit` supports an override key `'POST /designer/circuit'`
with `{status, body}` to simulate a registry rejection:

```js
// happy path is the default; a 400-rejection test looks like:
await mockApi(page, {
  'POST /designer/circuit': { status: 400, body: { detail: "ansatz 'ising' requires architecture='qrnn'." } },
})
// endpoint-absent (older backend) test:
await mockApi(page, { '/designer/circuit': null })
```

---

## 3. What was shipped this session (all on `main`)

| Commit | What |
| --- | --- |
| `ece5684` | Designer + Atlas aligned with the shipped backend contracts |
| `4ab0766` | Docs cleanup: 16 stale "proposed / pending merge" claims fixed |
| `5978220` | PLANS.md living entry: quantum-native expansion roadmap |
| `56180d5` | Causal two-stream fair CPU pair + 3-pair pilot recorded (pilot-only) |
| `3b5b0e5` | Atlas graph rebuilt as a force-clustered "research map" |
| `d8ba471` | Coordination log entries (latest sync point of both branches) |

Earlier in the same effort (already on main, listed for context): seed-band
val_ppl over steps in Study detail (`161afd6`), verdict revision-history
timeline (`ef2c0da`), Overview multi-seed studies strip (`b669f0a`),
run→scaling backlink (`8799d53`), scaling sweep-runs table with an integrity
fix (`cd009b0`), study-runs→run links (`d0c1d5d`).

The sections below give full design + implementation detail per item.

---

## 4. Designer contract alignment (shipped `ece5684`)

### 4.1 The backend contract (source of truth)

`GET /api/designer/circuit` → capabilities (registry-backed):

```json
{
  "schema_version": 1,
  "validation_only": true,
  "side_effect_free": true,
  "client_estimates_authoritative": false,
  "choices": {
    "architecture": ["qrnn"],
    "circuit_ansatz": ["hardware_efficient", "reuploading"],
    "qrnn_only_ansatz": ["ising"],
    "backend": ["pennylane", "tensorcircuit", "tensorcircuit_mps"],
    "readout": ["z", "zz"]
  },
  "defaults": { "ansatz": "reuploading", "n_qubits": 4, "n_circuit_layers": 2,
                "backend": "pennylane", "readout": "z", "architecture": null,
                "device": "default.qubit", "diff_method": "backprop",
                "shots": null, "mps_max_bond_dimension": null },
  "constraints": {
    "n_qubits": { "minimum": 1, "maximum": 12 },
    "n_circuit_layers": { "minimum": 1, "maximum": 8 },
    "qrnn_only_ansatz_requires_architecture": "qrnn",
    "tensorcircuit_mps_requires": ["mps_max_bond_dimension"]
  },
  "warnings": ["Validation never constructs a circuit, model, backend, job, or device.",
               "Circuit properties and diagnostics are not evidence of quantum advantage."]
}
```

`POST /api/designer/circuit` validates a request and REJECTS (400) when:
- `readout` is not `z` or `zz` (**there is no `all` readout** — the old UI
  offered it; that was the bug),
- `ansatz` is `ising` (a QRNN-only family) without `architecture='qrnn'`,
- architecture is `qrnn` but `backend != 'pennylane'` or `readout != 'z'`
  (the QRNN runtime does not dispatch through selectable backend/readout —
  those are compatibility values only),
- `backend == 'tensorcircuit_mps'` without `mps_max_bond_dimension` (it is an
  APPROXIMATE backend, never silently exact), or bond dim supplied for any
  other backend.

On success it returns `derived.trainable_circuit_parameters.value =
layers × qubits × 3` (+ `layers` for qrnn) — **the authoritative parameter
count**. Client-side gate counts are advisory; the response reviews them in
`client_estimates` and flags mismatches in `warnings`. `ignored_fields` lists
compatibility-only fields for qrnn.

### 4.2 Frontend implementation

`src/lib/circuitModel.js` — the pure rules live here:

```js
export const READOUTS = Object.freeze(['z', 'zz']) // registry READOUT_TYPES — no 'all'
export const QRNN_ONLY_ANSATZE = Object.freeze(['ising'])

export function designerConstraints({ ansatz, backend } = {}) {
  const isQrnn = QRNN_ONLY_ANSATZE.includes(ansatz)
  return {
    architecture: isQrnn ? 'qrnn' : null,
    backendLocked: isQrnn ? 'pennylane' : null, // compatibility value, not an execution selector
    readoutLocked: isQrnn ? 'z' : null,
    needsBondDim: !isQrnn && backend === 'tensorcircuit_mps',
  }
}
```

`toBenchSpec(circuit, { backend, readout, mpsMaxBondDimension })` applies
those rules so the emitted request is always registry-valid: ising forces
`architecture:'qrnn'` + pennylane/z, and `mps_max_bond_dimension` rides only
with `tensorcircuit_mps` (null otherwise — the backend rejects it elsewhere).

`src/lib/hooks.js` gained:

```js
export function useDesignerCapabilities() {
  return useQuery({
    queryKey: ['designer-capabilities'],
    queryFn: api.designerCapabilities,           // GET /designer/circuit
    staleTime: 5 * 60 * 1000,
    ...quiet404QueryOptions,                     // retry:false, no refocus spam
  })
}
```

(Note: the old `proposedQueryOptions` was renamed `quiet404QueryOptions`
because these endpoints are shipped — the quiet-404 behavior is retained only
so the UI degrades against an older backend build.)

`src/surfaces/Designer.jsx` behavior:
- Choices/bounds/defaults come from capabilities when present, with the static
  `circuitModel.js` constants as fallback.
- Selecting `ising` disables (locks) the Backend and Readout selects to
  `pennylane`/`z` and shows a hint explaining the compatibility values.
- Selecting `tensorcircuit_mps` reveals a required "Max bond dim" number
  input (min 1) and an "approximate, never silently exact" hint; the Validate
  button is disabled until it is ≥ 1.
- The Round-trip card is labeled `live`. On success it shows the
  registry-derived trainable-parameter count in the Properties card tagged
  `registry` (the drawn gate counts are relabeled "(drawn)" client
  estimates), plus `ignored_fields` and up to 3 backend warnings.
- Error handling distinguishes outage from rejection:

```jsx
validate.isError ? (
  isNotYetBuilt(validate.error) ? (
    <p className="hint">This backend build doesn't serve /designer/circuit yet — …</p>
  ) : (
    <p className="hint" style={{ color: 'var(--crit)' }}>
      Rejected by the registry: {validate.error?.message}
    </p>
  )
) : …
```

`api.js` `responseError` already surfaces the backend's `detail` string as
`error.message` and the HTTP status as `error.status` — `isNotYetBuilt(e)`
is just `e?.status === 404`.

E2E: `e2e/designer.spec.js` covers readout choices (`['z','zz']` exactly),
ising locks, MPS bond-dim control, live validation showing `24` derived
params (fixture), a 400 shown as "Rejected by the registry", and a 404 shown
as graceful degradation. Fixtures `DESIGNER_CAPABILITIES` /
`DESIGNER_VALIDATION` in `e2e/fixtures.js` mirror the shapes above.

---

## 5. Atlas live-ontology wiring (shipped `ece5684`)

`GET /api/atlas/ontology` is live and returns the canonical map (19 cells /
6 domains / 10 relations):

```
AtlasOntologyResponse {
  schema_version, source: 'backend-canonical', ontology_updated,
  research_map_schema_version, research_map_updated, note,
  claim_levels: [9 strings], replication_statuses: [6 strings],
  status_values: {8 entries},
  domains: [{ id, label, description, cells: [AtlasCell] }],
  relations: [{ from_cell, to_cell, type }]
}
AtlasCell { id, area_id, label, kind, portfolio, pipeline_stage,
  quantum_resource, advantage_target, seed_status, seed_claim_level,
  seed_replication_status, verdict_ref: null (ALWAYS null today), note }
```

Crucially the backend adopted the frontend seed's `seed_*` field names, so
`resolveOntology()` in `src/lib/atlasModel.js` consumes live data unchanged.
`Atlas.jsx` prefers live data (`ontologyQuery.data || ATLAS_SEED`); the
bundled seed is only an offline/older-backend fallback. Changes made:

- The fallback notice is runtime-truthful and distinguishes a 404 ("could not
  be fetched from this backend build") from a 5xx ("failed to load — check the
  backend's Atlas configuration", styled `.notice.crit`), via
  `isNotYetBuilt(ontologyQuery.error)`.
- Claim/replication filter dropdowns prefer the live vocabularies:
  `ontology.claim_levels?.length ? ontology.claim_levels : CLAIM_LEVELS`.
- Seed id drift fixed: `c_comm_limited` → `c_communication_limited` (deep
  links now survive seed↔live transitions).
- E2E exercises the live contract by default: the canonical ontology was
  captured **verbatim** from the backend function into
  `e2e/atlasOntology.fixture.js` (`ATLAS_ONTOLOGY`, served at
  `/atlas/ontology` in the mock table). One test overrides it with `null` to
  cover the seed fallback. If the backend ontology changes, regenerate the
  fixture with:

```bash
cd StateVectorAI && python -X utf8 -c "
import json, sys; sys.stdout.reconfigure(encoding='utf-8')
from qllm.dashboard.atlas import atlas_ontology_response
print(json.dumps(atlas_ontology_response().model_dump(), separators=(',',':')))"
```

**Known inert path (open backend ask, recorded in BUILD_COORDINATION.md):**
`verdict_ref` is always null today, so `joinCellVerdict()`'s verdict-join
refinement never fires and every cell renders with map-level (seed_*) claim
data. Do NOT try to fix this from the frontend — the backend must start
emitting `verdict_ref` (or a `claim_id` join convention must be agreed in the
coordination doc) once claim binding is validated.

---

## 6. Atlas "research map" graph (shipped `3b5b0e5`) — full design

The old graph was a flat 3-column tree. It is now a deterministic
force-clustered map. This is the largest piece; understand it before editing.

### 6.1 Files

| File | Content |
| --- | --- |
| `src/lib/atlasMapLayout.js` | ALL geometry: constants, hash, force sim, hulls, routes, labels, pan/zoom math. Pure, framework-free, `node --test`able. |
| `src/lib/atlasMapLayout.test.js` | 12 invariant tests (see 6.5) |
| `src/components/atlas/AtlasGraphSvg.jsx` | rendering + event wiring ONLY — zero geometry beyond string interpolation |
| `src/styles.css` (atlas block) | all `.atlas-map-*`, `.atlas-hull*`, `.atlas-route*`, `.atlas-cell*`, `.atlas-node-oc-*` styles |
| `src/components/atlas/AtlasLegend.jsx` | 3 rows; row 3 documents territories + route types |

The old `src/lib/atlasSvgLayout.js` (+ test) was DELETED. Do not resurrect it.

### 6.2 Determinism contract (the layout must never change between runs)

No `Math.random`, no `Date`, no locale-dependent ops. All variation derives
from `hash01(id)` (a mulberry32-style string hash → [0,1)). Fixed iteration
counts. Input-order traversal everywhere. `round2()` on every emitted
coordinate. The output is a pure function of `(resolved, expanded)`. The
determinism test calls `layoutMap` three times (with unrelated calls
interleaved) and asserts `deepStrictEqual` — keep it passing.

### 6.3 Layout algorithm (five stages, in `layoutMap(resolved, { expanded })`)

Constants (all exported, frozen):
`CELL_W=116, CELL_H=58, R_COLLIDE=68, HULL_PAD=22, HULL_SAMPLES=8, GUTTER=56,
SEAL_W=180, SEAL_H=44, MARGIN=40, TICKS=300, SETTLE_ITERS=40, ZOOM_MIN=0.6,
ZOOM_MAX=2.5`.

**Stage A — domain anchors (closed form).** Each visible domain gets a
cluster radius `clusterRadius(n)`; the second term is a ring-packing lower
bound that makes collision + containment jointly FEASIBLE (n centers pairwise
≥ 2·R_COLLIDE apart fit on a ring of radius `R_COLLIDE/sin(π/n)`):

```js
export function clusterRadius(n) {
  if (n <= 1) return 74
  return Math.max(34 * Math.sqrt(n) + 40, R_COLLIDE / Math.sin(Math.PI / n) + 8)
}
```

Anchors sit on an ellipse (`Ry = 0.62·Rx`), each domain's angular share
proportional to its arc demand `c_d = 2·(r_d + HULL_EXTENT) + GUTTER` where
`HULL_EXTENT = hypot(CELL_W,CELL_H)/2 + HULL_PAD ≈ 86.85` (the territory
outline reaches that far beyond cell centers — using only HULL_PAD here was a
bug that made neighboring hulls collide). `Rx = max(330, (ΣC × 1.12)/5.16)` —
the 1.12 factor compensates chords being shorter than arcs on the flattened
ellipse; 5.16 is the Ramanujan circumference of this ellipse per unit Rx.

**Stage B — seeding.** Cells seed on a phyllotaxis spiral inside their
cluster: `θ = k·2.39996113 + 2π·hash01(domainId)`,
`ρ = min(46·√(k+0.5), 0.9·r_d)`. Coincident-seed guard offsets by hashed
epsilons.

**Stage C — force simulation.** `TICKS=300` iterations, cooling
`alpha = 0.5 · 0.985^t`, all position-based (no velocities):
1. anchor spring: `p += (anchor − p) · 0.06 · alpha`
2. same-domain same-`pipeline_stage` pair springs: rest 110, gain 0.04·alpha
   (this creates spatial sub-clustering by stage)
3. cross-domain relation springs: rest 260, gain 0.008·alpha (related cells
   lean toward each other across territories)
4. collision: two passes/tick; pairs closer than `2·R_COLLIDE` are displaced
   apart symmetrically; exact-overlap pairs separate along a hashed angle
5. containment: cells outside `r_d − 4` are pulled 20 % back for the first
   80 % of ticks, then hard-clamped.

Then `resolveCollisions()` runs `SETTLE_ITERS=40` alternating passes of
{collision separation + hard containment clamp}. Because of the packing bound
in `clusterRadius`, this terminates with **zero overlaps and containment** —
both asserted by unit tests, so a tuning regression fails CI instead of
shipping a broken map.

Finally everything is translated so the padded content bbox starts at
`(MARGIN, MARGIN)`; `width`/`height` derive from the bbox; every coordinate is
`round2`ed.

**Stage D — territory hulls + seals.** Per expanded domain: sample
`HULL_SAMPLES=8` points on a circle of radius `HULL_EXTENT` around each member
cell center, take the convex hull (Andrew monotone chain, collinear-safe),
then smooth with a closed midpoint-quadratic path (path through edge
midpoints, hull vertices as `Q` control points — never spikes). Labels sit
above the hull (`minY − 12`), flipping below (`maxY + 16`) when they would
collide with an already-placed label (deterministic, domain order). Collapsed
domains emit a `seal` (a 180×44 rounded rect at the anchor) instead — seals
are **inert in v1** (no role, no tabIndex) to preserve the E2E contract that
`g[role="button"]` counts 19 expanded / 0 collapsed.

**Stage E — typed relation routes.** Only relations whose both cells are
present render. Endpoints are ray/rect-boundary intersections pushed 4 px
outward. Each route is a cubic Bézier bowed away from the content center
(`bow = clamp(0.18·len, 24, 80) + 18·fanIndex`, fan separating duplicate
pairs). Type styling is data-driven and shared with the legend:

```js
export const RELATION_STYLE = Object.freeze({
  constrains: { dash: null, marker: 'bar' },
  must_not_be_conflated_with: { dash: null, marker: 'bar' },
})
export function relationStyle(type) {
  return RELATION_STYLE[type] || { dash: '6 4', marker: 'arrow' } // associative default
}
```

The real ontology vocabulary is: `motivates, candidate_model_for,
must_not_be_conflated_with, constrains, system_level_followup,
redirects_toward_data_alignment, alternative_training_strategy,
deployment_variant, shares_measurement_features, related_access_model`.
Associative types are dashed with a chevron marker; constraint-like types are
solid with an inhibition bar. **All routes are `var(--accent)`** — type is
carried by dash + terminal only, because per-type colors would collide with
the cell outcome color channel (e.g. a blue edge would read as
"classical_holds").

Node output shape (consumed by the renderer):

```js
{ id, type:'cell', x, y /* CENTERS */, w:116, h:58, kind, outcome,
  claimRank, replicationRank, stage, domainId, label,
  lines: wrapLabel(label),          // ≤2 lines, ≤18 chars, char-count based
  ariaLabel,                        // spoken: label — outcome; claim: …; replication: …; kind cell
  cell }
```

### 6.4 Renderer (`AtlasGraphSvg.jsx`) — structure and event model

Public props unchanged: `{ resolved, expanded, onSelect, selectedId }`.
DOM skeleton:

```
div.atlas-graph-svg.card            (position:relative, overflow:hidden)
├─ div.atlas-map-controls           (.atlas-zoom-in / -out / -reset buttons)
└─ div.atlas-map-frame              tabIndex=0 role=group data-zoom="1.00"
   └─ svg viewBox="0 0 W H" role=group    ← role=img was an a11y bug (made
      ├─ defs: #atlas-dots pattern, #atlas-arrow + #atlas-bar markers    the 19 button cells presentational)
      └─ g.atlas-map-viewport  style transform: translate(tx,ty) scale(k)
         ├─ rect.atlas-graticule    (dot grid; the ONLY pan/dblclick target)
         ├─ path.atlas-hull × N  + text.atlas-hull-label
         ├─ g.atlas-seal × collapsed (inert)
         ├─ per relation: path.atlas-route [data-relation] + path.atlas-route-hit (12px invisible twin for hover)
         ├─ g.atlas-node-layer
         │   └─ per cell: g.atlas-cell.atlas-node-oc-{outcome} role=button tabIndex=0
         │        data-cell-id aria-label aria-pressed  style transform:translate(x,y)
         │        └─ g.atlas-cell-body        ← nested group is LOAD-BEARING:
         │             shape.atlas-cell-back     the CSS hover lift transforms
         │             shape.atlas-cell-face     the INNER group so it can't
         │             [shape.atlas-sel-ring]    clobber the positioning transform
         │             text lines + stage + <title>
         └─ hovered/selected routes only: g.atlas-route-label (type pill)
```

Cell visual encoding (unchanged semantics, new skin):
- fill = outcome **soft** token + stroke = outcome **full** token
  (`.atlas-node-oc-classical_holds .atlas-cell-face { fill:var(--c-soft);
  stroke:var(--c); }` etc.) — this also fixed a contrast failure where white
  text sat on saturated fills;
- `strokeWidth = 1 + claimRank × 0.5` (claim level → border width);
- `strokeDasharray = '5 3'` when `replicationRank === 0` (no replication);
- silhouette per kind via `shapeGeom(kind)`: rect rx10 (head-to-head),
  hexagon with 16 px cuts (quantum_only), diamond (suggested), ellipse
  (unexplored);
- **selection renders a SEPARATE ring** (`shapeGeom(kind, w, h, 4)`,
  `.atlas-sel-ring`, accent stroke). Never restyle the face on selection —
  the old renderer overwrote the claim/replication border with the selection
  stroke, destroying two encoding channels. There is an E2E test pinning this.

Pan/zoom state is `{ k, tx, ty }` starting at identity (first paint is
interaction-independent → visual baselines stay stable):
- **wheel zoom** only with ctrl/cmd held (never hijack page scroll), attached
  via `addEventListener('wheel', …, { passive:false })` in a `useEffect` (JSX
  `onWheel` can't reliably `preventDefault`); cursor-anchored via
  `zoomAtPoint` (pure lib fn, keeps the anchor point invariant, clamps k to
  [0.6, 2.5]);
- **drag pan** starts only on the graticule rect (never on cells — click-to-
  select must never be swallowed), uses pointer capture, adds a `dragging`
  class that kills the viewport transition during the drag;
- **`clampTransform`** guarantees ≥120 user units of the content always
  overlap the viewport (the map can't be lost off-screen);
- **keyboard** on the frame only when `e.target === e.currentTarget` (so
  Enter/Space on cells is untouched): arrows pan 40 units, `+`/`=`/`-` zoom,
  `0` resets; `data-zoom` always reflects `k.toFixed(2)` for E2E;
- **deep-link focus**: when `selectedId` changes and the node is outside the
  visible user-rect, the viewport recenters via `focusTransform` (same k);
  the 240 ms viewport CSS transition makes it glide; end state deterministic.

Hover model (**additive-only emphasis** — integrity rule): hovering a cell or
route adds `.hot` to incident routes (opacity .6→1, width 1.5→2) and shows
the relation-type pill; nothing is ever dimmed below resting state.

### 6.5 The 12 layout unit tests (keep all passing)

determinism (3× deep-equal); all coordinates round2ed + finite; zero card
overlap (pairwise ≥ 2·R_COLLIDE − 0.01); containment of all 4 card corners in
the own-domain hull polygon + **SAT-based convex disjointness between hull
polygons** (bbox tests false-positive on diagonal neighbors — do not "fix"
them back to bboxes); clusterRadius ≥ ring-packing bound for n=2..8; collapse
semantics (seal, no cells, no hull, relations filtered); channel fields +
ariaLabel present on every node; route dash/marker consistency + endpoints on
card boundaries; wrapLabel behavior; hull degeneracy (1-cell, collinear);
transform math (anchor invariance, clamping, focus centering, toUser
round-trip).

### 6.6 E2E additions (`e2e/atlas.spec.js`, 7 new tests)

zoom toolbar (data-zoom changes and resets); selection ring + aria-pressed +
face stroke ≠ ring stroke; 6 hulls ⇄ 6 seals on collapse with the
`g[role="button"]` count still 0; 10 typed routes with `motivates` dashed
`6 4` and `constrains` solid + `marker-end url(#atlas-bar)`; null-outcome
face opacity 1 and non-green; keyboard zoom on the frame while Enter on a
cell still selects. All 10 pre-existing atlas tests pass byte-unchanged.

---

## 7. Docs cleanup (shipped `4ab0766`) — what changed where

Current-state claims were reconciled with reality (backend fully merged at
`b54ad30`; both new endpoints in the committed `qllm/dashboard/openapi.json`):

- `docs/BUILD_COORDINATION.md`: `/atlas/ontology` and `/designer/circuit`
  moved into the **Stable** contract table; decision rows D1–D6 now read
  "shipped, on main" (D4 stays open/user-gated); UI-log entries appended.
- `docs/UI_REDESIGN_PLAN.md`: header status corrected (was "implementation
  not started"); verdict-store and SSE passages rewritten as
  "at design time … since shipped"; next-actions list struck through.
- `PLANS.md`: OpenAPI-codegen item un-blocked (snapshot IS on main; only the
  codegen work remains); Atlas placeholder language updated.
- `docs/FEATURE_UPGRADES.md`: §1 "partially built — scan shipped, synthesis
  D4-gated"; §2 "2b shipped; 2a zoom still proposed".

Rule followed (keep following it): historical log entries that were true when
written stay untouched; only CURRENT-state claims (tables, status headers)
get corrected, with dated status notes rather than silent rewrites.

---

## 8. Causal two-stream study execution (recorded `56180d5`) — READ CAREFULLY

This is research work with strict framing rules. What was run (CPU only,
fully isolated, per the packet in `PLANS.md` "R1 causal two-stream"):

```bat
.venv\Scripts\python.exe benchmarks\two_stream_probe.py --suite two-stream-causal-v2 ^
  --dataset text --variants quantum-bias classical-bias none --seeds 0 --steps 1500 ^
  --results-db %LOCALAPPDATA%\Temp\qllm-r1-causal-pair-isolated\results.sqlite ^
  --out-dir %LOCALAPPDATA%\Temp\qllm-r1-causal-pair-isolated\artifacts ^
  --device-target cpu --no-mlflow --dashboard
:: then the same command with --seeds 0 1 2 (completed cells are skipped)
```

Results (val_ppl; candidate 25,701 params, control 25,841 — ~0.5 % larger =
conservative, ablation 25,217):

| seed | quantum-bias | classical-bias | none | Δ (q−c) |
| --- | --- | --- | --- | --- |
| 0 | 8.9716 | 9.8266 | 10.1706 | −0.8550 |
| 1 | 8.8751 | 9.6609 | 9.7371 | −0.7858 |
| 2 | 9.6814 | 9.9992 | 10.4499 | −0.3178 |

Paired Δ mean **−0.6529, sd 0.2922**, candidate lower in 3/3.

**Framing rules you must never break when touching this topic:**
- Three pairs are PILOT-ONLY regardless of nominal statistics (protocol:
  `docs/RESEARCH_PROGRAM.md`; `minimum_confirmatory_pairs: 6` in
  `research/claims.yaml`). No p-value is claimed.
- The claim `two_stream_conditioning` stays `diagnostic` / `rerun_required`.
  `RESULTS.md` and `claims.yaml` were NOT touched and must not be without the
  user's explicit approval (claim promotion is human-gated).
- The isolated artifacts under `%LOCALAPPDATA%\Temp\qllm-r1-causal-pair-isolated`
  must be preserved and never appended to `two-stream-v1` or the shared
  `results/qllm_results.db`.
- Never pool with suite `two-stream-v1` (teacher-forced; the harness's
  `validate_suite` hard-rejects it — do not weaken that guard).
- Pilot-derived power plan (already recorded in PLANS.md): the 6-pair
  protocol floor governs; a 6-pair CPU confirmation (~5 min) is adequately
  powered IF pilot variance holds. Running it as a first-class Study + claim
  contract is a **user decision** — do not start it unprompted. The 12-seed
  3000-step GPU proposal remains NOT authorized.

---

## 9. Quantum-native expansion roadmap (recorded `5978220`)

Full text lives in `PLANS.md` under "Active plan: quantum-native expansion".
Summary for orientation (implementation is BACKEND-OWNED — coordinate before
touching, see §0 ownership):

The verification machinery (evidence ladder, claim ledger, fair-control
protocol, seed-axis statistics, scaling harness) is already task-agnostic and
**fails closed** on non-perplexity metrics — deliberately. What is hard-wired
to language modeling: `val_ppl` as a schema column and progress key
(`qllm/resultsdb.py`), the softmax-CE loop (`qllm/train/loop.py`), the
pairable-metric frozensets (`qllm/dashboard/lab.py:74`,
`qllm/dashboard/studies.py:84`), and the absence of a task-type dimension.

Five ordered steps (each independently shippable):
1. **Metric registry** in `qllm/registry.py`: `METRIC_TYPES` mapping
   `metric_type → {lower_is_better, units, pairable, extraction_key,
   comparator_class}`; replace the frozenset checks with registry lookups;
   serve to the frontend via `/config/choices`. **The critical property:
   metric admission and metric extraction must live in ONE entry** so a
   perplexity number can never be relabeled as an energy.
2. **Primary-metric indirection** in `resultsdb.py`
   (`primary_metric_name/value`; val_ppl becomes the sequence-modeling
   specialization).
3. **Task dimension**: `TASK_TYPES = (sequence_modeling, ground_state,
   combinatorial_optimization)` + a task-conditional `ProblemConfig` validated
   in `validate_config` (copy the existing `tensorcircuit_mps` conditional
   pattern in `qllm/config.py`).
4. **VQE vertical slice** — a SIBLING runner (never generalize `loop.py`)
   minimizing a problem-Hamiltonian expectation over the existing
   circuit/backend layer, `metric_type=ground_state_energy_error` vs exact
   diagonalization, writing through the same resultsdb/Study/verdict path.
5. **`solver_competition_v1` fairness schema** in `claims.yaml` (equal-budget,
   declared solver versions, certified optima) — distinct from
   `controlled_component_ablation_v1`, which stays QML-only.

Frontend implication when Step 1 lands: `src/lib/verdictView.js` and ~30
surfaces hard-code `val_ppl` labels — they should switch to a metric list
served by the backend. Do not pre-build this before the backend registry
exists.

Integrity guards (recorded in the plan; enforce in review): no metric
relabeling; per-task fairness schemas; pairing on the problem-instance axis;
shot/measurement budgets are part of any VQE/QAOA claim (analytic-gradient
convergence is a simulator diagnostic only); per-domain practical-effect
thresholds predeclared (e.g. chemical accuracy 1.6 mHa); new domains enter
the research map at level 0/unexplored with the full audit trail.

---

## 10. Remaining work queue (in recommended order)

### 10.1 Atlas: clickable seals (small, self-contained frontend)
Collapsed-domain seals are inert. Make them toggle expansion. This REQUIRES a
coordinated E2E change because two existing tests count
`.atlas-graph-svg g[role="button"]` (19 expanded / 0 collapsed) — an
interactive seal would enter that count.

Implementation:
1. `Atlas.jsx` already has `toggleDomain(domainId)` — pass it into
   `<AtlasGraphSvg onToggleDomain={toggleDomain} …>`.
2. In `AtlasGraphSvg.jsx` give the seal `role="button" tabIndex={0}`,
   `aria-label={`Expand ${s.label} (${s.count} cells)`}`, `aria-expanded={false}`, onClick +
   Enter/Space → `onToggleDomain(s.domainId)`. Optionally do the same on the
   hull label to collapse.
3. Update the two E2E locators from `g[role="button"]` to `.atlas-cell`
   in `atlas.spec.js` ("19 clickable cells", "collapse all"), and add a test:
   collapse all → click a seal → that domain's hull + cells return.
4. Keep the seal OUT of `.atlas-cell` class so cell counts stay clean.

### 10.2 OpenAPI type codegen (frontend, unblocked)
`qllm/dashboard/openapi.json` is on main. Add a codegen step (e.g.
`openapi-typescript` as a devDependency) producing `src/api.types.ts` (or
JSDoc typedefs if staying pure JS — this repo is currently JS-only; prefer
generating a `.d.ts` and referencing it from JSDoc to avoid a TS migration).
Wire an npm script `types:generate` and a CI check that regenerating produces
no diff. Do NOT hand-edit generated output.

### 10.3 Backend ask already posted (do not implement frontend-side)
Atlas `verdict_ref` emission (see §5). Wait for the backend log to answer.

### 10.4 Expansion Step 1 (backend-owned — needs the user's explicit nod)
See §9. Sketch that matches current backend conventions:

```python
# qllm/registry.py
METRIC_TYPES = {
    "strict_autoregressive_next_token": {
        "lower_is_better": True, "units": "ppl", "pairable": True,
        "extraction_key": "val_ppl", "comparator_class": "matched_control",
    },
    "validation_perplexity": {
        "lower_is_better": True, "units": "ppl", "pairable": True,
        "extraction_key": "val_ppl", "comparator_class": "matched_control",
    },
    # future: "ground_state_energy_error": {..., "extraction_key": "energy_error",
    #          "comparator_class": "classical_solver"},
}
```

Then `lab.py`/`studies.py` replace
`metric_type in PAIRABLE_VAL_PPL_METRIC_TYPES` with
`METRIC_TYPES.get(metric_type, {}).get("pairable", False)` AND extract scores
via `METRIC_TYPES[metric_type]["extraction_key"]` instead of the literal
`"val_ppl"` — admission and extraction from the same entry, never separately.

### 10.5 Gated items (user decision required — do not start)
- 6-pair CPU confirmation of the causal study as a first-class Study.
- 12-seed GPU confirmation (GPU_QUEUE.md item 6).
- D4 research-service providers/budget.
- Public Atlas export (exposure-gated). Auth.

---

## 11. Quick reference — current green state

- Branches: `main` and `ui-redesign` synced (last sync commit `d8ba471`).
- Tests: **102 unit / 70 functional E2E / 14 visual (84 Playwright total)** —
  all passing. CI runs the functional E2E on every frontend push
  (`.github/workflows/dashboard-frontend.yml`); visual snapshots are excluded
  from CI (win32 baselines, `--grep-invert @visual`).
- Backend validation from repo root (should stay green; you normally don't
  need it for frontend-only changes):

```powershell
python scripts/check_agent_setup.py
pytest -q            # use --basetemp if Windows temp raises WinError 5
```

- Untracked on main (LEAVE ALONE): `.env.example`, `qllm/research_llm.py`,
  `tests/test_research_llm.py` (pending the D4 decision).

If a check fails that you didn't touch, investigate before proceeding — never
weaken or skip a test to make a run pass, and report failures verbatim.
