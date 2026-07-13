// PLACEHOLDER Atlas ontology — a frontend-local seed, NOT the canonical source.
//
// docs/RESEARCH_MAP.yaml is a FLAT list of ~19 mechanism areas; it deliberately
// does not contain the curated domain → pipeline-component → head-to-head tree the
// Atlas needs. That tree is backend/docs-owned (a new RESEARCH_MAP section or
// sibling file served at GET /api/atlas/ontology). Until that ships, this file
// stands in: the domain grouping is hand-authored here, but every cell's
// integrity-bearing fields (seed_status / seed_claim_level /
// seed_replication_status) are transcribed VERBATIM from the matching
// RESEARCH_MAP area and are shown as an "unverified seed" fallback — the real
// claim/replication come from the /verdicts store when a snapshot matches.
//
// Replace this file wholesale when the canonical ontology endpoint lands.

export const ATLAS_SEED = Object.freeze({
  schema_version: 1,
  source: 'frontend-seed',
  note: 'PLACEHOLDER — replace when GET /api/atlas/ontology ships. Grouping is illustrative; per-area status/claim/replication are verbatim from RESEARCH_MAP.yaml.',
  domains: [
    {
      id: 'lm_components',
      label: 'Language-model components',
      description: 'Dropping quantum layers into a classical text model in place of classical ones.',
      cells: [
        { id: 'c_variational_swaps', area_id: 'variational_component_swaps', label: 'Variational embedding / attention / FFN / block swaps', kind: 'head_to_head', pipeline_stage: 'representation', quantum_resource: 'interference', advantage_target: 'predictive_quality', seed_status: 'negative', seed_claim_level: 'diagnostic', seed_replication_status: 'multi_seed_single_instance' },
        { id: 'c_two_stream', area_id: 'two_stream_conditioning', label: 'Quantum sentence summary conditioning a classical token model', kind: 'head_to_head', pipeline_stage: 'representation', quantum_resource: 'entanglement', advantage_target: 'predictive_quality', seed_status: 'blocked', seed_claim_level: 'diagnostic', seed_replication_status: 'multi_seed_single_instance' },
        { id: 'c_unitary_transplant', area_id: 'unitary_weight_transplant', label: 'Unitary weight transplant and low-rank warm starts', kind: 'head_to_head', pipeline_stage: 'model', quantum_resource: 'interference', advantage_target: 'parameter_efficiency', seed_status: 'quantum_inspired', seed_claim_level: 'diagnostic', seed_replication_status: 'multi_seed_single_instance' },
      ],
    },
    {
      id: 'sequence_memory',
      label: 'Sequence & memory models',
      description: 'Recurrent and memory tasks where quantum dynamics might carry more state.',
      cells: [
        { id: 'c_qrnn', area_id: 'qrnn_representability_and_training', label: 'Quantum recurrent model representability and optimization', kind: 'head_to_head', pipeline_stage: 'model', quantum_resource: 'entanglement', advantage_target: 'memory_efficiency', seed_status: 'partial', seed_claim_level: 'diagnostic', seed_replication_status: 'multi_seed_single_instance' },
        { id: 'c_monitored_memory', area_id: 'monitored_quantum_memory', label: 'Monitored Ising sequences and predictive-memory scaling', kind: 'head_to_head', pipeline_stage: 'data_acquisition', quantum_resource: 'entanglement', advantage_target: 'memory_efficiency', seed_status: 'blocked', seed_claim_level: 'diagnostic', seed_replication_status: 'single_task_instance' },
        { id: 'c_parity_memory', area_id: 'heuristic_parity_memory', label: 'Interleaved parity-memory task and routed phase cell', kind: 'head_to_head', pipeline_stage: 'model', quantum_resource: 'interference', advantage_target: 'memory_efficiency', seed_status: 'blocked', seed_claim_level: 'mechanism', seed_replication_status: 'single_task_instance' },
        { id: 'c_theorem_faithful', area_id: 'theorem_faithful_contextual_sequence_learning', label: 'Theorem-faithful contextual quantum sequence learning', kind: 'unexplored', pipeline_stage: 'model', quantum_resource: 'contextuality', advantage_target: 'memory_efficiency', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
      ],
    },
    {
      id: 'interference_generation',
      label: 'Interference & generation heads',
      description: 'Using coherent interference at the output / generation stage.',
      cells: [
        { id: 'c_interference_head', area_id: 'interference_output_head', label: 'Coherent interference as an output-head primitive', kind: 'head_to_head', pipeline_stage: 'inference', quantum_resource: 'interference', advantage_target: 'expressivity_or_depth', seed_status: 'partial', seed_claim_level: 'mechanism', seed_replication_status: 'multi_seed_single_instance' },
        { id: 'c_sequential_interference', area_id: 'sequential_interference', label: 'Sequential cancellation with interference heads', kind: 'head_to_head', pipeline_stage: 'generation', quantum_resource: 'interference', advantage_target: 'predictive_quality', seed_status: 'negative', seed_claim_level: 'diagnostic', seed_replication_status: 'multi_seed_single_instance' },
        { id: 'c_generative', area_id: 'quantum_generative_models', label: 'Quantum generative sampling on structured discrete distributions', kind: 'unexplored', pipeline_stage: 'generation', quantum_resource: 'interference', advantage_target: 'sample_complexity', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
      ],
    },
    {
      id: 'kernels_features',
      label: 'Kernels & feature maps',
      description: 'Quantum feature maps and kernels versus classical kernels/features.',
      cells: [
        { id: 'c_kernel_geometry', area_id: 'kernel_geometry_controls', label: 'Quantum-kernel geometry, target alignment, and engineered controls', kind: 'head_to_head', pipeline_stage: 'representation', quantum_resource: 'interference', advantage_target: 'sample_complexity', seed_status: 'methodology_only', seed_claim_level: 'diagnostic', seed_replication_status: 'within_study_resampling' },
        { id: 'c_projected_kernels', area_id: 'projected_quantum_kernels', label: 'Projected quantum kernels and observable-aligned features', kind: 'unexplored', pipeline_stage: 'representation', quantum_resource: 'entanglement', advantage_target: 'sample_complexity', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
        { id: 'c_shadow_models', area_id: 'post_variational_shadow_models', label: 'Quantum feature acquisition then classical training and deployment', kind: 'unexplored', pipeline_stage: 'representation', quantum_resource: 'measurement', advantage_target: 'sample_complexity', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
      ],
    },
    {
      id: 'trainability',
      label: 'Trainability & training',
      description: 'Whether quantum components can be trained at scale — diagnostics and methods.',
      cells: [
        { id: 'c_barren_plateau', area_id: 'barren_plateau_scaling', label: 'Trainability, concentration, and barren-plateau diagnostics', kind: 'head_to_head', pipeline_stage: 'training', quantum_resource: 'entanglement', advantage_target: 'training_time', seed_status: 'infrastructure', seed_claim_level: 'diagnostic', seed_replication_status: 'within_study_resampling' },
        { id: 'c_hardware_training', area_id: 'hardware_feasible_training_methods', label: 'Hardware-feasible optimization of quantum components', kind: 'head_to_head', pipeline_stage: 'training', quantum_resource: 'measurement', advantage_target: 'training_time', seed_status: 'open', seed_claim_level: 'untested', seed_replication_status: 'none' },
      ],
    },
    {
      id: 'frontier_paradigms',
      label: 'Frontier paradigms (quantum-native)',
      description: 'Greenfield settings where the input or channel is itself quantum — no direct classical head-to-head.',
      cells: [
        { id: 'c_coherent_data', area_id: 'coherent_quantum_data_learning', label: 'Learning from coherent quantum data or experiments', kind: 'quantum_only', pipeline_stage: 'data_acquisition', quantum_resource: 'entanglement', advantage_target: 'sample_complexity', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
        { id: 'c_reservoir', area_id: 'quantum_reservoir_and_dynamic_circuits', label: 'Fixed quantum reservoirs, mid-circuit measurements, and dynamic recurrence', kind: 'unexplored', pipeline_stage: 'model', quantum_resource: 'measurement', advantage_target: 'training_time', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
        { id: 'c_fault_tolerant', area_id: 'fault_tolerant_ml_primitives', label: 'Fault-tolerant quantum subroutines inside ML training or inference', kind: 'quantum_only', pipeline_stage: 'training', quantum_resource: 'fault_tolerance', advantage_target: 'query_complexity', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
        { id: 'c_communication_limited', area_id: 'communication_limited_quantum_learning', label: 'Entanglement-assisted distributed and communication-limited learning', kind: 'quantum_only', pipeline_stage: 'data_acquisition', quantum_resource: 'entanglement', advantage_target: 'query_complexity', seed_status: 'unexplored', seed_claim_level: 'untested', seed_replication_status: 'none' },
      ],
    },
  ],
  // Cross-cell relations mirror RESEARCH_MAP.yaml `relations` (illustrative subset).
  relations: [
    { from_cell: 'c_variational_swaps', to_cell: 'c_kernel_geometry', type: 'explained_by' },
    { from_cell: 'c_barren_plateau', to_cell: 'c_qrnn', type: 'constrains' },
    { from_cell: 'c_interference_head', to_cell: 'c_sequential_interference', type: 'related_mechanism' },
  ],
})

export default ATLAS_SEED
