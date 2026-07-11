export function cloneConfig(value) {
  return JSON.parse(JSON.stringify(value))
}

export function ensureBlocks(config) {
  const draft = cloneConfig(config)
  const model = draft.model
  if (model.arch !== 'transformer') {
    model.blocks = null
    return draft
  }
  const count = Number(model.n_blocks || 0)
  if (!model.blocks) {
    model.blocks = Array.from({ length: count }, () => ({
      attn_type: model.attn_type || 'classical',
      ffn_type: model.ffn_type || 'classical',
      quantum: cloneConfig(model.quantum),
    }))
  }
  return draft
}

export function changeArchitecture(config, arch, options = {}) {
  const next = cloneConfig(config)
  next.model.arch = arch
  if (arch === 'transformer') {
    next.model.encoder_kind = 'none'
    return ensureBlocks(next)
  }
  next.model.blocks = null
  next.model.attn_type = 'classical'
  next.model.ffn_type = 'classical'
  next.model.embed_type = 'classical'
  next.model.head_type = 'linear'
  if (arch !== 'two_stream') next.model.encoder_kind = 'none'
  if (
    !next.model.quantum
    && (options.quantumArchitectures || []).includes(arch)
    && options.quantumDefault
  ) {
    next.model.quantum = cloneConfig(options.quantumDefault)
  }
  return next
}
