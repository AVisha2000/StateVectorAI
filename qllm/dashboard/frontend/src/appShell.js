export const THEME_STORAGE_KEY = 'qllm-theme'

export const NAV_ITEMS = Object.freeze([
  { to: '/', label: 'Overview', end: true },
  { to: '/explore', label: 'Explore' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/models', label: 'Model Builder' },
  { to: '/studies', label: 'Studies' },
  { to: '/results', label: 'Results' },
  { to: '/datasets', label: 'Datasets & Tasks' },
  { to: '/gpu', label: 'System' },
  { to: '/docs', label: 'Docs' },
])

export function resolveInitialTheme(savedTheme, prefersLight = false) {
  if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme
  return prefersLight ? 'light' : 'dark'
}
