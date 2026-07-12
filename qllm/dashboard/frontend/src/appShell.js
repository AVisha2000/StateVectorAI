export const THEME_STORAGE_KEY = 'qllm-theme'

// New redesign information architecture, grouped for the sidebar. See
// docs/UI_REDESIGN_PLAN.md §4. Each item: { to, label, icon, end?, badge? }.
export const NAV_GROUPS = Object.freeze([
  Object.freeze({
    title: 'Research',
    items: Object.freeze([
      { to: '/', label: 'Overview', icon: '◈', end: true },
      { to: '/discover', label: 'Discover', icon: '✦', badge: 'COPILOT' },
      { to: '/library', label: 'Library', icon: '▤' },
      { to: '/atlas', label: 'Atlas', icon: '⬡' },
    ]),
  }),
  Object.freeze({
    title: 'Experiments',
    items: Object.freeze([
      { to: '/designer', label: 'Designer', icon: '⎔' },
      { to: '/bench', label: 'Bench', icon: '⚗' },
      { to: '/runs', label: 'Runs', icon: '≣' },
      { to: '/studies', label: 'Studies', icon: '⧉' },
      { to: '/verdicts', label: 'Verdicts', icon: '⚖' },
    ]),
  }),
  Object.freeze({
    title: 'System',
    items: Object.freeze([
      { to: '/datasets', label: 'Datasets', icon: '⊞' },
      { to: '/system', label: 'Queue & Backends', icon: '⌁' },
    ]),
  }),
])

// Flat ordered list, retained for tests and breadcrumb lookups.
export const NAV_ITEMS = Object.freeze(NAV_GROUPS.flatMap((group) => group.items))

// Old dashboard routes that must not 404 during the migration; each redirects
// to its new surface (docs/UI_REDESIGN_PLAN.md §4 migration table).
export const LEGACY_REDIRECTS = Object.freeze({
  '/overview': '/',
  '/explore': '/atlas',
  '/experiments': '/runs',
  '/jobs': '/runs',
  '/launch': '/bench',
  '/models': '/designer',
  '/results': '/verdicts',
  '/results/legacy': '/verdicts',
  '/scaling': '/runs',
  '/live': '/runs',
  '/gpu': '/system',
  '/docs': '/library',
})

export function resolveInitialTheme(savedTheme, prefersLight = false) {
  if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme
  return prefersLight ? 'light' : 'dark'
}

// Return the nav label for the active path, for the top-bar breadcrumb.
export function navTitleForPath(pathname) {
  if (pathname === '/') return 'Overview'
  const match = NAV_ITEMS.find((item) => item.to !== '/' && pathname.startsWith(item.to))
  return match ? match.label : ''
}
