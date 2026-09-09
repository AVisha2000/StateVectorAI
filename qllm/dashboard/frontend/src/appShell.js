export const THEME_STORAGE_KEY = 'qllm-theme'
export const PORTAL_BASE = '/portal'

export function portalPath(path = '/') {
  const suffix = path === '/' ? '' : (path.startsWith('/') ? path : `/${path}`)
  return `${PORTAL_BASE}${suffix}`
}

export function portalRelativePath(pathname) {
  if (pathname === PORTAL_BASE || pathname === `${PORTAL_BASE}/`) return '/'
  return pathname.startsWith(`${PORTAL_BASE}/`) ? pathname.slice(PORTAL_BASE.length) : pathname
}

export function isPortalPath(pathname) {
  return pathname === PORTAL_BASE || pathname.startsWith(`${PORTAL_BASE}/`)
}

// New redesign information architecture, grouped for the sidebar. See
// docs/UI_REDESIGN_PLAN.md §4. Each item: { to, label, icon, end?, badge? }.
export const NAV_GROUPS = Object.freeze([
  Object.freeze({
    title: 'Research',
    items: Object.freeze([
      { to: '/', label: 'Workboard', icon: '◈', end: true },
      { to: '/decisions', label: 'Decisions', icon: '□' },
      { to: '/discover', label: 'Discover', icon: '✦', badge: 'COPILOT' },
      { to: '/library', label: 'Library', icon: '▤' },
      { to: '/atlas', label: 'Atlas', icon: '⬡' },
    ]),
  }),
  Object.freeze({
    title: 'Experiments',
    items: Object.freeze([
      { to: '/lab', label: 'Lab overview', icon: '◉' },
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
  '/overview': portalPath('/lab'),
  '/explore': portalPath('/atlas'),
  '/experiments': portalPath('/runs'),
  '/jobs': portalPath('/runs'),
  '/launch': portalPath('/bench'),
  '/models': portalPath('/designer'),
  '/results': portalPath('/verdicts'),
  '/results/legacy': portalPath('/verdicts'),
  '/scaling': portalPath('/runs'),
  '/live': portalPath('/runs'),
  '/gpu': portalPath('/system'),
  '/docs': portalPath('/library'),
})

export function resolveInitialTheme(savedTheme, prefersLight = false) {
  if (savedTheme === 'light' || savedTheme === 'dark') return savedTheme
  return prefersLight ? 'light' : 'dark'
}

// Return the nav label for the active path, for the top-bar breadcrumb.
export function navTitleForPath(pathname) {
  const relativePath = portalRelativePath(pathname)
  if (relativePath === '/') return 'Workboard'
  if (relativePath.startsWith('/research/')) return 'Research record'
  const match = NAV_ITEMS.find((item) => item.to !== '/' && relativePath.startsWith(item.to))
  return match ? match.label : ''
}
