const paths = {
  'quantum-island': (
    <>
      <circle cx="12" cy="12" r="8" />
      <ellipse cx="12" cy="12" rx="8" ry="3" transform="rotate(-35 12 12)" />
      <path d="M12 20V4m0 8 5-5" />
      <circle cx="17" cy="7" r="1.5" />
    </>
  ),
  world: (
    <>
      <circle cx="12" cy="12" r="9" />
      <ellipse cx="12" cy="12" rx="4" ry="9" />
      <path d="M3 12h18M5 6.5h14M5 17.5h14" />
    </>
  ),
  mathematics: <path d="M5 5h14M6 5l6 7-6 7h13" />,
  physics: (
    <>
      <path d="M9 15c-1-5 2-10 9-11 1 7-3 11-8 12l-3 3-2-2 3-3M14 5l5 5M10 16l-1 5M7 12H3" />
      <circle cx="14" cy="9" r="1.3" />
    </>
  ),
  chemistry: (
    <>
      <path d="M9 3h6M10 3v7L4 19a1.3 1.3 0 0 0 1 2h14a1.3 1.3 0 0 0 1-2l-6-9V3M7 15h10" />
      <path d="M10 18h.01M14 17h.01" />
    </>
  ),
  biology: (
    <>
      <path d="M19 4C8 2 2 9 6 15s13 4 13-11ZM5 21L16 8M9 14v-4M9 14h5" />
    </>
  ),
  'ai-safety': (
    <>
      <path d="m12 3 8 3v6c0 5-8 9-8 9S4 17 4 12V6Z" />
      <path d="m8 12 3 3 5-6" />
    </>
  ),
  'machine-learning': (
    <>
      <circle cx="5" cy="6" r="2" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="19" cy="6" r="2" />
      <circle cx="19" cy="18" r="2" />
      <circle cx="12" cy="12" r="2.5" />
      <path d="m6.5 7.5 3.5 3M14 13.5l3.5 3M6.5 16.5l3.5-3M14 10.5l3.5-3" />
    </>
  ),
  arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
  back: <path d="M19 12H5m5-5-5 5 5 5" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  plus: <path d="M12 5v14M5 12h14" />,
  minus: <path d="M5 12h14" />,
  pause: <path d="M9 5v14M15 5v14" />,
  play: <path d="m8 4 12 8-12 8Z" />,
  reset: (
    <>
      <path d="M4 10a8 8 0 1 1 1 8M4 4v6h6" />
    </>
  ),
  coffee: (
    <>
      <path d="M4 9h12v6a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5ZM16 10h2a3 3 0 0 1 0 6h-2M7 3v3M12 3v3" />
    </>
  ),
  paper: (
    <>
      <path d="M6 3h9l4 4v14H6ZM14 3v5h5M9 12h7M9 16h7" />
    </>
  ),
  code: (
    <>
      <path d="m7 7-5 5 5 5m10-10 5 5-5 5M14 4l-4 16" />
    </>
  ),
  search: (
    <>
      <circle cx="10" cy="10" r="6" />
      <path d="m15 15 6 6" />
    </>
  ),
  send: <path d="m3 3 19 9-19 9 4-9Zm4 9h15" />,
  compass: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m16 8-2 6-6 2 2-6Z" />
    </>
  ),
}
export default function WorldIcon({ name, size = 20, ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name] || paths.world}
    </svg>
  )
}
