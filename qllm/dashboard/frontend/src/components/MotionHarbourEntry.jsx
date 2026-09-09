export default function MotionHarbourEntry({ studioId, hostname = globalThis.location?.hostname }) {
  // This prototype is a separate loopback service, not a publicly hosted game.
  if (studioId !== 'physics' || !['localhost', '127.0.0.1'].includes(hostname)) return null
  return (
    <div>
      <a className="rw-text-link" href="http://127.0.0.1:4178/">Explore Physics island →</a>
      <span className="rw-small-note">Classical mechanics · Unity · Local service required</span>
    </div>
  )
}
