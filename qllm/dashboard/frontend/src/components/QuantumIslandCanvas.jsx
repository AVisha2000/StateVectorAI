import { useEffect, useRef, useState } from 'react'
import * as T from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { buildQuantumIslandScene } from './quantumIslandScene.js'
import { disposeScene } from './researchPlanetScene.js'

export default function QuantumIslandCanvas({ presentation }) {
  const host = useRef(null), engine = useRef(null), latest = useRef(presentation)
  const [failed, setFailed] = useState(false)
  latest.current = presentation
  useEffect(() => {
    const element = host.current
    let renderer
    try { renderer = new T.WebGLRenderer({ antialias: true, powerPreference: 'low-power' }) }
    catch { setFailed(true); return }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    renderer.outputColorSpace = T.SRGBColorSpace
    renderer.toneMapping = T.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1
    renderer.domElement.setAttribute('role', 'img')
    renderer.domElement.setAttribute('aria-label', 'Quantum island with Bloch sphere. Mint dot is your state, amber diamond is the target. Drag to orbit the view.')
    element.append(renderer.domElement)
    const model = buildQuantumIslandScene()
    const camera = new T.PerspectiveCamera(42, 1, 0.1, 60)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 2, 0)
    controls.enablePan = false; controls.enableZoom = false
    controls.minPolarAngle = Math.PI / 6; controls.maxPolarAngle = Math.PI / 2.1
    const draw = () => renderer.render(model.scene, camera)
    const resize = () => {
      const width = element.clientWidth, height = element.clientHeight
      if (!width || !height) return
      renderer.setSize(width, height)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      const distance = camera.aspect < 1 ? 10.6 : 9.4
      camera.position.set(0.48, 0.28, 0.82).normalize().multiplyScalar(distance).add(controls.target)
      controls.update(); draw()
    }
    controls.addEventListener('change', draw)
    const lost = (event) => { event.preventDefault(); setFailed(true) }
    renderer.domElement.addEventListener('webglcontextlost', lost)
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    model.update(latest.current)
    resize()
    engine.current = { update: (next) => { model.update(next); draw() } }
    return () => {
      engine.current = null
      observer.disconnect()
      controls.removeEventListener('change', draw)
      controls.dispose()
      renderer.domElement.removeEventListener('webglcontextlost', lost)
      disposeScene(model.scene)
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])
  useEffect(() => { engine.current?.update(presentation) }, [presentation])
  return <div className="qg-canvas" ref={host} data-renderer={failed ? 'unavailable' : 'webgl'}>
    {failed && <div className="qg-scene-fallback" role="status">The 3D view is unavailable. Pulse controls, state coordinates and your notebook still work.</div>}
  </div>
}
