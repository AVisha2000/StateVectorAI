import { useEffect, useRef, useState } from 'react'
import * as T from 'three'
import { TrackballControls } from 'three/addons/controls/TrackballControls.js'
import { buildPlanet, disposeScene } from './researchPlanetScene.js'

const Y_AXIS = new T.Vector3(0, 1, 0)
export default function ResearchPlanet({ studios, selectedId, onSelect, paused, reducedMotion, command }) {
  const hostRef = useRef(null),
    labelsRef = useRef(new Map()),
    stateRef = useRef({}),
    engineRef = useRef(null)
  const [failed, setFailed] = useState(false)
  stateRef.current = { paused, reducedMotion, onSelect, selectedId }
  useEffect(() => {
    const host = hostRef.current
    let renderer
    try {
      renderer = new T.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' })
    } catch {
      setFailed(true)
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    renderer.outputColorSpace = T.SRGBColorSpace
    renderer.toneMapping = T.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.17
    renderer.domElement.setAttribute(
      'aria-label',
      'Research planet. Drag freely in any direction, scroll to zoom, or use the controls and studio list.',
    )
    renderer.domElement.setAttribute('role', 'img')
    host.prepend(renderer.domElement)
    const scene = new T.Scene(),
      camera = new T.PerspectiveCamera(40, 1, 0.1, 80)
    camera.position.set(0, 0.65, 11.9)
    const controls = new TrackballControls(camera, renderer.domElement)
    controls.noPan = true
    controls.rotateSpeed = 2
    controls.zoomSpeed = 0.9
    controls.staticMoving = true
    controls.minDistance = 5.8
    controls.maxDistance = 17
    const planet = buildPlanet(studios)
    scene.add(planet.group)
    scene.add(new T.HemisphereLight('#e4efff', '#607e68', 1.6))
    const sun = new T.DirectionalLight('#ffe7c8', 2.8)
    sun.position.set(-4, 7, 6)
    scene.add(sun)
    const rim = new T.DirectionalLight('#9adbe3', 2)
    rim.position.set(5, 0, -4)
    scene.add(rim)
    const raycaster = new T.Raycaster(),
      pointer = new T.Vector2(),
      projected = new T.Vector3(),
      worldPoint = new T.Vector3(),
      normal = new T.Vector3(),
      eye = new T.Vector3()
    let width = 1,
      height = 1,
      elapsed = 0,
      previous = 0,
      active = true,
      intersecting = true,
      interacted = false,
      transition = null,
      pointerStart = null,
      homeDistance = 11.9
    const resize = () => {
      width = host.clientWidth
      height = host.clientHeight
      if (!width || !height) return
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
      renderer.setSize(width, height)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      controls.handleResize()
      homeDistance = Math.max(11.9, 3.95 / (Math.tan(T.MathUtils.degToRad(20)) * camera.aspect))
      controls.maxDistance = Math.max(17, homeDistance * 1.2)
      if (!interacted && !transition) camera.position.set(0, 0.65, homeDistance)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(host)
    resize()
    window.addEventListener('resize', resize)
    const start = () => {
      interacted = true
      transition = null
    }
    controls.addEventListener('start', start)
    const down = (event) => {
      pointerStart = [event.clientX, event.clientY]
    }
    const up = (event) => {
      if (!pointerStart || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5)
        return
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(pointer, camera)
      const hit = raycaster.intersectObject(planet.group, true)[0],
        studio = studios.find((studio) => studio.id === hit?.object.userData.studioId)
      if (studio) stateRef.current.onSelect(studio)
    }
    renderer.domElement.addEventListener('pointerdown', down)
    renderer.domElement.addEventListener('pointerup', up)
    const visibility = () => {
      active = !document.hidden && intersecting
    }
    document.addEventListener('visibilitychange', visibility)
    const intersection = new IntersectionObserver(([entry]) => {
      intersecting = entry.isIntersecting
      visibility()
    })
    intersection.observe(host)
    const lost = (event) => {
      event.preventDefault()
      setFailed(true)
    }
    renderer.domElement.addEventListener('webglcontextlost', lost)
    engineRef.current = {
      stopAutoRotation() {
        interacted = true
      },
      command(action) {
        interacted = true
        let position = camera.position.clone(),
          target = new T.Vector3(),
          up = camera.up.clone()
        if (action.type === 'focus') {
          const entry = planet.anchors.get(action.id)
          if (!entry) return
          const outward = entry.normal.clone().applyQuaternion(planet.group.quaternion)
          position = outward.clone().multiplyScalar(9.4)
          target = outward.clone().multiplyScalar(0.75)
          up = new T.Vector3(0, 1, 0)
          if (Math.abs(outward.dot(up)) > 0.9) up.set(0, 0, -1)
        } else if (action.type === 'reset') {
          position.set(0, 0.65, homeDistance)
          up.set(0, 1, 0)
        } else if (action.type === 'zoom') {
          position.multiplyScalar(action.direction > 0 ? 0.85 : 1.18)
          position.clampLength(6, 17)
        } else if (action.type === 'rotate') {
          position.applyAxisAngle(
            action.axis === 'vertical' ? new T.Vector3(1, 0, 0) : Y_AXIS,
            action.direction * 0.3,
          )
          target.copy(controls.target)
        }
        transition = {
          start: performance.now(),
          from: camera.position.clone(),
          to: position,
          fromTarget: controls.target.clone(),
          target,
          fromUp: camera.up.clone(),
          up,
        }
      },
    }
    renderer.setAnimationLoop((now) => {
      const dt = Math.min((now - previous) / 1000 || 0, 0.05)
      previous = now
      if (!active) return
      const state = stateRef.current
      if (!state.paused && !state.reducedMotion) {
        elapsed += dt
        planet.animate(elapsed)
        if (!interacted && !transition) camera.position.applyAxisAngle(Y_AXIS, dt * 0.015)
      }
      if (transition) {
        const fraction = state.reducedMotion ? 1 : Math.min((now - transition.start) / 850, 1),
          ease = fraction * fraction * (3 - 2 * fraction)
        camera.position.lerpVectors(transition.from, transition.to, ease)
        controls.target.lerpVectors(transition.fromTarget, transition.target, ease)
        camera.up.lerpVectors(transition.fromUp, transition.up, ease).normalize()
        camera.lookAt(controls.target)
        if (fraction === 1) transition = null
      } else controls.update()
      camera.updateMatrixWorld()
      planet.group.updateMatrixWorld(true)
      for (const [id, entry] of planet.anchors) {
        const label = labelsRef.current.get(id)
        if (!label) continue
        entry.pin.getWorldPosition(worldPoint)
        normal.copy(entry.normal).applyQuaternion(planet.group.quaternion)
        eye.copy(camera.position).sub(worldPoint).normalize()
        projected.copy(worldPoint).project(camera)
        const visible = normal.dot(eye) > 0.14 && Math.abs(projected.x) < 0.92 && Math.abs(projected.y) < 0.91
        const labelX = Math.max(77, Math.min(width - 77, (projected.x * 0.5 + 0.5) * width))
        const labelY = Math.max(
          28,
          (-projected.y * 0.5 + 0.5) * height - (id === 'machine-learning' ? height * 0.095 : 0),
        )
        label.style.transform = `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`
        label.style.visibility = visible ? 'visible' : 'hidden'
        label.style.opacity = visible ? '1' : '0'
        label.tabIndex = visible ? 0 : -1
      }
      host.dataset.camera = `${camera.position.x.toFixed(2)},${camera.position.y.toFixed(2)},${camera.position.z.toFixed(2)}`
      renderer.render(scene, camera)
    })
    return () => {
      engineRef.current = null
      renderer.setAnimationLoop(null)
      observer.disconnect()
      intersection.disconnect()
      controls.removeEventListener('start', start)
      controls.dispose()
      renderer.domElement.removeEventListener('pointerdown', down)
      renderer.domElement.removeEventListener('pointerup', up)
      renderer.domElement.removeEventListener('webglcontextlost', lost)
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('resize', resize)
      disposeScene(scene)
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [studios])
  useEffect(() => {
    if (command) engineRef.current?.command(command)
  }, [command])
  return (
    <div
      ref={hostRef}
      className="planet-canvas"
      data-agent-motion={paused || reducedMotion ? 'paused' : 'active'}
    >
      {failed ? (
        <div className="planet-unavailable" role="status">
          The 3D view is unavailable on this device. Every studio is still accessible from the list below.
        </div>
      ) : (
        studios.map((studio) => (
          <button
            key={studio.id}
            ref={(element) => {
              if (element) labelsRef.current.set(studio.id, element)
              else labelsRef.current.delete(studio.id)
            }}
            className={`planet-label ${selectedId === studio.id ? 'is-selected' : ''}`}
            style={{ '--studio-color': studio.color }}
            onPointerEnter={() => engineRef.current?.stopAutoRotation()}
            onFocus={() => engineRef.current?.stopAutoRotation()}
            onClick={() => onSelect(studio)}
            aria-label={studio.kind === 'quantum-game' ? 'Visit Quantum computing island' : `Visit ${studio.name}, ${studio.label} studio`}
          >
            <span className="planet-label-dot" />
            <span>{studio.short}</span>
            <span className="planet-label-arrow">↗</span>
          </button>
        ))
      )}
    </div>
  )
}
