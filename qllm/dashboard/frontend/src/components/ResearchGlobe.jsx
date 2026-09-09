import { useEffect, useMemo, useRef, useState } from 'react'
import Globe from 'react-globe.gl'
import * as THREE from 'three'
import { agentMotionTimeAt, agentStrideAt, cappedPixelRatio, circlePatch, deferredCleanup, setAgentMotionPaused } from '../lib/researchWorld.js'

function agentObject(agent) {
  const group = new THREE.Group()
  const regionColor = agent.color || '#ffffff'
  const body = new THREE.Mesh(new THREE.ConeGeometry(1.8, 4.8, 6), new THREE.MeshStandardMaterial({ color: '#102033', emissive: regionColor, emissiveIntensity: 0.35, roughness: 0.7 }))
  body.position.y = 2.5
  const head = new THREE.Mesh(new THREE.SphereGeometry(1.15, 14, 10), new THREE.MeshStandardMaterial({ color: '#f2f7ff', roughness: 0.85 }))
  head.position.y = 5.65
  const ring = new THREE.Mesh(new THREE.TorusGeometry(2.25, 0.2, 6, 24), new THREE.MeshBasicMaterial({ color: agent.statusColor || '#a9b8c8', transparent: true, opacity: 0.72 }))
  ring.rotation.x = Math.PI / 2
  ring.position.y = 0.12
  group.add(ring, body, head)
  group.userData.agentId = agent.id
  group.userData.body = body
  group.userData.latitude = agent.latitude
  group.userData.ring = ring
  group.userData.statusColor = agent.statusColor
  group.userData.motionClock = { paused: false, pausedAt: null, pausedDuration: 0 }
  return group
}

export default function ResearchGlobe({ agents, regions, onSelect, selectedId, paused, reducedMotion, focusRegion, zoomLevel }) {
  const globeRef = useRef()
  const frameRef = useRef()
  const intersectingRef = useRef(true)
  const [size, setSize] = useState({ width: 760, height: 520 })
  const data = useMemo(() => agents.map((agent) => ({ ...agent, color: regions.find((region) => region.id === agent.regionId)?.color })), [agents, regions])
  const globeMaterial = useMemo(() => new THREE.MeshPhongMaterial({ color: '#14304f', shininess: 9, specular: '#547ca2', transparent: true, opacity: 0.98 }), [])
  const agentObjects = useMemo(() => new Map(data.map((agent) => [agent.id, agentObject(agent)])), [data])
  const objectCleanup = useMemo(() => deferredCleanup(() => { for (const object of agentObjects.values()) disposeObject(object) }), [agentObjects])
  const materialCleanup = useMemo(() => deferredCleanup(() => globeMaterial.dispose()), [globeMaterial])
  const regionPatches = useMemo(() => regions.map((region) => ({ ...region, polygon: circlePatch(region.latitude, region.longitude, region.radius) })), [regions])
  const questionAgents = useMemo(() => data.filter((agent) => agent.has_question), [data])

  useEffect(() => {
    const element = frameRef.current
    if (!element) return undefined
    const update = () => {
      const next = { width: Math.max(280, Math.floor(element.clientWidth)), height: Math.max(420, Math.floor(element.clientWidth * 0.68)) }
      setSize((current) => current.width === next.width && current.height === next.height ? current : next)
      globeRef.current?.renderer?.().setPixelRatio(cappedPixelRatio(globalThis.devicePixelRatio))
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    window.addEventListener('resize', update)
    return () => { observer.disconnect(); window.removeEventListener('resize', update) }
  }, [])

  useEffect(() => {
    if (!globeRef.current) return
    const target = focusRegion || { latitude: 15, longitude: 15 }
    globeRef.current.pointOfView({ lat: target.latitude, lng: target.longitude, altitude: zoomLevel }, 0)
  }, [focusRegion, zoomLevel])

  useEffect(() => {
    const controls = globeRef.current?.controls?.()
    if (!controls) return undefined
    controls.autoRotate = !paused && !reducedMotion
    controls.autoRotateSpeed = 0.25
    return () => { controls.autoRotate = false }
  }, [paused, reducedMotion])

  useEffect(() => {
    const scene = globeRef.current?.scene?.()
    if (!scene) return undefined
    const previous = scene.onBeforeRender
    const updateAgentMotion = (...args) => {
      previous?.apply(scene, args)
      const now = animationNow()
      for (const object of agentObjects.values()) updateAgentPose(object, now)
      scene.updateMatrixWorld(true)
    }
    scene.onBeforeRender = updateAgentMotion
    return () => {
      if (scene.onBeforeRender === updateAgentMotion) scene.onBeforeRender = previous
    }
  }, [agentObjects])

  useEffect(() => {
    const element = frameRef.current
    if (!element || !globeRef.current) return undefined
    const syncAnimation = () => {
      if (!document.hidden && intersectingRef.current) globeRef.current?.resumeAnimation()
      else globeRef.current?.pauseAnimation()
    }
    document.addEventListener('visibilitychange', syncAnimation)
    const observer = new IntersectionObserver(([entry]) => {
      intersectingRef.current = entry.isIntersecting
      syncAnimation()
    })
    observer.observe(element)
    syncAnimation()
    return () => { document.removeEventListener('visibilitychange', syncAnimation); observer.disconnect() }
  }, [])

  useEffect(() => {
    objectCleanup.mount()
    return () => objectCleanup.unmount()
  }, [objectCleanup])

  useEffect(() => {
    materialCleanup.mount()
    return () => materialCleanup.unmount()
  }, [materialCleanup])

  const markCanvasAccessible = () => {
    const renderer = globeRef.current?.renderer?.()
    renderer?.setPixelRatio(cappedPixelRatio(globalThis.devicePixelRatio))
    const canvas = renderer?.domElement
    canvas?.setAttribute('role', 'img')
    canvas?.setAttribute('aria-label', `Illustrated globe showing ${agents.length} research agents across ${regions.map((region) => region.label).join(', ') || 'no public regions'}`)
  }

  useEffect(() => {
    const now = animationNow()
    for (const [agentId, object] of agentObjects) {
      object.userData.motionClock = setAgentMotionPaused(object.userData.motionClock, paused || reducedMotion, now)
      const isSelected = agentId === selectedId
      object.userData.ring.scale.setScalar(isSelected ? 1.45 : 1)
      object.userData.ring.material.color.set(object.userData.statusColor || '#a9b8c8')
      object.userData.ring.material.opacity = isSelected ? 1 : 0.72
    }
  }, [agentObjects, paused, reducedMotion, selectedId])

  return (
    <div className="research-globe-canvas" ref={frameRef} data-agent-motion={paused || reducedMotion ? 'paused' : 'walking'}>
      <Globe
        ref={globeRef}
        width={size.width}
        height={size.height}
        animateIn={false}
        backgroundColor="rgba(0,0,0,0)"
        globeMaterial={globeMaterial}
        showGraticules
        atmosphereColor="#5da9ff"
        atmosphereAltitude={0.16}
        onGlobeReady={markCanvasAccessible}
        polygonsData={regionPatches}
        polygonGeoJsonGeometry="polygon"
        polygonCapColor="color"
        polygonSideColor={() => 'rgba(255,255,255,0.08)'}
        polygonAltitude={() => 0.012}
        polygonsTransitionDuration={0}
        objectsData={data}
        objectThreeObject={(agent) => agentObjects.get(agent.id)}
        objectLat="latitude"
        objectLng="longitude"
        objectAltitude={0.04}
        onObjectClick={onSelect}
        labelsData={regions}
        labelLat="latitude"
        labelLng="longitude"
        labelText="label"
        labelColor={() => '#edf6ff'}
        labelSize={() => 0.65}
        labelDotRadius={() => 0.16}
        labelResolution={2}
        labelsTransitionDuration={0}
        htmlElementsData={questionAgents}
        htmlLat="latitude"
        htmlLng="longitude"
        htmlAltitude={() => 0.08}
        htmlElement={() => questionBubble()}
        htmlTransitionDuration={0}
      />
    </div>
  )
}

function disposeObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose()
    if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose())
    else child.material?.dispose()
  })
}

function updateAgentPose(object, now) {
  const clock = object.userData.motionClock
  if (clock.paused) return
  const step = agentStrideAt(agentMotionTimeAt(clock, now), object.userData.latitude)
  object.userData.body.position.y = 2.5 + Math.abs(step) * 0.3
  object.rotation.y = step * 0.14
}

function questionBubble() {
  const element = document.createElement('span')
  element.className = 'research-question-bubble'
  element.setAttribute('aria-hidden', 'true')
  element.textContent = '?'
  return element
}

function animationNow() { return globalThis.performance?.now?.() ?? Date.now() }
