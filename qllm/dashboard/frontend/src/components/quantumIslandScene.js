import * as T from 'three'

// Bloch (x,y,z) -> world (x,z,-y): a right-handed frame with |0> at the top.
export const blochPoint = (r, radius = 1.35) => new T.Vector3(r[0], r[2], -r[1]).multiplyScalar(radius)
const CENTER = new T.Vector3(0, 2.6, 0)

export function buildQuantumIslandScene() {
  const scene = new T.Scene()
  scene.background = new T.Color('#10252c')
  scene.fog = new T.FogExp2('#10252c', 0.022)
  scene.add(new T.HemisphereLight('#cbf4ec', '#213c45', 2.4))
  const sun = new T.DirectionalLight('#ffdbb2', 3.2)
  sun.position.set(-3, 7, 4)
  scene.add(sun)
  const rim = new T.DirectionalLight('#86a8ef', 2.5)
  rim.position.set(3, 4, -4)
  scene.add(rim)
  const add = (geometry, color, position, extra = {}) => {
    const object = new T.Mesh(geometry, new T.MeshStandardMaterial({ color, roughness: 0.75, ...extra }))
    object.position.set(...position)
    scene.add(object)
    return object
  }
  add(new T.CylinderGeometry(2.55, 2.1, 0.65, 9), '#557879', [0, -0.03, 0], { flatShading: true })
  add(new T.CylinderGeometry(2.5, 2.5, 0.12, 64), '#abc0aa', [0, 0.32, 0])
  add(new T.CylinderGeometry(1.83, 1.88, 0.1, 64), '#25454d', [0, 0.43, 0])
  add(new T.CylinderGeometry(0.63, 0.83, 0.25, 32), '#718e8f', [0, 0.58, 0], { metalness: 0.5 })
  const ring = (radius, color, y, thickness = 0.018) => {
    const object = add(new T.TorusGeometry(radius, thickness, 8, 80), color, [0, y, 0], { emissive: color, emissiveIntensity: 0.28 })
    object.rotation.x = Math.PI / 2
    return object
  }
  ring(1.8, '#81bcaf', 0.5)
  ring(0.59, '#9ff0d6', 0.72)
  for (let i = 0; i < 12; i++) {
    const a = i * Math.PI / 6
    const tick = add(new T.BoxGeometry(0.025, 0.02, 0.16), '#a9c8be', [Math.sin(a) * 1.6, 0.493, Math.cos(a) * 1.6])
    tick.rotation.y = a
  }
  // The four pylons and crystals are scenery; they never encode measurements.
  for (let i = 0; i < 4; i++) {
    const a = Math.PI / 4 + i * Math.PI / 2, x = Math.cos(a) * 2.12, z = Math.sin(a) * 2.12
    add(new T.CylinderGeometry(0.11, 0.16, 0.72, 8), '#5d7f81', [x, 0.74, z])
    add(new T.OctahedronGeometry(0.16), '#b6a4e3', [x, 1.2, z], { emissive: '#7a62ad', emissiveIntensity: 0.35 })
  }
  for (let i = 0; i < 6; i++) {
    const a = 0.25 + i * 1.04, radius = 4.1 + (i % 2) * 1.8
    const rock = add(new T.DodecahedronGeometry(0.45 + (i % 3) * 0.13), '#31555b', [Math.cos(a) * radius, -0.2, Math.sin(a) * radius], { flatShading: true })
    rock.scale.set(1.2, 0.8, 1)
  }
  const sphere = new T.Group()
  sphere.position.copy(CENTER)
  scene.add(sphere)
  for (let i = 0; i < 3; i++) {
    const geometry = new T.TorusGeometry(1.35, 0.007, 6, 96)
    const object = new T.Mesh(geometry, new T.MeshBasicMaterial({ color: '#81a5b4', transparent: true, opacity: 0.65 }))
    if (i === 1) object.rotation.y = Math.PI / 2
    if (i === 2) object.rotation.x = Math.PI / 2
    sphere.add(object)
  }
  for (const latitude of [-0.675, 0.675]) {
    const circle = new T.Mesh(new T.TorusGeometry(Math.sqrt(1.35 ** 2 - latitude ** 2), 0.004, 6, 80), new T.MeshBasicMaterial({ color: '#628896', transparent: true, opacity: 0.4 }))
    circle.rotation.x = Math.PI / 2
    circle.position.y = latitude
    sphere.add(circle)
  }
  const label = (text, position, color = '#d7e6df') => {
    const canvas = document.createElement('canvas')
    canvas.width = 256; canvas.height = 96
    const context = canvas.getContext('2d')
    context.fillStyle = color
    context.font = '500 48px system-ui'
    context.textAlign = 'center'; context.textBaseline = 'middle'
    context.fillText(text, 128, 48)
    const texture = new T.CanvasTexture(canvas)
    texture.colorSpace = T.SRGBColorSpace
    const sprite = new T.Sprite(new T.SpriteMaterial({ map: texture, depthTest: false }))
    sprite.position.copy(position)
    sprite.scale.set(0.68, 0.255, 1)
    sphere.add(sprite)
  }
  label('|0⟩', new T.Vector3(0, 1.63, 0))
  label('|1⟩', new T.Vector3(0, -1.56, 0))
  label('+X', new T.Vector3(1.66, 0, 0))
  label('+Y', new T.Vector3(0, 0, -1.66))
  const axes = new T.BufferGeometry().setFromPoints([
    new T.Vector3(-1.45, 0, 0), new T.Vector3(1.45, 0, 0),
    new T.Vector3(0, -1.45, 0), new T.Vector3(0, 1.45, 0),
    new T.Vector3(0, 0, -1.45), new T.Vector3(0, 0, 1.45),
  ])
  sphere.add(new T.LineSegments(axes, new T.LineBasicMaterial({ color: '#7194a0', transparent: true, opacity: 0.35 })))
  const arrow = new T.ArrowHelper(new T.Vector3(0, 1, 0), new T.Vector3(), 1.35, '#a4f4d6', 0.14, 0.085)
  sphere.add(arrow)
  const marker = new T.Mesh(new T.SphereGeometry(0.07, 20, 12), new T.MeshBasicMaterial({ color: '#b2ffe1' }))
  sphere.add(marker)
  const target = new T.Mesh(new T.OctahedronGeometry(0.12), new T.MeshBasicMaterial({ color: '#f4bb84', wireframe: true }))
  sphere.add(target)
  const makeLine = (color, opacity) => {
    const line = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial({ color, transparent: true, opacity }))
    sphere.add(line)
    return line
  }
  const trail = makeLine('#a4f4d6', 0.94), comparison = makeLine('#c0a7f0', 0.65)
  const updateLine = (line, points) => {
    line.geometry.dispose()
    line.geometry = new T.BufferGeometry().setFromPoints(points.map((point) => blochPoint(point)))
    line.visible = points.length > 1
  }
  return {
    scene,
    update({ state, points, target: targetVector, ghost = [] }) {
      const position = blochPoint(state)
      marker.position.copy(position)
      arrow.setDirection(position.clone().normalize())
      target.position.copy(blochPoint(targetVector))
      updateLine(trail, points)
      updateLine(comparison, ghost)
    },
  }
}
