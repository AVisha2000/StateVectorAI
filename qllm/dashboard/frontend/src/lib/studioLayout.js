// Main furniture placement is shared by the visible room and its walk footprints.
// Y-up project frame; x/z are horizontal, turn is Three.js positive Y rotation.
const item = (id, value) => Object.freeze({ id, turn: 0, ...value });
export const STUDIO_LAYOUT = Object.freeze({
  table: item("table", { x: 0, z: .35, radius: .47 }),
  researcher: item("researcher", { x: -.78, z: .15, turn: .72, radius: .19 }),
  researcherChair: item("researcher-chair", { x: -.85, z: .07, turn: .7, width: .39, depth: .4025, offsetZ: -.02625 }),
  visitorChair: item("visitor-chair", { x: .88, z: .9, turn: -2.3, width: .39, depth: .4025, offsetZ: -.02625 }),
  // Reserve the side panel's space even before a result is pinned, so pinning
  // cannot materialize a collider around a visitor already standing there.
  board: item("board", { x: -1.12, z: -.85, turn: .22, width: 1.12, depth: .065,
    pinX: .9, pinWidth: .59, footprintWidth: 1.755, footprintDepth: .072, offsetX: .3175 }),
  lab: item("lab", { x: .94, z: -.85, radius: .62 * 1.02 }),
  bookcase: item("bookcase", { x: .4, z: -1.67, width: .85, depth: .23 }),
});
export const VISITOR_SPAWN = Object.freeze({ x: .83, z: 1.65 });
export const VISITOR_RADIUS = .11;
export const WALK_EXTENT = 2.05;

export const WALK_OBSTACLES = Object.freeze(Object.values(STUDIO_LAYOUT).map(part => {
  const ox = part.offsetX || 0, oz = part.offsetZ || 0;
  return Object.freeze({ ...part,
    width: part.footprintWidth ?? part.width,
    depth: part.footprintDepth ?? part.depth,
    x: part.x + Math.cos(part.turn)*ox + Math.sin(part.turn)*oz,
    z: part.z - Math.sin(part.turn)*ox + Math.cos(part.turn)*oz,
    padding: VISITOR_RADIUS,
  });
}));
