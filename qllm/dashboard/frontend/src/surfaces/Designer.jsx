import { PageHeader, Scaffold } from '../lib/ui.jsx'

export default function Designer() {
  return (
    <>
      <PageHeader
        title="Designer"
        sub="Zoom from the whole field down to the gates. Classical slots show the network; quantum slots show the circuit. Built for people who aren't circuit experts."
      />
      <Scaffold
        phase="Phase 5 — Designer (highest-risk, de-risk first)"
        blurb="A semantic-zoom builder continuous with the Atlas. The renderer splits into a reusable read-only preview and a from-scratch/OSS interactive editor (no drawer provides live editing)."
        bullets={[
          'Semantic zoom: ML › LLMs › Attention › Encoder › model (new curated ontology)',
          'Read-only preview via server-rendered qml.draw_mpl → SVG',
          'Interactive gate editor (OSS JS composer or from-scratch SVG)',
          'Classical ↔ quantum toggle; round-trips to registry.py so a built circuit runs on the Bench',
          'Community circuit stream is a parking-lot item (no confirmed source yet)',
        ]}
      />
    </>
  )
}
