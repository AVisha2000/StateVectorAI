import { simulateStudio } from './studioSimulation.js'
self.onmessage = ({ data }) => {
  try {
    self.postMessage({ result: simulateStudio(data.id, data.parameter) })
  } catch (error) {
    self.postMessage({ error: error.message })
  }
}
