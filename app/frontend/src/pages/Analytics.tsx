import { mockAlerts, mockDetectionTimeline } from '../mock/commandCenter'
import DetectionTimeline from '../components/DetectionTimeline'
import LiveAlerts from '../components/LiveAlerts'

export default function Analytics() {
  return (
    <div className="cc-split-row cc-page-top">
      <DetectionTimeline data={mockDetectionTimeline} />
      <LiveAlerts data={mockAlerts} />
    </div>
  )
}
