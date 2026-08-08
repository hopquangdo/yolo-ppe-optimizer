import type { Camera } from '../mock/cameras'
import { liveStats } from '../mock/commandCenter'

const MOCK_STREAM_URL = 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4'

interface Props {
  camera: Camera
}

export default function LiveCameraView({ camera }: Props) {
  return (
    <section className="chart-card cc-live-view">
      <div className="chart-card-header">
        <h3>LIVE CAMERA VIEW — {camera.name}</h3>
        <span className="camera-count">{camera.id}</span>
      </div>

      <div className="cc-video-frame">
        {camera.online ? (
          <>
            <video src={MOCK_STREAM_URL} autoPlay loop muted playsInline />
            <span className="live-badge">
              <span className="live-dot" /> LIVE
            </span>
            <div className="cc-detection-box" style={{ top: '28%', left: '38%', width: '22%', height: '52%' }}>
              <span className="cc-detection-tag person">👤 Person</span>
              <span className="cc-detection-tag violation">⚠ No Helmet</span>
            </div>
          </>
        ) : (
          <div className="camera-offline">Mất kết nối</div>
        )}
      </div>

      <div className="cc-live-stats">
        <span>FPS <strong>{liveStats.fps}</strong></span>
        <span>LATENCY <strong>{liveStats.latencyMs}ms</strong></span>
        <span>AI <strong>{liveStats.aiConfidence}%</strong></span>
        <span>{liveStats.resolution}</span>
      </div>
    </section>
  )
}
