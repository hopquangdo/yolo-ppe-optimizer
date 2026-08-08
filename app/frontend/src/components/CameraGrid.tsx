import { useState } from 'react'
import { mockCameras } from '../mock/cameras'

// Placeholder stream — thay bằng URL HLS/RTSP-over-WebRTC thật từ edge inference service khi có.
const MOCK_STREAM_URL = 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4'

export default function CameraGrid() {
  const [selected, setSelected] = useState<string[]>(mockCameras.slice(0, 3).map((c) => c.id))

  function toggle(id: string) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]))
  }

  const selectedCameras = mockCameras.filter((c) => selected.includes(c.id))

  return (
    <section className="chart-card">
      <div className="chart-card-header">
        <h3>Camera trực tiếp</h3>
        <span className="camera-count">{selectedCameras.length} / {mockCameras.length} đang xem</span>
      </div>

      <div className="camera-chip-row">
        {mockCameras.map((cam) => (
          <button
            key={cam.id}
            className={`camera-chip${selected.includes(cam.id) ? ' active' : ''}${cam.online ? '' : ' offline'}`}
            onClick={() => toggle(cam.id)}
            aria-pressed={selected.includes(cam.id)}
          >
            <span className={`status-dot${cam.online ? ' good' : ' critical'}`} />
            {cam.name}
          </button>
        ))}
      </div>

      {selectedCameras.length === 0 ? (
        <p className="camera-empty">Chọn ít nhất 1 camera để xem stream.</p>
      ) : (
        <div className="camera-tile-grid">
          {selectedCameras.map((cam) => (
            <div key={cam.id} className="camera-tile">
              {cam.online ? (
                <>
                  <video src={MOCK_STREAM_URL} autoPlay loop muted playsInline />
                  <span className="live-badge">
                    <span className="live-dot" /> LIVE
                  </span>
                </>
              ) : (
                <div className="camera-offline">Mất kết nối</div>
              )}
              <div className="camera-tile-caption">
                <span>{cam.name}</span>
                <span className="camera-tile-zone">{cam.zone}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
