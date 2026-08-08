import type { Camera } from '../mock/cameras'

interface Props {
  cameras: Camera[]
  activeId: string
  onSelect: (id: string) => void
  onAddClick: () => void
}

export default function CameraSidebar({ cameras, activeId, onSelect, onAddClick }: Props) {
  return (
    <aside className="cc-sidebar">
      <div className="cc-sidebar-label">CAMERAS</div>
      <ul className="cc-camera-list">
        {cameras.map((cam) => (
          <li key={cam.id}>
            <button
              className={`cc-camera-item${cam.id === activeId ? ' active' : ''}`}
              onClick={() => onSelect(cam.id)}
            >
              <span className={`status-dot${cam.online ? ' good' : ' critical'}`} />
              <span className="cc-camera-item-text">
                <span className="cc-camera-id">{cam.id}</span>
                <span className="cc-camera-zone">{cam.zone}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      <button className="cc-add-camera" onClick={onAddClick}>
        + Add Camera
      </button>
    </aside>
  )
}
