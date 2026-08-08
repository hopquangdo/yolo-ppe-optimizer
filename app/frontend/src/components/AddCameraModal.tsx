import { useState, type FormEvent } from 'react'
import type { Camera } from '../mock/cameras'

interface Props {
  onClose: () => void
  onAdd: (camera: Camera) => void
}

export default function AddCameraModal({ onClose, onAdd }: Props) {
  const [name, setName] = useState('')
  const [zone, setZone] = useState('')
  const [id, setId] = useState('')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!name.trim() || !zone.trim()) return
    onAdd({
      id: id.trim() || `CAM-${Math.floor(100 + Math.random() * 900)}`,
      name: name.trim(),
      zone: zone.trim(),
      online: true,
    })
    onClose()
  }

  return (
    <div className="cc-modal-backdrop" onClick={onClose}>
      <form className="cc-modal" onClick={(e) => e.stopPropagation()} onSubmit={handleSubmit}>
        <h3>Thêm camera mới</h3>

        <label className="cc-field">
          <span>Mã camera (tuỳ chọn)</span>
          <input value={id} onChange={(e) => setId(e.target.value)} placeholder="CAM-06" />
        </label>

        <label className="cc-field">
          <span>Tên vị trí</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Cổng phụ" required />
        </label>

        <label className="cc-field">
          <span>Khu vực</span>
          <input value={zone} onChange={(e) => setZone(e.target.value)} placeholder="Xưởng C" required />
        </label>

        <div className="cc-modal-actions">
          <button type="button" className="cc-btn-secondary" onClick={onClose}>
            Huỷ
          </button>
          <button type="submit" className="cc-btn-primary">
            Thêm camera
          </button>
        </div>
      </form>
    </div>
  )
}
