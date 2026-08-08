import { useState } from 'react'
import { mockCameras, type Camera } from '../mock/cameras'
import CameraSidebar from '../components/CameraSidebar'
import LiveCameraView from '../components/LiveCameraView'
import AiStatusRow from '../components/AiStatusRow'
import AddCameraModal from '../components/AddCameraModal'

export default function LiveView() {
  const [cameras, setCameras] = useState<Camera[]>(mockCameras)
  const [activeId, setActiveId] = useState(mockCameras[0].id)
  const [showAddModal, setShowAddModal] = useState(false)
  const activeCamera = cameras.find((c) => c.id === activeId) ?? cameras[0]

  function handleAddCamera(camera: Camera) {
    setCameras((prev) => [...prev, camera])
    setActiveId(camera.id)
  }

  return (
    <>
      <div className="cc-main cc-page-top">
        <CameraSidebar
          cameras={cameras}
          activeId={activeId}
          onSelect={setActiveId}
          onAddClick={() => setShowAddModal(true)}
        />
        <LiveCameraView camera={activeCamera} />
      </div>

      <AiStatusRow />

      {showAddModal && <AddCameraModal onClose={() => setShowAddModal(false)} onAdd={handleAddCamera} />}
    </>
  )
}
