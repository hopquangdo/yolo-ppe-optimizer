import { useEffect, useState } from 'react'

export default function CommandHeader() {
  const [now, setNow] = useState(new Date())

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="cc-header">
      <div className="cc-header-title">
        <span className="cc-shield">🛡</span> AI VISION COMMAND CENTER
      </div>
      <div className="cc-header-status">
        <span className="status-dot good pulse" />
        SYSTEM ONLINE
        <span className="cc-clock">{now.toLocaleTimeString('vi-VN', { hour12: false })}</span>
      </div>
    </header>
  )
}
