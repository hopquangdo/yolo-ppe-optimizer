import type { Alert } from '../mock/commandCenter'

const SEVERITY_DOT: Record<Alert['severity'], string> = {
  critical: '🔴',
  warning: '🟠',
  info: '🟡',
}

interface Props {
  data: Alert[]
}

export default function LiveAlerts({ data }: Props) {
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h3>LIVE ALERTS</h3>
      </div>
      <ul className="cc-alert-list">
        {data.map((a) => (
          <li key={a.id} className={`cc-alert-item ${a.severity}`}>
            <span>{SEVERITY_DOT[a.severity]}</span>
            <span className="cc-alert-time">{a.time}</span>
            <span className="cc-alert-camera">{a.camera}</span>
            <span className="cc-alert-type">{a.type}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
