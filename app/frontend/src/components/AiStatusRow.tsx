import { aiStatus } from '../mock/commandCenter'
import StatTile from './StatTile'

export default function AiStatusRow() {
  return (
    <section>
      <div className="cc-section-label">AI STATUS</div>
      <div className="stat-row">
        <StatTile label="👤 Persons" value={String(aiStatus.persons)} />
        <StatTile label="🪖 Compliant" value={String(aiStatus.compliant)} />
        <StatTile label="⚠ Violations" value={String(aiStatus.violations)} />
        <StatTile label="🚨 Critical" value={String(aiStatus.critical)} />
      </div>
    </section>
  )
}
