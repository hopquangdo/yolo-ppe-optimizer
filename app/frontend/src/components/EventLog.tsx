import type { EventLogRow } from '../mock/commandCenter'

interface Props {
  data: EventLogRow[]
}

export default function EventLog({ data }: Props) {
  return (
    <section className="chart-card">
      <div className="chart-card-header">
        <h3>EVENT LOG</h3>
      </div>
      <table className="data-table cc-event-table">
        <thead>
          <tr>
            <th>Thời gian</th>
            <th>Camera</th>
            <th>Vi phạm</th>
            <th>Đối tượng</th>
            <th>Mức độ</th>
          </tr>
        </thead>
        <tbody>
          {data.map((e) => (
            <tr key={e.id}>
              <td>{e.time}</td>
              <td>{e.camera}</td>
              <td>{e.violation}</td>
              <td>{e.personId}</td>
              <td>
                <span className={`cc-severity-badge ${e.severity}`}>
                  {e.severity === 'critical' ? 'CRITICAL' : 'WARNING'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
