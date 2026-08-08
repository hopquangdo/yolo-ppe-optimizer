import { mockEventLog } from '../mock/commandCenter'
import EventLog from '../components/EventLog'

export default function EventLogPage() {
  return (
    <div className="cc-page-top">
      <EventLog data={mockEventLog} />
    </div>
  )
}
