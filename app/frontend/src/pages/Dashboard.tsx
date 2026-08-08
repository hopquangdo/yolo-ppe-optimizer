import { useEffect, useState } from 'react'
import { api, type Violation, type ViolationStats } from '../api/client'
import { mockViolations, mockStatsByType, mockDailyCounts } from '../mock/violations'
import StatTile from '../components/StatTile'
import ViolationBarChart from '../components/ViolationBarChart'
import DailyTrendChart from '../components/DailyTrendChart'
import CameraGrid from '../components/CameraGrid'

// USE_MOCK=true trong lúc backend/DB chưa sẵn sàng — chuyển sang gọi api.* thật khi có.
const USE_MOCK = true

const LABELS: Record<string, string> = {
  no_helmet: 'Không mũ bảo hộ',
  no_vest: 'Không áo phản quang',
  no_mask: 'Không khẩu trang',
  no_gloves: 'Không găng tay',
  no_boots: 'Không giày bảo hộ',
}

export default function Dashboard() {
  const [stats, setStats] = useState<ViolationStats[]>([])
  const [recent, setRecent] = useState<Violation[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (USE_MOCK) {
      setStats(mockStatsByType)
      setRecent(mockViolations)
      return
    }
    Promise.all([api.getStatsByType(), api.getViolations()])
      .then(([statsRes, violationsRes]) => {
        setStats(statsRes)
        setRecent(violationsRes)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) return <p className="error-text">Lỗi tải dữ liệu: {error}</p>

  const total = recent.length
  const last24h = recent.filter((v) => Date.now() - new Date(v.created_at).getTime() < 24 * 3600_000).length
  const topZone = mostFrequent(recent.map((v) => v.zone))
  const topType = stats[0]

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>PPE Violation Dashboard</h1>
        {USE_MOCK && <span className="mock-badge">Mock data</span>}
      </header>

      <section className="stat-row">
        <StatTile label="Tổng vi phạm (7 ngày)" value={String(total)} />
        <StatTile label="Trong 24h qua" value={String(last24h)} />
        <StatTile label="Khu vực nhiều vi phạm nhất" value={topZone ?? '–'} />
        <StatTile
          label="Loại vi phạm phổ biến nhất"
          value={topType ? (LABELS[topType.violation_type] ?? topType.violation_type) : '–'}
          hint={topType ? `${topType.count} lần` : undefined}
        />
      </section>

      <CameraGrid />

      <section className="chart-row">
        <DailyTrendChart data={mockDailyCounts} />
        <ViolationBarChart data={stats} />
      </section>

      <section className="chart-card">
        <div className="chart-card-header">
          <h3>Vi phạm gần đây</h3>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Thời gian</th>
              <th>Khu vực</th>
              <th>Loại vi phạm</th>
              <th>Camera</th>
              <th>Độ tin cậy</th>
            </tr>
          </thead>
          <tbody>
            {recent.slice(0, 15).map((v) => (
              <tr key={v.id}>
                <td>{new Date(v.created_at).toLocaleString('vi-VN')}</td>
                <td>{v.zone}</td>
                <td>{LABELS[v.violation_type] ?? v.violation_type}</td>
                <td>{v.camera_id}</td>
                <td>{(v.confidence * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

function mostFrequent(values: string[]): string | undefined {
  if (values.length === 0) return undefined
  const counts = new Map<string, number>()
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1)
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0]
}
