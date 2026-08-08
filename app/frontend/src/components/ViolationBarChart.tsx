import { useState } from 'react'
import type { ViolationStats } from '../api/client'

const CATEGORICAL = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)', 'var(--series-5)']

const LABELS: Record<string, string> = {
  no_helmet: 'Không mũ bảo hộ',
  no_vest: 'Không áo phản quang',
  no_mask: 'Không khẩu trang',
  no_gloves: 'Không găng tay',
  no_boots: 'Không giày bảo hộ',
}

interface Props {
  data: ViolationStats[]
}

export default function ViolationBarChart({ data }: Props) {
  const [hovered, setHovered] = useState<number | null>(null)
  const [showTable, setShowTable] = useState(false)
  const max = Math.max(...data.map((d) => d.count), 1)
  const barHeight = 28
  const gap = 12
  const chartHeight = data.length * (barHeight + gap)

  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h3>Vi phạm theo loại</h3>
        <button className="table-toggle" onClick={() => setShowTable((s) => !s)}>
          {showTable ? 'Xem biểu đồ' : 'Xem bảng'}
        </button>
      </div>

      {showTable ? (
        <table className="data-table">
          <thead>
            <tr>
              <th>Loại vi phạm</th>
              <th>Số lượng</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d) => (
              <tr key={d.violation_type}>
                <td>{LABELS[d.violation_type] ?? d.violation_type}</td>
                <td>{d.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <svg width="100%" height={chartHeight} role="img" aria-label="Biểu đồ vi phạm theo loại">
          {data.map((d, i) => {
            const width = (d.count / max) * 100
            const y = i * (barHeight + gap)
            const isHovered = hovered === i
            return (
              <g key={d.violation_type} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
                <text x="0" y={y + barHeight / 2 + 4} className="bar-label" fontSize="12">
                  {LABELS[d.violation_type] ?? d.violation_type}
                </text>
                <rect
                  x="42%"
                  y={y}
                  width={`${width * 0.58}%`}
                  height={barHeight}
                  rx={4}
                  fill={CATEGORICAL[i % CATEGORICAL.length]}
                  opacity={isHovered ? 1 : 0.9}
                />
                <text x={`${42 + width * 0.58}%`} y={y + barHeight / 2 + 4} className="bar-value" fontSize="12" dx={6}>
                  {d.count}
                </text>
                {isHovered && (
                  <title>
                    {LABELS[d.violation_type] ?? d.violation_type}: {d.count} vi phạm
                  </title>
                )}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
