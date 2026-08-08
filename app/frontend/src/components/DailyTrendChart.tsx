import { useState } from 'react'
import type { DailyCount } from '../mock/violations'

interface Props {
  data: DailyCount[]
}

const WIDTH = 600
const HEIGHT = 200
const PAD = 32

export default function DailyTrendChart({ data }: Props) {
  const [hovered, setHovered] = useState<number | null>(null)
  const max = Math.max(...data.map((d) => d.count), 1)
  const stepX = (WIDTH - PAD * 2) / (data.length - 1)

  const points = data.map((d, i) => {
    const x = PAD + i * stepX
    const y = HEIGHT - PAD - (d.count / max) * (HEIGHT - PAD * 2)
    return { x, y, ...d }
  })

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')

  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h3>Vi phạm 7 ngày gần nhất</h3>
      </div>
      <svg width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Xu hướng vi phạm 7 ngày">
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} className="axis-line" />
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinecap="round" />
        {points.map((p, i) => (
          <g key={p.date} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
            <circle cx={p.x} cy={p.y} r={hovered === i ? 6 : 4} fill="var(--series-1)" />
            <rect x={p.x - stepX / 2} y={0} width={stepX} height={HEIGHT} fill="transparent" />
            <text x={p.x} y={HEIGHT - PAD + 16} textAnchor="middle" className="axis-label" fontSize="10">
              {p.date.slice(5)}
            </text>
            {hovered === i && (
              <title>
                {p.date}: {p.count} vi phạm
              </title>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}
