import { useState } from 'react'
import type { DetectionPoint } from '../mock/commandCenter'

interface Props {
  data: DetectionPoint[]
}

const WIDTH = 480
const HEIGHT = 140
const PAD = 24

export default function DetectionTimeline({ data }: Props) {
  const [hovered, setHovered] = useState<number | null>(null)
  const max = Math.max(...data.map((d) => d.count), 1)
  const stepX = (WIDTH - PAD * 2) / (data.length - 1)

  const points = data.map((d, i) => {
    const x = PAD + i * stepX
    const y = HEIGHT - PAD - (d.count / max) * (HEIGHT - PAD * 2)
    return { x, y, ...d }
  })

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${HEIGHT - PAD} L ${points[0].x} ${HEIGHT - PAD} Z`

  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h3>DETECTION TIMELINE</h3>
      </div>
      <svg width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Timeline số lượng phát hiện">
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} className="axis-line" />
        <path d={areaPath} fill="var(--series-1)" opacity={0.12} stroke="none" />
        <path d={linePath} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinecap="round" />
        {points.map((p, i) =>
          i % 4 === 0 || i === points.length - 1 ? (
            <g key={p.time} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
              <circle cx={p.x} cy={p.y} r={hovered === i ? 5 : 3} fill="var(--series-1)" />
              <rect x={p.x - stepX * 2} y={0} width={stepX * 4} height={HEIGHT} fill="transparent" />
              <text x={p.x} y={HEIGHT - PAD + 14} textAnchor="middle" className="axis-label" fontSize="10">
                {p.time}
              </text>
              {hovered === i && (
                <title>
                  {p.time}: {p.count} phát hiện
                </title>
              )}
            </g>
          ) : null,
        )}
      </svg>
    </div>
  )
}
