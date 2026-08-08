import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Live View', icon: '📹', end: true },
  { to: '/analytics', label: 'Analytics', icon: '📊' },
  { to: '/events', label: 'Event Log', icon: '📋' },
]

export default function NavSidebar() {
  return (
    <nav className="cc-nav">
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) => `cc-nav-item${isActive ? ' active' : ''}`}
        >
          <span className="cc-nav-icon">{item.icon}</span>
          <span className="cc-nav-label">{item.label}</span>
        </NavLink>
      ))}
    </nav>
  )
}
