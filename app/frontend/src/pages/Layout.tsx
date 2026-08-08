import { Outlet } from 'react-router-dom'
import CommandHeader from '../components/CommandHeader'
import NavSidebar from '../components/NavSidebar'

export default function Layout() {
  return (
    <div className="cc-root">
      <CommandHeader />
      <div className="cc-shell">
        <NavSidebar />
        <div className="cc-page">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
