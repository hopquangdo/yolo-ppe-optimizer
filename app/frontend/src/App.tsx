import { HashRouter, Routes, Route } from 'react-router-dom'
import Layout from './pages/Layout'
import LiveView from './pages/LiveView'
import Analytics from './pages/Analytics'
import EventLogPage from './pages/EventLogPage'
import './App.css'
import './CommandCenter.css'

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<LiveView />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="events" element={<EventLogPage />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}

export default App
