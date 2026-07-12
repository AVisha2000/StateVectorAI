import { Link } from 'react-router-dom'
import { PageHeader } from '../lib/ui.jsx'

export default function NotFound() {
  return (
    <>
      <PageHeader title="Not found" sub="That route doesn’t exist in the redesigned cockpit." />
      <div className="scaffold">
        <p style={{ margin: 0 }}>
          Head back to <Link to="/">Overview</Link>, or pick a surface from the sidebar.
        </p>
      </div>
    </>
  )
}
