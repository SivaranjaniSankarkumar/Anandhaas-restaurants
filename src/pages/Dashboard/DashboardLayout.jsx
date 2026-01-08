import Sidebar from '../../components/Sidebar/Sidebar';
import Navbar from '../../components/Navbar/Navbar';
import { Outlet } from 'react-router-dom';

export default function DashboardLayout() {
  return (
    <div className="font-sans bg-cream min-h-screen flex">
      <Sidebar />
      <div style={{ marginLeft: '250px', flex: 1 }}>
        <Navbar />
        <Outlet />
      </div>
    </div>
  );
}
