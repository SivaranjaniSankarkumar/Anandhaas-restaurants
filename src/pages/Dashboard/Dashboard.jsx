import restaurantBg from '../../assets/images/restaurant-bg.jpg';
import { ExternalLink } from 'lucide-react';

export default function Dashboard() {
  const handleOpenDashboard = () => {
    window.open('YOUR_RESTAURANT_QUICKSUITE_DASHBOARD_LINK_HERE', '_blank');
  };

  return (
    <div className="h-full w-full relative">
      <div 
        className="fixed inset-0 bg-cover bg-center bg-no-repeat opacity-40 z-0"
        style={{ backgroundImage: `url(${restaurantBg})` }}
      />
      <div className="relative z-10 h-full w-full flex items-start justify-center pt-32">
        <div className="text-center">
          <button
            onClick={handleOpenDashboard}
            className="flex items-center gap-3 bg-primary-600 hover:bg-primary-700 text-white font-medium py-4 px-8 rounded-xl transition-colors shadow-lg"
          >
            <ExternalLink className="w-5 h-5" />
            Open QuickSight Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
