import { Mic } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function FAB() {
  const navigate = useNavigate();

  return (
    <button 
      onClick={() => navigate('/voice-assistant')}
      className="fixed bottom-6 right-6 flex items-center gap-3 bg-gold text-white shadow-xl px-5 py-3 rounded-full text-lg font-medium z-50 hover:bg-yellow-600 transition-all"
    >
      <span>Ask Anandhaas AI</span>
      <Mic className="w-6 h-6" />
    </button>
  );
}
