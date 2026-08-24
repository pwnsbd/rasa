import { useState } from 'react';
import NavRail, { type Zone } from './components/NavRail';
import MainStage from './screens/MainStage';
import DistillationRoom from './screens/DistillationRoom';
import MediaPage from './screens/MediaPage';
import Settings from './screens/Settings';

// Root shell: routes between the four zones (spec §4.2). Only the active
// zone is mounted — besides being the simplest routing model, MainStage
// relies on this to refetch its essence shelf each time it becomes active
// (see DistillationRoom.tsx's comment on the "appears only once you leave"
// behavior).
export default function App() {
  const [zone, setZone] = useState<Zone>('stage');

  return (
    <div className="h-screen w-screen flex bg-dusk overflow-hidden">
      <NavRail zone={zone} onChange={setZone} />
      <main className="flex-1 min-w-0">
        {zone === 'stage' && <MainStage />}
        {zone === 'distillation' && <DistillationRoom />}
        {zone === 'media' && <MediaPage />}
        {zone === 'settings' && <Settings />}
      </main>
    </div>
  );
}
