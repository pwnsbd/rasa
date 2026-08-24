import Settings from './screens/Settings';

// Placeholder root — routes between the four zones (spec §4.2) land here as
// they're built. Only Settings/device-status exists so far.
export default function App() {
  return (
    <div className="min-h-screen bg-dusk">
      <Settings />
    </div>
  );
}
