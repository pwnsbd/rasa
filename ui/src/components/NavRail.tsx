import { AlchemyGearIcon, AlembicIcon, FlaskIcon, GridIcon } from './icons';

export type Zone = 'stage' | 'distillation' | 'media' | 'settings';

const ITEMS: { zone: Zone; label: string; Icon: (p: { className?: string }) => JSX.Element }[] = [
  { zone: 'stage', label: 'Main Stage', Icon: FlaskIcon },
  { zone: 'distillation', label: 'Distillation Room', Icon: AlembicIcon },
  { zone: 'media', label: 'Media Page', Icon: GridIcon },
  { zone: 'settings', label: 'Settings', Icon: AlchemyGearIcon },
];

export default function NavRail(props: { zone: Zone; onChange: (z: Zone) => void }) {
  return (
    <nav className="w-16 shrink-0 bg-charcoal border-r border-white/5 flex flex-col items-center py-6 gap-2">
      {ITEMS.map(({ zone, label, Icon }) => {
        const active = props.zone === zone;
        return (
          <button
            key={zone}
            title={label}
            onClick={() => props.onChange(zone)}
            className={`w-11 h-11 rounded-card flex items-center justify-center transition-colors ${
              active ? 'bg-surface text-gold' : 'text-ink-soft hover:text-ink hover:bg-surface/50'
            }`}
          >
            <Icon className="w-5 h-5" />
          </button>
        );
      })}
    </nav>
  );
}
