// Thin-line "alkaline"-style icon set (spec §4.1). Alchemical glyphs for nav
// (flask/alembic/gear-with-triangle) rather than generic folder/gear icons.
// Plain inline SVG — no icon library, keeps the app fully offline.

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.4,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

export function FlaskIcon(props: { className?: string }) {
  return (
    <svg {...base} className={props.className}>
      <path d="M9.5 3h5M10 3v6.2L5.2 18.4A1.6 1.6 0 0 0 6.6 21h10.8a1.6 1.6 0 0 0 1.4-2.6L14 9.2V3" />
      <path d="M8.3 15h7.4" />
    </svg>
  );
}

export function AlembicIcon(props: { className?: string }) {
  return (
    <svg {...base} className={props.className}>
      <circle cx="10" cy="14" r="6" />
      <path d="M13.8 10.2 19 5m0 0h-3.4M19 5v3.4" />
      <path d="M10 11v6" />
    </svg>
  );
}

export function GridIcon(props: { className?: string }) {
  return (
    <svg {...base} className={props.className}>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="1" />
    </svg>
  );
}

export function AlchemyGearIcon(props: { className?: string }) {
  return (
    <svg {...base} className={props.className}>
      <circle cx="12" cy="12" r="7.5" />
      <path d="M12 8.2 15 14h-6z" />
    </svg>
  );
}

export function BottleBadge(props: { color: [number, number, number]; size?: number }) {
  const { color, size = 18 } = props;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <path
        d="M9.5 3h5M10 3v5.5L6 16.5A1.4 1.4 0 0 0 7.3 18.6h9.4A1.4 1.4 0 0 0 18 16.5L14 8.5V3"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d={`M7.4 17h9.2a0.9 0.9 0 0 0 0.7-1.4l-2-4H8.7l-2 4A0.9 0.9 0 0 0 7.4 17z`}
        fill={`rgb(${color[0]}, ${color[1]}, ${color[2]})`}
        opacity="0.9"
      />
    </svg>
  );
}
