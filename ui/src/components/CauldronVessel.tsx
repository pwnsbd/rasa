import { rgbCss } from '../lib/color';

// The Cauldron's vessel — same construction pattern as DistillationBottle
// (clip-path + gradient-filled liquid + outline stroke), a round pot
// instead of a bottle. `fill` 0..1 drives the liquid level exactly like the
// bottle does during the pour/seal animation; while mixing (before "Distill
// blend" is pressed) callers just pass fill={1} so the color updates live
// as ingredients/weights change, with `swirling` adding a slow wobble for a
// "something is happening in here" feel.
export default function CauldronVessel(props: {
  fill: number;
  color: [number, number, number];
  swirling?: boolean;
  size?: number;
}) {
  const { fill, color, swirling = false, size = 120 } = props;
  const bowlPath = 'M8 10 Q6 10 6 15 C6 30 16 42 32 42 C48 42 58 30 58 15 Q58 10 56 10 Z';
  const liquidTop = 10;
  const liquidBottom = 42;
  const liquidHeight = Math.max(0, (liquidBottom - liquidTop) * fill);
  const liquidY = liquidBottom - liquidHeight;

  return (
    <svg width={size} height={(size / 64) * 48} viewBox="0 0 64 48">
      <defs>
        <clipPath id="cauldron-clip">
          <path d={bowlPath} />
        </clipPath>
        <linearGradient id="cauldron-liquid-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={rgbCss(color, 0.95)} />
          <stop offset="100%" stopColor={rgbCss(color, 0.55)} />
        </linearGradient>
      </defs>

      <g clipPath="url(#cauldron-clip)">
        <rect x="0" y="0" width="64" height="48" fill="rgba(255,255,255,0.03)" />
        {fill > 0 && (
          <rect
            x="0"
            y={liquidY}
            width="64"
            height={liquidHeight}
            fill="url(#cauldron-liquid-grad)"
            className={swirling ? 'cauldron-swirl' : undefined}
            style={{ filter: `drop-shadow(0 0 8px ${rgbCss(color, 0.7)})` }}
          />
        )}
      </g>

      <ellipse cx="32" cy="10" rx="26" ry="3" fill="none" stroke="currentColor" strokeWidth="1.2" className="text-ink-soft" />
      <path d="M6 15 Q0 15 2 22" fill="none" stroke="currentColor" strokeWidth="1.2" className="text-ink-soft" />
      <path d="M58 15 Q64 15 62 22" fill="none" stroke="currentColor" strokeWidth="1.2" className="text-ink-soft" />
      <path d={bowlPath} fill="none" stroke="currentColor" strokeWidth="1.3" className="text-ink-soft" />
    </svg>
  );
}
