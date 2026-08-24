import { rgbCss } from '../lib/color';

// A bottle outline that fills bottom-up with a colored "liquid" as extraction
// progresses (spec §4.2.2's "glowing, lava-like liquid pours into an empty
// bottle outline until sealed"). `fill` is 0..1, driven by DistillationRoom's
// GSAP tween.
export default function DistillationBottle(props: {
  fill: number;
  color: [number, number, number];
  size?: number;
}) {
  const { fill, color, size = 96 } = props;
  const bottlePath =
    'M15 4 H25 V15 L33 48 A7 7 0 0 1 26 57 H14 A7 7 0 0 1 7 48 L15 15 Z';
  const liquidHeight = 60 * fill;
  const liquidY = 60 - liquidHeight;

  return (
    <svg width={size} height={(size / 40) * 62} viewBox="0 0 40 62">
      <defs>
        <clipPath id="bottle-clip">
          <path d={bottlePath} />
        </clipPath>
        <linearGradient id="bottle-liquid-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={rgbCss(color, 0.95)} />
          <stop offset="100%" stopColor={rgbCss(color, 0.6)} />
        </linearGradient>
      </defs>

      <g clipPath="url(#bottle-clip)">
        <rect x="0" y="0" width="40" height="62" fill="rgba(255,255,255,0.03)" />
        {fill > 0 && (
          <rect
            x="0"
            y={liquidY}
            width="40"
            height={liquidHeight}
            fill="url(#bottle-liquid-grad)"
            style={{ filter: `drop-shadow(0 0 6px ${rgbCss(color, 0.8)})` }}
          />
        )}
      </g>

      <path d={bottlePath} fill="none" stroke="currentColor" strokeWidth="1.3" className="text-ink-soft" />
    </svg>
  );
}
