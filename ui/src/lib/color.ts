export function rgbCss([r, g, b]: [number, number, number], alpha = 1): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
