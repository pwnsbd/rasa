export function rgbCss([r, g, b]: [number, number, number], alpha = 1): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Cheap client-side average color for a freshly-added Cauldron photo
// ingredient (see DistillationRoom.tsx) — just for the live mix preview
// while dragging weights around; the real blend's color comes from the
// sidecar's own saturation-weighted _dominant_color once distilled. A tiny
// downscale (16x16) is plenty for an average and keeps this instant.
export function estimateAverageColor(dataUrl: string): Promise<[number, number, number]> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const size = 16;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve([160, 160, 160]);
        return;
      }
      ctx.drawImage(img, 0, 0, size, size);
      const { data } = ctx.getImageData(0, 0, size, size);
      let r = 0;
      let g = 0;
      let b = 0;
      const n = size * size;
      for (let i = 0; i < data.length; i += 4) {
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
      }
      resolve([Math.round(r / n), Math.round(g / n), Math.round(b / n)]);
    };
    img.onerror = () => resolve([160, 160, 160]);
    img.src = dataUrl;
  });
}
