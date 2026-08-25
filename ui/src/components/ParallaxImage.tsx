import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

// Depth-based parallax mapping (see sidecar/depth.py + generation.py's
// compute_depth): on hover, samples the color image at
// `uv + (mouse - center) * strength * depth(uv)` so near content shifts
// more than far content as the cursor moves — continuous per-pixel
// displacement, not a 2-3 layer cutout trick. Drop-in replacement for a
// plain <img> — falls back to exactly that when there's no depth map
// (older creations, or compute_depth=False), so nothing regresses.
//
// The WebGL scene is only mounted while actually hovered (see `active`
// below), not for every card in a grid simultaneously — a Media Page can
// hold many items, and holding a live WebGLRenderer + render loop per card
// risks the browser's concurrent-context ceiling. This also just matches
// the actual requirement ("on hover"), not a compromise.
interface ParallaxImageProps {
  src: string;
  depthSrc?: string | null;
  alt?: string;
  className?: string; // sizing/visual classes for the box (width/aspect/rounding) — NOT object-fit, see `fit`
  style?: React.CSSProperties; // for callers (e.g. Lightbox) that need an explicit computed pixel size rather than a CSS class
  onClick?: React.MouseEventHandler;
  fit?: 'cover' | 'contain';
  strength?: number;
}

const DEFAULT_STRENGTH = 0.05; // small — spec calls for "subtle" 2.5D, not full 3D
const SETTLE_EPSILON = 0.001;
const LERP_FACTOR = 0.08;

interface SceneState {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  material: THREE.ShaderMaterial;
  colorTex: THREE.Texture;
  depthTex: THREE.Texture;
  frame: number;
  targetMouse: THREE.Vector2;
  currentMouse: THREE.Vector2;
  hovering: boolean;
}

export default function ParallaxImage({
  src,
  depthSrc,
  alt = '',
  className,
  style,
  onClick,
  fit = 'cover',
  strength = DEFAULT_STRENGTH,
}: ParallaxImageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<SceneState | null>(null);
  const [active, setActive] = useState(false);

  function teardown() {
    const s = sceneRef.current;
    if (!s) return;
    cancelAnimationFrame(s.frame);
    s.material.dispose();
    s.colorTex.dispose();
    s.depthTex.dispose();
    s.renderer.dispose();
    sceneRef.current = null;
  }

  useEffect(() => teardown, []);

  function setupScene() {
    if (!depthSrc || !containerRef.current || !canvasRef.current || !imgRef.current) return;
    const container = containerRef.current;
    const canvas = canvasRef.current;
    const rect = container.getBoundingClientRect();
    const containerAspect = rect.width / rect.height || 1;
    const imageAspect = (imgRef.current.naturalWidth || 1) / (imgRef.current.naturalHeight || 1);

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setClearAlpha(0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(rect.width, rect.height, false);

    const camera = new THREE.OrthographicCamera(-containerAspect / 2, containerAspect / 2, 0.5, -0.5, 0.1, 10);
    camera.position.z = 1;

    const scene = new THREE.Scene();
    const loader = new THREE.TextureLoader();
    const colorTex = loader.load(src);
    const depthTex = loader.load(depthSrc);
    for (const t of [colorTex, depthTex]) {
      t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
      t.minFilter = THREE.LinearFilter;
      t.magFilter = THREE.LinearFilter;
    }

    // Same object-fit semantics as CSS, done manually since we're not
    // using CSS to size the image inside the canvas: "cover" crops the
    // texture via repeat/offset so the plane (which fills the camera
    // exactly) shows a cropped-to-fill image; "contain" leaves the texture
    // uncropped and instead shrinks the plane to the image's own aspect,
    // letterboxed by the transparent canvas around it.
    let planeW = containerAspect;
    let planeH = 1;
    if (fit === 'cover') {
      if (imageAspect > containerAspect) {
        const rx = containerAspect / imageAspect;
        colorTex.repeat.set(rx, 1);
        colorTex.offset.set((1 - rx) / 2, 0);
        depthTex.repeat.set(rx, 1);
        depthTex.offset.set((1 - rx) / 2, 0);
      } else {
        const ry = imageAspect / containerAspect;
        colorTex.repeat.set(1, ry);
        colorTex.offset.set(0, (1 - ry) / 2);
        depthTex.repeat.set(1, ry);
        depthTex.offset.set(0, (1 - ry) / 2);
      }
    } else if (imageAspect > containerAspect) {
      planeW = containerAspect;
      planeH = containerAspect / imageAspect;
    } else {
      planeH = 1;
      planeW = imageAspect;
    }

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: colorTex },
        uDepth: { value: depthTex },
        uMouse: { value: new THREE.Vector2(0.5, 0.5) },
        uStrength: { value: strength },
      },
      transparent: true,
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      // Depth Anything's convention (see sidecar/depth.py): brighter = closer.
      // Near content gets a bigger UV offset -> shifts more with the cursor.
      fragmentShader: `
        uniform sampler2D uColor;
        uniform sampler2D uDepth;
        uniform vec2 uMouse;
        uniform float uStrength;
        varying vec2 vUv;
        void main() {
          float d = texture2D(uDepth, vUv).r;
          vec2 offset = (uMouse - 0.5) * uStrength * d;
          vec2 sampleUv = clamp(vUv + offset, vec2(0.02), vec2(0.98));
          gl_FragColor = texture2D(uColor, sampleUv);
        }
      `,
    });

    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(planeW, planeH), material);
    scene.add(mesh);

    sceneRef.current = {
      renderer,
      scene,
      camera,
      material,
      colorTex,
      depthTex,
      frame: 0,
      targetMouse: new THREE.Vector2(0.5, 0.5),
      currentMouse: new THREE.Vector2(0.5, 0.5),
      hovering: true,
    };

    renderer.render(scene, camera); // first frame at mouse=center == zero displacement == identical to the <img> underneath, no visible pop on swap
    tick();
  }

  function tick() {
    const s = sceneRef.current;
    if (!s) return;
    s.currentMouse.lerp(s.targetMouse, LERP_FACTOR);
    s.material.uniforms.uMouse.value.copy(s.currentMouse);
    s.renderer.render(s.scene, s.camera);

    const settled = s.currentMouse.distanceTo(s.targetMouse) < SETTLE_EPSILON;
    if (!s.hovering && settled) {
      teardown();
      setActive(false);
      return;
    }
    s.frame = requestAnimationFrame(tick);
  }

  // Runs after `active` flips true and the canvas has actually mounted
  // (React commits the DOM before effects run), so canvasRef is populated.
  useEffect(() => {
    if (active && !sceneRef.current) setupScene();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  function handleEnter() {
    if (!depthSrc) return;
    setActive(true);
  }

  function handleMove(e: React.PointerEvent) {
    const s = sceneRef.current;
    if (!s || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = 1 - (e.clientY - rect.top) / rect.height; // screen-down -> UV-up
    s.targetMouse.set(Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y)));
    s.hovering = true;
  }

  function handleLeave() {
    const s = sceneRef.current;
    if (s) {
      s.targetMouse.set(0.5, 0.5);
      s.hovering = false; // tick() tears itself down once settled back to center
    } else {
      setActive(false);
    }
  }

  if (!depthSrc) {
    return <img src={src} alt={alt} className={className} style={{ ...style, objectFit: fit }} onClick={onClick} />;
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ ...style, position: 'relative', overflow: 'hidden' }}
      onClick={onClick}
      onPointerEnter={handleEnter}
      onPointerMove={handleMove}
      onPointerLeave={handleLeave}
    >
      <img ref={imgRef} src={src} alt={alt} className="w-full h-full" style={{ objectFit: fit }} />
      {active && <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />}
    </div>
  );
}
