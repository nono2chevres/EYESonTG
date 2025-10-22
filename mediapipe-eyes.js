// mediapipe-eyes.js
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

let faceLandmarker;

// indices MediaPipe des contours d’yeux (classiques)
const LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246];
const RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466];

export async function initFace() {
  if (faceLandmarker) return faceLandmarker;

  const baseUrl = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm';
  const fileset = await FilesetResolver.forVisionTasks(baseUrl);

  // modèle officiel MediaPipe (hébergé sur le CDN)
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: `${baseUrl}/face_landmarker.task`
    },
    numFaces: 1,
    runningMode: 'IMAGE',
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false
  });

  return faceLandmarker;
}

function xyFromLandmarks(landmarks, w, h) {
  // convertit points normalisés [0..1] → pixels
  return landmarks.map(p => ({ x: Math.round(p.x * w), y: Math.round(p.y * h) }));
}

function boxFromPoints(points, pad = 40) {
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  // carré centré sur les deux yeux, avec marge
  const half = Math.max((maxX - minX), (maxY - minY)) / 2 + pad;
  return { left: Math.round(cx - half), top: Math.round(cy - half), size: Math.round(2 * half) };
}

export async function detectEyesBox(imagePath) {
  const img = await loadImage(imagePath);
  const w = img.width, h = img.height;

  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  const imageData = ctx.getImageData(0, 0, w, h);

  const lm = await initFace();
  const result = lm.detect(imageData);
  if (!result?.faceLandmarks?.length) return null;

  const pts = xyFromLandmarks(result.faceLandmarks[0], w, h);
  const bothEyes = [...LEFT_EYE.map(i => pts[i]), ...RIGHT_EYE.map(i => pts[i])];
  const { left, top, size } = boxFromPoints(bothEyes, 90);

  // clamp dans l’image
  const x = Math.max(0, left);
  const y = Math.max(0, top);
  const s = Math.min(size, Math.min(w - x, h - y));

  return { left: x, top: y, width: s, height: s };
}
