import { createCanvas } from '@napi-rs/canvas';

function clamp(value, min, max) {
  if (Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function createCropFromEyes(leftEye, rightEye, imageWidth, imageHeight) {
  if (!leftEye || !rightEye) return null;
  const minImageSide = Math.max(1, Math.min(imageWidth, imageHeight));
  const marginX = 2.4;
  const marginYTop = 2.8;
  const marginYBottom = 2.2;

  const rawLeft = Math.min(
    leftEye.center[0] - leftEye.radiusX * marginX,
    rightEye.center[0] - rightEye.radiusX * marginX,
  );
  const rawRight = Math.max(
    leftEye.center[0] + leftEye.radiusX * marginX,
    rightEye.center[0] + rightEye.radiusX * marginX,
  );
  const rawTop = Math.min(
    leftEye.center[1] - leftEye.radiusY * marginYTop,
    rightEye.center[1] - rightEye.radiusY * marginYTop,
  );
  const rawBottom = Math.max(
    leftEye.center[1] + leftEye.radiusY * marginYBottom,
    rightEye.center[1] + rightEye.radiusY * marginYBottom,
  );

  const spanX = rawRight - rawLeft;
  const spanY = rawBottom - rawTop;
  const baseSide = Math.max(spanX, spanY);
  const minSide = minImageSide * 0.32;
  const targetSide = clamp(baseSide * 1.1, minSide, minImageSide * 1.05);
  const side = Math.max(1, Math.min(Math.round(targetSide), minImageSide));

  const centerX = clamp((rawLeft + rawRight) / 2, side / 2, imageWidth - side / 2);
  const centerY = clamp((rawTop + rawBottom) / 2, side / 2, imageHeight - side / 2);

  const left = clamp(Math.round(centerX - side / 2), 0, Math.max(imageWidth - side, 0));
  const top = clamp(Math.round(centerY - side / 2), 0, Math.max(imageHeight - side, 0));
  const finalSide = Math.min(side, imageWidth - left, imageHeight - top);

  return {
    left,
    top,
    width: Math.max(1, Math.round(finalSide)),
    height: Math.max(1, Math.round(finalSide)),
  };
}

function convertEyeToCrop(eye, crop) {
  const minRadius = Math.max(4, Math.round(Math.min(crop.width, crop.height) * 0.05));
  const maxRadiusX = Math.max(minRadius, Math.round(crop.width * 0.5));
  const maxRadiusY = Math.max(minRadius, Math.round(crop.height * 0.5));
  return {
    cx: clamp(Math.round(eye.center[0] - crop.left), 0, Math.max(crop.width - 1, 0)),
    cy: clamp(Math.round(eye.center[1] - crop.top), 0, Math.max(crop.height - 1, 0)),
    radiusX: clamp(Math.round(eye.radiusX), minRadius, maxRadiusX),
    radiusY: clamp(Math.round(eye.radiusY), minRadius, maxRadiusY),
  };
}

function extractFeatureMaps(imageData, width, height, topLimit) {
  const total = width * height;
  const brightness = new Float32Array(total);
  const whiteness = new Float32Array(total);
  const saturation = new Float32Array(total);
  const validity = new Uint8Array(total);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const mapIndex = y * width + x;
      const idx = mapIndex * 4;
      const r = imageData[idx];
      const g = imageData[idx + 1];
      const b = imageData[idx + 2];
      const a = imageData[idx + 3];
      if (a < 40) continue;

      const maxChannel = Math.max(r, g, b);
      const minChannel = Math.min(r, g, b);
      if (maxChannel < 35) continue;

      const value = maxChannel / 255;
      const lightness = (r + g + b) / (3 * 255);
      const sat = maxChannel === 0 ? 0 : (maxChannel - minChannel) / maxChannel;
      brightness[mapIndex] = (value * 0.65) + (lightness * 0.35);
      whiteness[mapIndex] = 1 - sat;
      saturation[mapIndex] = sat;
      validity[mapIndex] = 1;
    }
  }

  const gradient = new Float32Array(total);
  for (let y = 1; y < Math.min(topLimit - 1, height - 1); y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const mapIndex = y * width + x;
      if (!validity[mapIndex]) continue;
      const left = brightness[mapIndex - 1];
      const right = brightness[mapIndex + 1];
      const up = brightness[mapIndex - width];
      const down = brightness[mapIndex + width];
      const diag1 = brightness[mapIndex - width - 1];
      const diag2 = brightness[mapIndex - width + 1];
      const diag3 = brightness[mapIndex + width - 1];
      const diag4 = brightness[mapIndex + width + 1];
      const gx = Math.abs(right - left) + 0.5 * Math.abs(diag2 - diag3);
      const gy = Math.abs(down - up) + 0.5 * Math.abs(diag4 - diag1);
      gradient[mapIndex] = Math.min(Math.sqrt(gx * gx + gy * gy), 2);
    }
  }

  return { brightness, whiteness, saturation, validity, gradient };
}

function buildScoreMap(featureMaps, width, topLimit) {
  const { brightness, whiteness, saturation, validity, gradient } = featureMaps;
  const total = width * topLimit;
  const scoreMap = new Float32Array(total);
  let globalMax = 0;

  for (let y = 0; y < topLimit; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const mapIndex = y * width + x;
      if (!validity[mapIndex]) continue;
      const w = whiteness[mapIndex];
      const b = brightness[mapIndex];
      const s = saturation[mapIndex];
      const g = gradient[mapIndex];
      const highlight = Math.pow(Math.max(w, b), 1.15);
      const chromaBoost = Math.max(0, s - 0.2) * (0.35 + b * 0.3);
      let score = (Math.pow(w, 1.35) * Math.pow(b, 1.05)) + g * 0.28 + chromaBoost * 0.45 + highlight * 0.15;
      if (y < topLimit * 0.15) score *= 0.85;
      if (y > topLimit * 0.85) score *= 0.5;
      score -= 0.18;
      if (score <= 0) continue;
      scoreMap[mapIndex] = score;
      if (score > globalMax) globalMax = score;
    }
  }

  return { scoreMap, globalMax };
}

function extractComponents(scoreMap, width, topLimit, featureMaps) {
  const total = width * topLimit;
  const visited = new Uint8Array(total);
  const queue = new Uint32Array(total);
  const components = [];
  const minSeedScore = 0.22;
  const minNeighborScore = 0.12;

  const { whiteness, brightness, saturation, gradient } = featureMaps;

  for (let index = 0; index < total; index += 1) {
    if (visited[index]) continue;
    if (scoreMap[index] < minSeedScore) continue;

    let qHead = 0;
    let qTail = 0;
    queue[qTail++] = index;
    visited[index] = 1;

    let area = 0;
    let weight = 0;
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumYY = 0;
    let minX = width;
    let maxX = -1;
    let minY = topLimit;
    let maxY = -1;
    let whiteSum = 0;
    let brightSum = 0;
    let satSum = 0;
    let gradSum = 0;

    while (qHead < qTail) {
      const current = queue[qHead++];
      const score = scoreMap[current];
      if (score <= 0) continue;

      const y = Math.floor(current / width);
      const x = current - y * width;

      area += 1;
      weight += score;
      sumX += x * score;
      sumY += y * score;
      sumXX += x * x * score;
      sumYY += y * y * score;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      whiteSum += whiteness[current];
      brightSum += brightness[current];
      satSum += saturation[current];
      gradSum += gradient[current];

      const neighbors = [
        current - width,
        current + width,
        current - 1,
        current + 1,
        current - width - 1,
        current - width + 1,
        current + width - 1,
        current + width + 1,
      ];

      for (let n = 0; n < neighbors.length; n += 1) {
        const neighbor = neighbors[n];
        if (neighbor < 0 || neighbor >= total) continue;
        if (visited[neighbor]) continue;
        if (scoreMap[neighbor] < minNeighborScore) continue;
        visited[neighbor] = 1;
        queue[qTail++] = neighbor;
      }
    }

    if (!area || weight <= 0) continue;

    const spanX = maxX - minX + 1;
    const spanY = maxY - minY + 1;
    if (spanX <= 2 || spanY <= 2) continue;

    const areaRatio = area / (width * topLimit);
    if (areaRatio < 0.001) continue;
    if (areaRatio > 0.1) continue;

    const aspect = spanX / spanY;
    if (aspect < 0.45 || aspect > 3.2) continue;

    const meanX = sumX / weight;
    const meanY = sumY / weight;
    const meanWhiteness = whiteSum / area;
    const meanBrightness = brightSum / area;
    const meanSaturation = satSum / area;
    const meanGradient = gradSum / area;

    if (meanBrightness < 0.18) continue;
    if (meanWhiteness < 0.26 && meanSaturation < 0.28) continue;
    if (meanY / topLimit > 0.88) continue;

    const varX = Math.max(sumXX / weight - meanX * meanX, 0);
    const varY = Math.max(sumYY / weight - meanY * meanY, 0);

    const radiusX = Math.max(Math.sqrt(varX) * 2.4, spanX * 0.33);
    const radiusY = Math.max(Math.sqrt(varY) * 2.0, spanY * 0.35);

    const confidence = weight * (0.7 + meanWhiteness * 0.5 + meanBrightness * 0.4 + meanGradient * 0.6);

    components.push({
      meanX,
      meanY,
      radiusX,
      radiusY,
      area,
      weight,
      spanX,
      spanY,
      meanWhiteness,
      meanBrightness,
      meanSaturation,
      meanGradient,
      confidence,
    });
  }

  return components;
}

function selectBestPair(components, context) {
  if (components.length < 2) return null;
  const {
    scaleInv,
    originalWidth,
    originalHeight,
  } = context;

  let best = null;

  for (let i = 0; i < components.length - 1; i += 1) {
    for (let j = i + 1; j < components.length; j += 1) {
      let left = components[i];
      let right = components[j];
      if (left.meanX > right.meanX) {
        const tmp = left;
        left = right;
        right = tmp;
      }

      const leftCenterX = left.meanX * scaleInv;
      const rightCenterX = right.meanX * scaleInv;
      const leftCenterY = left.meanY * scaleInv;
      const rightCenterY = right.meanY * scaleInv;

      const horizontalDist = rightCenterX - leftCenterX;
      const distNorm = horizontalDist / originalWidth;
      if (distNorm < 0.12 || distNorm > 0.7) continue;

      const verticalDiffNorm = Math.abs(leftCenterY - rightCenterY) / originalHeight;
      if (verticalDiffNorm > 0.18) continue;

      const areaRatio = Math.min(left.area, right.area) / Math.max(left.area, right.area);
      if (areaRatio < 0.35) continue;

      const brightnessBalance = Math.min(left.meanBrightness, right.meanBrightness) /
        Math.max(left.meanBrightness, right.meanBrightness);
      if (brightnessBalance < 0.35) continue;

      const confidence =
        left.confidence +
        right.confidence +
        areaRatio * 0.8 +
        (1 - Math.min(verticalDiffNorm, 1)) * 0.6 +
        Math.max(0, 0.4 - Math.abs(distNorm - 0.32)) * 1.1 +
        brightnessBalance * 0.5;

      const leftEye = {
        center: [leftCenterX, leftCenterY],
        radiusX: left.radiusX * scaleInv,
        radiusY: left.radiusY * scaleInv,
      };
      const rightEye = {
        center: [rightCenterX, rightCenterY],
        radiusX: right.radiusX * scaleInv,
        radiusY: right.radiusY * scaleInv,
      };

      const crop = createCropFromEyes(leftEye, rightEye, originalWidth, originalHeight);
      if (!crop) continue;

      const eyes = [convertEyeToCrop(leftEye, crop), convertEyeToCrop(rightEye, crop)].sort((a, b) => a.cx - b.cx);

      if (!best || confidence > best.confidence) {
        best = {
          crop,
          eyes,
          absoluteEyes: [leftEye, rightEye],
          source: 'anime-library',
          confidence,
        };
      }
    }
  }

  return best;
}

export async function detectAnimeEyesLibrary(img) {
  if (!img?.width || !img?.height) return null;

  const originalWidth = img.width;
  const originalHeight = img.height;
  const maxSide = 720;
  const scale = Math.min(1, maxSide / Math.max(originalWidth, originalHeight));
  const scaledWidth = Math.max(32, Math.round(originalWidth * scale));
  const scaledHeight = Math.max(32, Math.round(originalHeight * scale));
  const scaleInv = scale ? 1 / scale : 1;

  const canvas = createCanvas(scaledWidth, scaledHeight);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);
  const { data } = ctx.getImageData(0, 0, scaledWidth, scaledHeight);

  const topLimit = Math.max(8, Math.round(scaledHeight * 0.8));

  const featureMaps = extractFeatureMaps(data, scaledWidth, scaledHeight, topLimit);
  const { scoreMap, globalMax } = buildScoreMap(featureMaps, scaledWidth, topLimit);
  if (globalMax <= 0) return null;

  const components = extractComponents(scoreMap, scaledWidth, topLimit, featureMaps);
  if (!components.length) return null;

  const best = selectBestPair(components, {
    scaleInv,
    originalWidth,
    originalHeight,
  });

  return best;
}

export default detectAnimeEyesLibrary;
