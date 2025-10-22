import fs from "fs";
import path from "path";

// --- Chargement du JSON Pantone complet ---
const dataPath = path.resolve("./pantone-data.json");
export const PANTONE_COLORS = JSON.parse(fs.readFileSync(dataPath, "utf8"));

// --- Conversion HEX → RGB ---
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

// --- Conversion RGB → LAB (standard sRGB → D65) ---
function rgbToLab({ r, g, b }) {
  let [R, G, B] = [r, g, b].map((v) => {
    v /= 255;
    return v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92;
  });

  const X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
  const Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) / 1.0;
  const Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;

  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : (7.787 * t) + 16 / 116);
  return {
    L: 116 * f(Y) - 16,
    a: 500 * (f(X) - f(Y)),
    b: 200 * (f(Y) - f(Z))
  };
}

// --- Distance ΔE2000 (ultra précise) ---
function deltaE2000(lab1, lab2) {
  const { L: L1, a: a1, b: b1 } = lab1;
  const { L: L2, a: a2, b: b2 } = lab2;

  const avgL = (L1 + L2) / 2;
  const C1 = Math.sqrt(a1 ** 2 + b1 ** 2);
  const C2 = Math.sqrt(a2 ** 2 + b2 ** 2);
  const avgC = (C1 + C2) / 2;

  const G = 0.5 * (1 - Math.sqrt((avgC ** 7) / (avgC ** 7 + 25 ** 7)));
  const a1p = a1 * (1 + G);
  const a2p = a2 * (1 + G);
  const C1p = Math.sqrt(a1p ** 2 + b1 ** 2);
  const C2p = Math.sqrt(a2p ** 2 + b2 ** 2);
  const avgCp = (C1p + C2p) / 2;

  const h1p = (Math.atan2(b1, a1p) * 180) / Math.PI % 360;
  const h2p = (Math.atan2(b2, a2p) * 180) / Math.PI % 360;
  const deltahp = Math.abs(h1p - h2p) > 180
    ? (h2p <= h1p ? h2p - h1p + 360 : h2p - h1p - 360)
    : h2p - h1p;

  const deltaLp = L2 - L1;
  const deltaCp = C2p - C1p;
  const deltaHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((deltahp / 2) * Math.PI / 180);

  const avgHp = Math.abs(h1p - h2p) > 180
    ? (h1p + h2p + 360) / 2
    : (h1p + h2p) / 2;

  const T = 1 - 0.17 * Math.cos((avgHp - 30) * Math.PI / 180)
    + 0.24 * Math.cos((2 * avgHp) * Math.PI / 180)
    + 0.32 * Math.cos((3 * avgHp + 6) * Math.PI / 180)
    - 0.20 * Math.cos((4 * avgHp - 63) * Math.PI / 180);

  const deltaRo = 30 * Math.exp(-(((avgHp - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt((avgCp ** 7) / (avgCp ** 7 + 25 ** 7));
  const Sl = 1 + ((0.015 * (avgL - 50) ** 2) / Math.sqrt(20 + (avgL - 50) ** 2));
  const Sc = 1 + 0.045 * avgCp;
  const Sh = 1 + 0.015 * avgCp * T;
  const Rt = -Math.sin(2 * deltaRo * Math.PI / 180) * Rc;

  return Math.sqrt(
    (deltaLp / Sl) ** 2 +
    (deltaCp / Sc) ** 2 +
    (deltaHp / Sh) ** 2 +
    Rt * (deltaCp / Sc) * (deltaHp / Sh)
  );
}

// --- Fonction principale : trouver les 3 Pantone les plus proches ---
export function getPantoneColor({ r, g, b }, top = 3) {
  const target = rgbToLab({ r, g, b });
  const results = [];

  for (const p of PANTONE_COLORS) {
    const lab = rgbToLab(hexToRgb(p.hex));
    const dist = deltaE2000(target, lab);
    results.push({ ...p, deltaE: +dist.toFixed(3) });
  }

  // Trie par proximité croissante
  results.sort((a, b) => a.deltaE - b.deltaE);

  const topMatches = results.slice(0, top);
  const best = topMatches[0];

  // Optionnel : affichage console
  console.log(`🎨 Closest Pantone: ${best.name} (${best.hex}) ΔE=${best.deltaE}`);
  console.log("🪄 Other close matches:");
  topMatches.slice(1).forEach((m, i) =>
    console.log(`   ${i + 2}. ${m.name} (${m.hex}) ΔE=${m.deltaE}`)
  );

  return { best, topMatches };
}
