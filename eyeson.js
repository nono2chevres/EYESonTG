// eyeson.js
import { Telegraf, Markup } from 'telegraf';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import module from 'module';
import { fileURLToPath } from 'url';
import sharp from 'sharp';
import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';
import { createCanvas, loadImage, Canvas, Image, ImageData } from '@napi-rs/canvas';

// Human attend des objets globaux type DOM en environnement Node.
if (typeof globalThis.Image === 'undefined') {
  globalThis.Image = Image;
  globalThis.HTMLImageElement = Image;
}
if (typeof globalThis.Canvas === 'undefined') {
  globalThis.Canvas = Canvas;
  globalThis.HTMLCanvasElement = Canvas;
}
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = ImageData;
}
const canvasProto = typeof Canvas === 'function'
  ? (Canvas.prototype && typeof Canvas.prototype.getContext === 'function'
      ? Canvas.prototype
      : Object.getPrototypeOf(new Canvas(1, 1)))
  : null;
if (canvasProto && typeof canvasProto.getContext === 'function' && !canvasProto.__eyesonPatchedGetContext) {
  const originalGetContext = canvasProto.getContext;
  canvasProto.getContext = function patchedGetContext(type, ...args) {
    if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
      try {
        return originalGetContext.call(this, type, ...args);
      } catch (err) {
        return null;
      }
    }
    return originalGetContext.call(this, type, ...args);
  };
  canvasProto.__eyesonPatchedGetContext = true;
}

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu'; // backend pur JS

import { getPantoneColor } from './pantone-utils.js';

// ───────────────────────────────────────────────────────────────────────────────
// .env & DB
// ───────────────────────────────────────────────────────────────────────────────
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Ajoute notre shim local (./stubs) dans la résolution des modules Node.
const stubNodeModules = path.join(__dirname, 'stubs');
const nodePathEntries = process.env.NODE_PATH ? process.env.NODE_PATH.split(path.delimiter).filter(Boolean) : [];
if (!nodePathEntries.includes(stubNodeModules)) {
  nodePathEntries.push(stubNodeModules);
  process.env.NODE_PATH = nodePathEntries.join(path.delimiter);
  if (typeof module._initPaths === 'function') {
    module._initPaths();
  }
}

const stubTfjsNodeEntry = path.join(stubNodeModules, '@tensorflow', 'tfjs-node', 'index.js');
const stubTfjsNodePkg = path.join(stubNodeModules, '@tensorflow', 'tfjs-node', 'package.json');
const ModuleCtor = module.Module || module;
if (!ModuleCtor.__eyesonPatchedTfjsNode) {
  const originalResolveFilename = ModuleCtor._resolveFilename;
  const patched = function eyesonResolve(request, parent, isMain, options) {
    if (request === '@tensorflow/tfjs-node') {
      return stubTfjsNodeEntry;
    }
    if (request.startsWith('@tensorflow/tfjs-node/')) {
      if (request.endsWith('/package.json')) {
        return stubTfjsNodePkg;
      }
      return stubTfjsNodeEntry;
    }
    return originalResolveFilename.call(this, request, parent, isMain, options);
  };
  ModuleCtor._resolveFilename = patched;
  if (module._resolveFilename !== patched) {
    module._resolveFilename = patched;
  }
  ModuleCtor.__eyesonPatchedTfjsNode = true;
}

const humanModule = await import('@vladmandic/human');
const Human = humanModule.Human || humanModule.default?.Human || humanModule.default;

const db = new Low(new JSONFile('./users.json'), { users: {} });
await db.read();
if (!db.data.users) db.data.users = {};
const saveDB = () => db.write();

// ───────────────────────────────────────────────────────────────────────────────
// Dossiers requis
// ───────────────────────────────────────────────────────────────────────────────
['input', 'output'].forEach((d) => {
  if (!fs.existsSync(d)) fs.mkdirSync(d);
});

// ───────────────────────────────────────────────────────────────────────────────
/** Human config : CPU only, mesh + iris pour détecter précisément les yeux */
const human = new Human({
  backend: 'cpu',          // pas de tfjs-node, pas de wasm
  modelBasePath: 'https://vladmandic.github.io/human/models', // CDN gratuit
  cacheSensitivity: 0,
  filter: { enabled: false },
  face: {
    enabled: true,
    detector: { rotation: true, return: true },
    mesh: { enabled: true },     // 468 points
    iris: { enabled: true },     // points iris pour affiner
    description: false,
    emotion: false
  }
});
await tf.setBackend('cpu');
await tf.ready();
await human.load();
console.log('✅ Human prêt — backend:', tf.getBackend());

// ───────────────────────────────────────────────────────────────────────────────
// Utilitaires détection yeux via landmarks (mesh)
// ───────────────────────────────────────────────────────────────────────────────
/** indices mesh approximatifs des contours/coins des yeux (MediaPipe topology) */
const LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173];
const RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398];
const LEFT_IRIS_IDX = [468, 469, 470, 471];
const RIGHT_IRIS_IDX = [473, 474, 475, 476];

function clamp(value, min, max) {
  if (Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function averagePoint(points) {
  if (!points.length) return [0, 0];
  const [sumX, sumY] = points.reduce(
    (acc, p) => [acc[0] + p[0], acc[1] + p[1]],
    [0, 0]
  );
  return [sumX / points.length, sumY / points.length];
}

function computeEyeMetrics(mesh, eyelidIdx, irisIdx) {
  const eyelidPts = eyelidIdx
    .map((i) => mesh[i])
    .filter(Boolean)
    .map((pt) => [pt[0], pt[1]]);
  if (!eyelidPts.length) return null;

  const xs = eyelidPts.map((p) => p[0]);
  const ys = eyelidPts.map((p) => p[1]);
  const radiusX = (Math.max(...xs) - Math.min(...xs)) / 2;
  const radiusY = (Math.max(...ys) - Math.min(...ys)) / 2;

  let centerPts = eyelidPts;
  if (irisIdx && irisIdx.every((i) => mesh[i])) {
    centerPts = irisIdx.map((i) => [mesh[i][0], mesh[i][1]]);
  }

  return {
    center: averagePoint(centerPts),
    radiusX: Math.max(radiusX, 1),
    radiusY: Math.max(radiusY, 1),
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

function defaultEyesForCrop(crop) {
  const width = Math.max(crop.width, 1);
  const height = Math.max(crop.height, 1);
  const baseRadius = Math.max(6, Math.round(Math.min(width, height) * 0.12));
  const eyeY = Math.round(height * 0.38);
  const radiusY = Math.round(baseRadius * 0.9);
  return [
    {
      cx: Math.round(width * 0.35),
      cy: eyeY,
      radiusX: baseRadius,
      radiusY,
    },
    {
      cx: Math.round(width * 0.65),
      cy: eyeY,
      radiusX: baseRadius,
      radiusY,
    },
  ];
}

async function createEyesMask(width, height, eyes) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);

  if (!eyes?.length) {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    return canvas.encode('png');
  }

  const minRadius = Math.max(Math.min(width, height) * 0.12, 12);
  const maxRadiusX = width * 0.65;
  const maxRadiusY = height * 0.9;

  ctx.fillStyle = 'white';
  eyes.forEach((eye) => {
    const radiusX = clamp(eye.radiusX * 1.8, minRadius, maxRadiusX);
    const radiusY = clamp(eye.radiusY * 2.4, minRadius * 1.2, maxRadiusY);
    const shiftY = clamp(eye.radiusY * 0.6, -height * 0.15, height * 0.3);
    const cx = clamp(eye.cx, 0, width);
    const cy = clamp(eye.cy + shiftY, 0, height);

    ctx.beginPath();
    ctx.ellipse(cx, cy, radiusX, radiusY, 0, 0, Math.PI * 2);
    ctx.fill();
  });

  if (eyes.length >= 2) {
    const sortedEyes = [...eyes].sort((a, b) => a.cx - b.cx);
    const centerX = (sortedEyes[0].cx + sortedEyes[1].cx) / 2;
    const stripWidth = Math.abs(sortedEyes[1].cx - sortedEyes[0].cx) * 0.38;
    const startY = Math.min(
      sortedEyes[0].cy + sortedEyes[0].radiusY * 0.6,
      sortedEyes[1].cy + sortedEyes[1].radiusY * 0.6
    );
    const rectLeft = clamp(centerX - stripWidth / 2, 0, width);
    const rectTop = clamp(startY, 0, height);
    const rectWidth = Math.max(0, Math.min(stripWidth, width - rectLeft));
    const rectHeight = Math.max(0, height - rectTop);
    if (rectWidth > 0 && rectHeight > 0) {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.fillRect(rectLeft, rectTop, rectWidth, rectHeight);
      ctx.restore();
    }
  }

  const rawMask = await canvas.encode('png');
  return sharp(rawMask).blur(2).toBuffer();
}

/** calcule une bounding box (x,y,w,h) à partir d'une liste de points [x,y] */
function bboxFromPoints(pts, margin = 20) {
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  const minX = Math.min(...xs) - margin;
  const minY = Math.min(...ys) - margin;
  const maxX = Math.max(...xs) + margin;
  const maxY = Math.max(...ys) + margin;
  return {
    left: Math.floor(Math.max(0, minX)),
    top: Math.floor(Math.max(0, minY)),
    width: Math.floor(maxX - minX),
    height: Math.floor(maxY - minY)
  };
}

/** fusionne 2 boxes en une seule */
function mergeBoxes(a, b, extra = 30) {
  const left = Math.min(a.left, b.left) - extra;
  const top = Math.min(a.top, b.top) - extra;
  const right = Math.max(a.left + a.width, b.left + b.width) + extra;
  const bottom = Math.max(a.top + a.height, b.top + b.height) + extra;
  return {
    left: Math.floor(Math.max(0, left)),
    top: Math.floor(Math.max(0, top)),
    width: Math.floor(right - left),
    height: Math.floor(bottom - top)
  };
}

/**
 * Détecte les yeux: renvoie un crop carré autour des deux yeux.
 * Fallback: si pas de mesh, prend le tiers supérieur de la box visage.
 */
async function detectEyesCrop(imagePath) {
  const buf = fs.readFileSync(imagePath);
  const img = await loadImage(buf);
  const { width, height } = img;
  const minImageSide = Math.min(width, height);

  const baseFallback = () => {
    const w = Math.max(1, Math.floor(minImageSide * 0.7));
    const left = Math.round(clamp((width - w) / 2, 0, Math.max(width - w, 0)));
    const top = Math.round(clamp(height * 0.15, 0, Math.max(height - w, 0)));
    const crop = { left, top, width: w, height: w };
    return { crop, eyes: defaultEyesForCrop(crop) };
  };

  const result = await human.detect(img);
  const face = result?.face?.[0];

  const fallbackFromFaceBox = () => {
    if (!face?.box) return null;
    const { x, y, width: bw, height: bh } = face.box;
    const baseSide = Math.max(1, Math.floor(Math.min(bw, bh) * 0.9));
    const targetSide = Math.max(baseSide * 1.4, minImageSide * 0.35);
    const finalSide = Math.max(1, Math.round(Math.min(targetSide, minImageSide)));
    const centerX = clamp(x + bw / 2, finalSide / 2, width - finalSide / 2);
    const centerYBase = y + bh * 0.35;
    const centerY = clamp(centerYBase, finalSide / 2, height - finalSide / 2);
    const crop = {
      left: Math.round(centerX - finalSide / 2),
      top: Math.round(centerY - finalSide / 2),
      width: finalSide,
      height: finalSide,
    };
    return { crop, eyes: defaultEyesForCrop(crop) };
  };

  if (!face) {
    return baseFallback();
  }

  if (face.mesh?.length) {
    const mesh = face.mesh;
    const leftPts = LEFT_EYE_IDX.map((i) => mesh[i]).filter(Boolean).map((pt) => [pt[0], pt[1]]);
    const rightPts = RIGHT_EYE_IDX.map((i) => mesh[i]).filter(Boolean).map((pt) => [pt[0], pt[1]]);

    if (leftPts.length && rightPts.length) {
      const leftBox = bboxFromPoints(leftPts, 18);
      const rightBox = bboxFromPoints(rightPts, 18);
      const merged = mergeBoxes(leftBox, rightBox, 30);

      const verticalSpan = Math.max(leftBox.height, rightBox.height);
      const baseSide = Math.max(merged.width, merged.height);
      const targetSide = Math.max(
        baseSide,
        baseSide + verticalSpan * 2.5,
        minImageSide * 0.35
      );
      const finalSide = Math.max(1, Math.round(Math.min(targetSide, minImageSide)));

      const centerX = clamp(merged.left + merged.width / 2, finalSide / 2, width - finalSide / 2);
      const centerYOffset = Math.min(verticalSpan * 0.6, finalSide * 0.3);
      const centerYBase = merged.top + merged.height / 2 + centerYOffset;
      const centerY = clamp(centerYBase, finalSide / 2, height - finalSide / 2);

      const crop = {
        left: Math.round(centerX - finalSide / 2),
        top: Math.round(centerY - finalSide / 2),
        width: finalSide,
        height: finalSide,
      };

      const leftEyeInfo = computeEyeMetrics(mesh, LEFT_EYE_IDX, LEFT_IRIS_IDX);
      const rightEyeInfo = computeEyeMetrics(mesh, RIGHT_EYE_IDX, RIGHT_IRIS_IDX);
      let eyes = [leftEyeInfo, rightEyeInfo]
        .filter(Boolean)
        .map((eye) => convertEyeToCrop(eye, crop));
      if (eyes.length !== 2) {
        eyes = defaultEyesForCrop(crop);
      } else {
        eyes = eyes.sort((a, b) => a.cx - b.cx);
      }
      return { crop, eyes };
    }
  }

  const fallbackBox = fallbackFromFaceBox();
  if (fallbackBox) return fallbackBox;

  return baseFallback();
}

// ───────────────────────────────────────────────────────────────────────────────
// Rendu EyesOn
// ───────────────────────────────────────────────────────────────────────────────
async function generateEyesOn(imagePath, name, settings = {}) {
  // 1) Détection & crop
  const detection = await detectEyesCrop(imagePath);
  const { crop, eyes } = detection;

  // 2) Couleur Pantone
  let pantone;
  if (settings.fixedPantone) {
    pantone = settings.fixedPantone;
  } else {
    const stats = await sharp(imagePath).stats();
    const { r, g, b } = stats.dominant;
    const { best, topMatches } = getPantoneColor({ r, g, b }, 3);
    pantone = best;
    settings.lastPantoneCandidates = topMatches;
  }

  // 3) Traitement
  let pipeline = sharp(imagePath).extract(crop);
  if (settings.blur) pipeline = pipeline.blur(settings.blur);
  if (settings.brightness) pipeline = pipeline.modulate({ brightness: settings.brightness });

  let eyesBuf = await pipeline.toBuffer();

  if (eyes?.length) {
    const maskBuffer = await createEyesMask(crop.width, crop.height, eyes);
    eyesBuf = await sharp(eyesBuf)
      .ensureAlpha()
      .composite([{ input: maskBuffer, blend: 'dest-in' }])
      .toBuffer();
  }

  eyesBuf = await sharp(eyesBuf).resize(1200, 1200).toBuffer();

  // 4) Composition finale
  const canvas = createCanvas(1200, 1400);
  const ctx = canvas.getContext('2d');

  // fond pantone
  ctx.fillStyle = pantone.hex;
  ctx.fillRect(0, 0, 1200, 1200);

  // image yeux
  const eyesImg = await loadImage(eyesBuf);
  ctx.drawImage(eyesImg, 100, 100, 1000, 1000);

  // textes
  ctx.fillStyle = '#000';
  ctx.font = 'bold 70px sans-serif';
  ctx.fillText('EYESON®', 70, 1240);

  ctx.font = '40px sans-serif';
  ctx.fillStyle = '#555';
  ctx.fillText(pantone.name, 70, 1300);

  ctx.fillStyle = '#000';
  ctx.font = '50px sans-serif';
  ctx.fillText(name, 70, 1360);

  const outPath = path.join('output', `EYESON_${Date.now()}.png`);
  fs.writeFileSync(outPath, await canvas.encode('png'));
  return { outPath, pantone };
}

// ───────────────────────────────────────────────────────────────────────────────
// Bot TG (UI + flux)
// ───────────────────────────────────────────────────────────────────────────────
const { BOT_TOKEN = '' } = process.env;
if (!BOT_TOKEN) {
  throw new Error('Missing BOT_TOKEN environment variable.');
}

const bot = new Telegraf(BOT_TOKEN);
const userSettings = {}; // { [userId]: { imgPath, brightness, blur, name, fixedPantone, lastPantoneCandidates } }

const LANGUAGES = {
  en: {
    welcome: "👁️ *Welcome to EYESON®!*",
    description: "Send a photo, and I'll generate your custom *EyesOn®* artwork based on your eyes and Pantone colors.",
    chooseLang: "🌍 Choose your language:",
    create: "🎨 Create my EyesOn",
    sendPhoto: "📸 Please send me a photo now!",
    askName: "👁️ Send the character name you want to display:",
    first: (p) => `🧠 First proposal → ${p.name} ${p.hex}`,
    newVer: "✨ New version with your settings",
    noImg: "No image loaded.",
    pantonePicked: (n) => `✅ ${n} selected`,
    fixed: (n, h) => `🎨 Fixed shade → ${n} ${h}`,
    adjust: "⚙️ Adjust your settings:"
  },
  fr: {
    welcome: "👁️ *Bienvenue sur EYESON® !*",
    description: "Envoie une photo et je générerai ton visuel *EyesOn®* personnalisé à partir de tes yeux et de ta couleur Pantone dominante.",
    chooseLang: "🌍 Choisis ta langue :",
    create: "🎨 Créer mon visuel EyesOn",
    sendPhoto: "📸 Envoie-moi une photo maintenant !",
    askName: "👁️ Envoie le nom du personnage que tu veux afficher :",
    first: (p) => `🧠 Première proposition → ${p.name} ${p.hex}`,
    newVer: "✨ Nouvelle version avec tes réglages",
    noImg: "Aucune image chargée.",
    pantonePicked: (n) => `✅ ${n} sélectionné`,
    fixed: (n, h) => `🎨 Teinte fixée → ${n} ${h}`,
    adjust: "⚙️ Ajuste tes réglages :"
  },
  es: {
    welcome: "👁️ *¡Bienvenido a EYESON®!*",
    description: "Envíame una foto y generaré tu arte *EyesOn®* personalizado basado en tus ojos y tu color Pantone dominante.",
    chooseLang: "🌍 Elige tu idioma:",
    create: "🎨 Crear mi EyesOn",
    sendPhoto: "📸 ¡Envíame una foto ahora!",
    askName: "👁️ Envía el nombre del personaje que quieres mostrar:",
    first: (p) => `🧠 Primera propuesta → ${p.name} ${p.hex}`,
    newVer: "✨ Nueva versión con tus ajustes",
    noImg: "No hay imagen cargada.",
    pantonePicked: (n) => `✅ ${n} seleccionado`,
    fixed: (n, h) => `🎨 Tono fijado → ${n} ${h}`,
    adjust: "⚙️ Ajusta tus parámetros:"
  },
};

const settingsKeyboard = (userId) => {
  const s = userSettings[userId] || { brightness: 1, blur: 0 };
  return Markup.inlineKeyboard([
    [
      Markup.button.callback(`☀️ Luminosité: ${s.brightness.toFixed(1)}`, 'noop'),
      Markup.button.callback('➕', 'brightness_up'),
      Markup.button.callback('➖', 'brightness_down'),
    ],
    [
      Markup.button.callback(`💧 Flou: ${s.blur.toFixed(1)}`, 'noop'),
      Markup.button.callback('➕', 'blur_up'),
      Markup.button.callback('➖', 'blur_down'),
    ],
    [Markup.button.callback('🎨 Régénérer', 'regenerate')],
  ]);
};

function pantoneButtons(candidates) {
  return Markup.inlineKeyboard(
    candidates.slice(0, 3).map((p) => [
      Markup.button.callback(
        `${p.name} ${p.hex}`,
        `choose_pantone::${encodeURIComponent(p.name)}::${p.hex}`
      ),
    ])
  );
}

bot.start(async (ctx) => {
  const userLangCode = ctx.from.language_code?.substring(0, 2) || process.env.DEFAULT_LANG || 'en';
  const lang = LANGUAGES[userLangCode] ? userLangCode : 'en';

  db.data.users[ctx.from.id] = { lang };
  await saveDB();

  const t = LANGUAGES[lang];
  await ctx.replyWithMarkdown(
    `${t.welcome}\n\n${t.description}\n\n${t.chooseLang}`,
    Markup.inlineKeyboard([
      [
        Markup.button.callback('🇫🇷 Français', 'lang_fr'),
        Markup.button.callback('🇬🇧 English', 'lang_en'),
        Markup.button.callback('🇪🇸 Español', 'lang_es'),
      ],
      [Markup.button.callback(t.create, 'create')],
    ])
  );
});

bot.action(['lang_fr', 'lang_en', 'lang_es'], async (ctx) => {
  const lang = ctx.callbackQuery.data.split('_')[1];
  db.data.users[ctx.from.id] = { ...db.data.users[ctx.from.id], lang };
  await saveDB();

  const t = LANGUAGES[lang];
  await ctx.editMessageText(
    `${t.welcome}\n\n${t.description}\n\n${t.chooseLang}`,
    {
      parse_mode: 'Markdown',
      reply_markup: Markup.inlineKeyboard([
        [
          Markup.button.callback('🇫🇷 Français', 'lang_fr'),
          Markup.button.callback('🇬🇧 English', 'lang_en'),
          Markup.button.callback('🇪🇸 Español', 'lang_es'),
        ],
        [Markup.button.callback(t.create, 'create')],
      ]).reply_markup,
    }
  );
  await ctx.answerCbQuery(`✅ Langue changée : ${lang.toUpperCase()}`);
});

bot.action('create', async (ctx) => {
  const lang = db.data.users[ctx.from.id]?.lang || 'en';
  const t = LANGUAGES[lang];
  await ctx.answerCbQuery();
  await ctx.reply(`${t.description}\n\n${t.sendPhoto}`, { parse_mode: 'Markdown' });
});

bot.on('photo', async (ctx) => {
  const fileId = ctx.message.photo.pop().file_id;
  const fileLink = await ctx.telegram.getFileLink(fileId);
  const imgPath = `input/${ctx.from.id}.jpg`;

  const res = await fetch(fileLink.href);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(imgPath, buf);

  userSettings[ctx.from.id] = { brightness: 1, blur: 0, imgPath };

  const lang = db.data.users[ctx.from.id]?.lang || 'en';
  const t = LANGUAGES[lang];
  await ctx.reply(t.askName);
});

bot.on('text', async (ctx) => {
  const settings = userSettings[ctx.from.id];
  if (!settings?.imgPath) return;

  const name = ctx.message.text.trim();
  settings.name = name;

  const lang = db.data.users[ctx.from.id]?.lang || 'en';
  const t = LANGUAGES[lang];

  try {
    const { outPath, pantone } = await generateEyesOn(settings.imgPath, name, settings);
    await ctx.replyWithPhoto({ source: outPath }, {
      caption: t.first(pantone),
      parse_mode: 'Markdown',
    });

    if (settings.lastPantoneCandidates?.length) {
      await ctx.reply('🎨 Choisis ta teinte Pantone :', pantoneButtons(settings.lastPantoneCandidates));
    }

    await ctx.reply(t.adjust, settingsKeyboard(ctx.from.id));
  } catch (err) {
    console.error(err);
    await ctx.reply('❌ Erreur lors de la génération. Essaie une autre photo.');
  }
});

bot.action(['brightness_up', 'brightness_down', 'blur_up', 'blur_down', 'regenerate'], async (ctx) => {
  const s = userSettings[ctx.from.id];
  const lang = db.data.users[ctx.from.id]?.lang || 'en';
  const t = LANGUAGES[lang];

  if (!s) return ctx.answerCbQuery(t.noImg);

  switch (ctx.callbackQuery.data) {
    case 'brightness_up': s.brightness = Math.min((s.brightness || 1) + 0.1, 2); break;
    case 'brightness_down': s.brightness = Math.max((s.brightness || 1) - 0.1, 0.5); break;
    case 'blur_up': s.blur = Math.min((s.blur || 0) + 0.5, 5); break;
    case 'blur_down': s.blur = Math.max((s.blur || 0) - 0.5, 0); break;
    case 'regenerate': {
      const { outPath } = await generateEyesOn(s.imgPath, s.name, s);
      await ctx.replyWithPhoto({ source: outPath }, {
        caption: t.newVer,
        parse_mode: 'Markdown',
        ...settingsKeyboard(ctx.from.id),
      });
      return;
    }
  }

  await ctx.editMessageReplyMarkup(settingsKeyboard(ctx.from.id).reply_markup);
  ctx.answerCbQuery('🔧 OK');
});

bot.action('noop', (ctx) => ctx.answerCbQuery(''));

bot.action(/choose_pantone::(.+?)::(#?[0-9A-Fa-f]{6})/, async (ctx) => {
  const s = userSettings[ctx.from.id];
  const lang = db.data.users[ctx.from.id]?.lang || 'en';
  const t = LANGUAGES[lang];

  if (!s) return ctx.answerCbQuery(t.noImg);

  const name = decodeURIComponent(ctx.match[1]);
  const hex = ctx.match[2].toUpperCase();
  s.fixedPantone = { name, hex };

  await ctx.answerCbQuery(t.pantonePicked(name));
  const { outPath } = await generateEyesOn(s.imgPath, s.name, s);

  await ctx.replyWithPhoto({ source: outPath }, {
    caption: t.fixed(name, hex),
    parse_mode: 'Markdown',
    ...settingsKeyboard(ctx.from.id),
  });
});

// ───────────────────────────────────────────────────────────────────────────────
if (process.env.DISABLE_BOT === '1') {
  console.log('🚫 Lancement du bot Telegram désactivé (DISABLE_BOT=1).');
} else {
  bot.launch();
  console.log('🤖 Bot EyesOn + Human (CPU) lancé !');
}
