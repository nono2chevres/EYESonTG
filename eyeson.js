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

function normalizeBoxToCrop(box, crop) {
  if (!box || !crop || !crop.width || !crop.height) return null;

  const left = (box.left - crop.left) / crop.width;
  const top = (box.top - crop.top) / crop.height;
  const right = (box.left + box.width - crop.left) / crop.width;
  const bottom = (box.top + box.height - crop.top) / crop.height;

  const clampedLeft = Math.max(0, Math.min(1, left));
  const clampedTop = Math.max(0, Math.min(1, top));
  const clampedRight = Math.max(0, Math.min(1, right));
  const clampedBottom = Math.max(0, Math.min(1, bottom));

  const width = Math.max(0, clampedRight - clampedLeft);
  const height = Math.max(0, clampedBottom - clampedTop);

  if (width <= 0 || height <= 0) return null;

  return {
    left: clampedLeft,
    top: clampedTop,
    width,
    height
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

  const result = await human.detect(img);
  const face = result?.face?.[0];
  if (!face) {
    // fallback très simple: zone centrale haute
    const w = Math.floor(Math.min(width, height) * 0.7);
    const x = Math.floor((width - w) / 2);
    const y = Math.floor(height * 0.15);
    return { crop: { left: x, top: y, width: w, height: w }, eyes: null };
  }

  if (face.mesh?.length) {
    const mesh = face.mesh; // tableau de points [x,y,z] en pixels
    const leftPts = LEFT_EYE_IDX.map(i => [mesh[i][0], mesh[i][1]]);
    const rightPts = RIGHT_EYE_IDX.map(i => [mesh[i][0], mesh[i][1]]);

    const leftBox = bboxFromPoints(leftPts, 18);
    const rightBox = bboxFromPoints(rightPts, 18);
    let eyes = mergeBoxes(leftBox, rightBox, 30);

    // forcer carré centré
    const side = Math.max(eyes.width, eyes.height);
    const cx = eyes.left + eyes.width / 2;
    const cy = eyes.top + eyes.height / 2;
    const left = Math.max(0, Math.floor(cx - side / 2));
    const top = Math.max(0, Math.floor(cy - side / 2));
    const boundedSide = Math.min(side, width - left, height - top);
    const safeSide = Math.max(1, Math.floor(boundedSide));

    const crop = { left, top, width: safeSide, height: safeSide };
    const normalizedLeft = normalizeBoxToCrop(leftBox, crop);
    const normalizedRight = normalizeBoxToCrop(rightBox, crop);

    const eyesMeta = (normalizedLeft || normalizedRight)
      ? { leftBox: normalizedLeft, rightBox: normalizedRight }
      : null;

    return { crop, eyes: eyesMeta };
  }

  // fallback via box visage si pas de mesh
  if (face.box) {
    const { x, y, width: bw, height: bh } = face.box;
    const w = Math.floor(bw * 0.9);
    const h = w;
    const left = Math.floor(x + (bw - w) / 2);
    const top = Math.floor(y + bh * 0.2);
    const crop = { left: Math.max(0, left), top: Math.max(0, top), width: w, height: h };
    return { crop, eyes: null };
  }

  // fallback global
  const w = Math.floor(Math.min(width, height) * 0.7);
  const x = Math.floor((width - w) / 2);
  const y = Math.floor(height * 0.15);
  return { crop: { left: x, top: y, width: w, height: w }, eyes: null };
}

function applyEyeMask(ctx, eyes, options = {}) {
  const { leftBox, rightBox } = eyes || {};
  if (!leftBox && !rightBox) return;

  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  const expandX = options.expandX ?? 1.6;
  const expandY = options.expandY ?? 1.9;
  const downShift = options.downShift ?? 0.12;
  const bridgeStrength = options.bridgeStrength ?? 0.28;

  const ellipses = [];
  [leftBox, rightBox].forEach((box) => {
    if (!box) return;
    const radiusX = (box.width / 2) * width * expandX;
    const radiusY = (box.height / 2) * height * expandY;
    const baseCenterX = (box.left + box.width / 2) * width;
    const baseCenterY = (box.top + box.height / 2) * height;
    const centerY = baseCenterY + radiusY * downShift;
    ellipses.push({
      centerX: baseCenterX,
      centerY,
      radiusX,
      radiusY
    });
  });

  if (!ellipses.length) return;

  ctx.save();
  ctx.globalCompositeOperation = 'destination-in';
  ctx.beginPath();
  ellipses.forEach(({ centerX, centerY, radiusX, radiusY }) => {
    ctx.ellipse(centerX, centerY, Math.max(1, radiusX), Math.max(1, radiusY), 0, 0, Math.PI * 2);
  });

  if (ellipses.length === 2 && bridgeStrength > 0) {
    const [left, right] = ellipses[0].centerX <= ellipses[1].centerX ? ellipses : [ellipses[1], ellipses[0]];
    const midX = (left.centerX + right.centerX) / 2;
    const midY = (left.centerY + right.centerY) / 2;
    const halfWidth = Math.abs(right.centerX - left.centerX) / 2 + Math.max(left.radiusX, right.radiusX) * 0.05;
    const radiusY = Math.max(left.radiusY, right.radiusY) * bridgeStrength;
    ctx.ellipse(midX, midY, Math.max(1, halfWidth), Math.max(1, radiusY), 0, 0, Math.PI * 2);
  }

  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.restore();
}

// ───────────────────────────────────────────────────────────────────────────────
// Rendu EyesOn
// ───────────────────────────────────────────────────────────────────────────────
async function generateEyesOn(imagePath, name, settings = {}) {
  // 1) Détection & crop
  const { crop, eyes } = await detectEyesCrop(imagePath);

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

  const eyesBuf = await pipeline.resize(1200, 1200).toBuffer();

  // 4) Composition finale
  const canvas = createCanvas(1200, 1400);
  const ctx = canvas.getContext('2d');

  // fond pantone
  ctx.fillStyle = pantone.hex;
  ctx.fillRect(0, 0, 1200, 1200);

  // image yeux
  const eyesImg = await loadImage(eyesBuf);
  const maskedEyesCanvas = createCanvas(1000, 1000);
  const maskedEyesCtx = maskedEyesCanvas.getContext('2d');
  maskedEyesCtx.drawImage(eyesImg, 0, 0, 1000, 1000);
  if (eyes) {
    applyEyeMask(maskedEyesCtx, eyes, settings.eyeMask);
  }
  ctx.drawImage(maskedEyesCanvas, 100, 100);

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
