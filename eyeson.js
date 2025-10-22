// eyeson.js
import { Telegraf, Markup } from 'telegraf';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';
import { createCanvas, loadImage } from '@napi-rs/canvas';

import Human from '@vladmandic/human';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu'; // backend pur JS

import { getPantoneColor } from './pantone-utils.js';

// ───────────────────────────────────────────────────────────────────────────────
// .env & DB
// ───────────────────────────────────────────────────────────────────────────────
dotenv.config();

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
    return { left: x, top: y, width: w, height: w };
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
    const safeSide = Math.min(side, width - left, height - top);

    return { left, top, width: safeSide, height: safeSide };
  }

  // fallback via box visage si pas de mesh
  if (face.box) {
    const { x, y, width: bw, height: bh } = face.box;
    const w = Math.floor(bw * 0.9);
    const h = w;
    const left = Math.floor(x + (bw - w) / 2);
    const top = Math.floor(y + bh * 0.2);
    return { left: Math.max(0, left), top: Math.max(0, top), width: w, height: h };
  }

  // fallback global
  const w = Math.floor(Math.min(width, height) * 0.7);
  const x = Math.floor((width - w) / 2);
  const y = Math.floor(height * 0.15);
  return { left: x, top: y, width: w, height: w };
}

// ───────────────────────────────────────────────────────────────────────────────
// Rendu EyesOn
// ───────────────────────────────────────────────────────────────────────────────
async function generateEyesOn(imagePath, name, settings = {}) {
  // 1) Détection & crop
  const crop = await detectEyesCrop(imagePath);

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
const bot = new Telegraf(process.env.BOT_TOKEN);
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
bot.launch();
console.log('🤖 Bot EyesOn + Human (CPU) lancé !');
