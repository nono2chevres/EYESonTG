import axios from "axios";
import * as cheerio from "cheerio";
import fs from "fs";

const BASE_URL = "https://www.colorcodeslab.com/en/pantone-color/";

async function scrapePantone() {
  console.log("🎨 Scraping Pantone data from ColorCodesLab...");

  try {
    const { data } = await axios.get(BASE_URL);
    const $ = cheerio.load(data);

    const colors = [];

    $("a.hexColorLists").each((_, el) => {
      const name = $(el).find("span.hexColorCode").text().trim();
      const hex = $(el).find("span.hexColorFill").attr("style")?.match(/#([0-9A-Fa-f]{6})/);
      if (name && hex) {
        colors.push({
          name: `PANTONE ${name}`,
          hex: `#${hex[1].toUpperCase()}`
        });
      }
    });

    console.log(`✅ Found ${colors.length} Pantone colors.`);
    fs.writeFileSync("pantone-data.json", JSON.stringify(colors, null, 2));
    console.log("💾 Saved to pantone-data.json");

  } catch (err) {
    console.error("❌ Error scraping:", err.message);
  }
}

scrapePantone();
