import sharp from 'sharp';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const sizes = [16, 48, 128];
const inputSvg = resolve(__dirname, '../src/icons/icon.svg');

async function generateIcons() {
    for (const size of sizes) {
        await sharp(inputSvg)
            .resize(size, size)
            .png()
            .toFile(resolve(__dirname, `../src/icons/icon${size}.png`));
        console.log(`Generated icon${size}.png`);
    }
}

generateIcons().catch(console.error); 