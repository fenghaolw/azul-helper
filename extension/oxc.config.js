import { resolve } from 'path';
import fs from 'fs-extra';

export default {
    entry: {
        popup: resolve(__dirname, 'src/popup.tsx'),
        content: resolve(__dirname, 'src/content.ts'),
        background: resolve(__dirname, 'src/background.ts')
    },
    output: {
        dir: 'dist',
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: (assetInfo) => {
            if (!assetInfo.name) return '[name][extname]';
            if (/\.(svg|png|jpg|jpeg|gif)$/.test(assetInfo.name)) {
                return 'icons/[name][extname]';
            }
            return '[name][extname]';
        }
    },
    resolve: {
        alias: {
            '@': resolve(__dirname, 'src'),
            '@ai': resolve(__dirname, '../webapp/src')
        }
    },
    plugins: [
        {
            name: 'copy-manifest-and-assets',
            closeBundle: async () => {
                // Copy manifest.json
                await fs.copy(
                    resolve(__dirname, 'src/manifest.json'),
                    resolve(__dirname, 'dist/manifest.json')
                );

                // Copy icons directory if it exists
                const iconsDir = resolve(__dirname, 'src/icons');
                if (await fs.pathExists(iconsDir)) {
                    await fs.copy(iconsDir, resolve(__dirname, 'dist/icons'));
                }

                // Copy popup.html to root of dist
                await fs.copy(
                    resolve(__dirname, 'src/popup.html'),
                    resolve(__dirname, 'dist/popup.html')
                );

                // Clean up src directory in dist
                await fs.remove(resolve(__dirname, 'dist/src'));
            }
        }
    ]
}; 