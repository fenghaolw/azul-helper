import { defineConfig } from 'vite';
import { resolve } from 'path';
import fs from 'fs-extra';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        popup: resolve(__dirname, 'src/popup.html'),
        content: resolve(__dirname, 'src/content.ts'),
        background: resolve(__dirname, 'src/background.ts')
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: (assetInfo) => {
          if (!assetInfo.name) return '[name][extname]';
          if (/\.(svg|png|jpg|jpeg|gif)$/.test(assetInfo.name)) {
            return 'icons/[name][extname]';
          }
          return '[name][extname]';
        }
      }
    },
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 0 // Prevent inlining of assets
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

        // Move popup.html to root of dist
        if (await fs.pathExists(resolve(__dirname, 'dist/src/popup.html'))) {
          await fs.move(
            resolve(__dirname, 'dist/src/popup.html'),
            resolve(__dirname, 'dist/popup.html'),
            { overwrite: true }
          );
        }

        // Clean up src directory in dist
        await fs.remove(resolve(__dirname, 'dist/src'));
      }
    }
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@ai': resolve(__dirname, '../src')  // Point to main project's src directory
    }
  }
}); 