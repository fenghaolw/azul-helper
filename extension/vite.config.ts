import { defineConfig } from 'vite';
import { resolve } from 'path';
import fs from 'fs-extra';
import preact from '@preact/preset-vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        popup: resolve(__dirname, 'src/popup.tsx'),
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
    preact({
      jsxImportSource: 'preact'
    }),
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
    },
    // Bundle analyzer (only in analyze mode)
    ...(process.env.ANALYZE ? [visualizer({
      filename: 'dist/bundle-analysis.html',
      open: true,
      gzipSize: true,
      brotliSize: true,
    })] : [])
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@ai': resolve(__dirname, '../webapp/src')  // Point to webapp's src directory
    }
  }
});
