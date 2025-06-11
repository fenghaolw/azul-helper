import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { resolve } from 'path';

export default defineConfig({
    plugins: [
        preact(),
        viteStaticCopy({
            targets: [
                { src: 'src/manifest.json', dest: '.' },
                { src: 'src/icons', dest: '.' },
                { src: 'src/popup.html', dest: '.' }
            ]
        })
    ],
    build: {
        outDir: 'dist',
        rollupOptions: {
            input: {
                popup: resolve(__dirname, 'src/popup.tsx'),
                content: resolve(__dirname, 'src/content.ts'),
                background: resolve(__dirname, 'src/background.ts')
            },
            output: {
                entryFileNames: '[name].js',
                chunkFileNames: '[name].js',
                assetFileNames: '[name][extname]'
            }
        },
        emptyOutDir: true,
        sourcemap: true,
        target: 'esnext',
        minify: false
    },
    resolve: {
        alias: {
            '@': resolve(__dirname, 'src'),
            '@ai': resolve(__dirname, '../webapp/src')
        }
    }
}); 