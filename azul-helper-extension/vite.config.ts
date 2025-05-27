import { defineConfig } from 'vite';
import { resolve } from 'path';
import fs from 'fs-extra';
import preact from '@preact/preset-vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig(({ mode }) => ({
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
        },
        // More aggressive tree shaking
        manualChunks: undefined, // Disable automatic chunking for better tree shaking
      },
      treeshake: mode === 'production' ? {
        moduleSideEffects: false, // Assume no side effects for better tree shaking
        propertyReadSideEffects: false, // Assume property reads have no side effects
        tryCatchDeoptimization: false, // Don't deoptimize try-catch blocks
        unknownGlobalSideEffects: false, // Assume unknown globals have no side effects
      } : true, // Use default tree shaking in development
      external: [], // Don't externalize anything for Chrome extension
    },
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 0, // Prevent inlining of assets
    minify: mode === 'production' ? 'terser' : 'esbuild', // Use Terser for production
    sourcemap: mode === 'development', // Generate source maps in development
    terserOptions: mode === 'production' ? {
      compress: {
        drop_console: true, // Remove console.log statements
        drop_debugger: true, // Remove debugger statements
        pure_funcs: ['console.log', 'console.info', 'console.debug'], // Remove specific console methods
        passes: 3, // Multiple passes for better compression
        unsafe: true, // Enable unsafe optimizations
        unsafe_comps: true, // Unsafe comparisons
        unsafe_math: true, // Unsafe math optimizations
        unsafe_proto: true, // Unsafe prototype optimizations
      },
      mangle: {
        safari10: true, // Better compatibility
        properties: {
          regex: /^_/, // Mangle properties starting with underscore
        },
      },
      format: {
        comments: false, // Remove all comments
      },
    } : undefined,
    reportCompressedSize: true, // Show compressed sizes
    chunkSizeWarningLimit: 1000, // Warn for chunks larger than 1MB
  },
  esbuild: {
    // Additional tree shaking optimizations
    treeShaking: true,
    // Remove unused imports
    ignoreAnnotations: false,
    // Drop console in production only
    drop: mode === 'production' ? ['console', 'debugger'] : [],
    // Minify identifiers in production
    minifyIdentifiers: mode === 'production',
    minifySyntax: mode === 'production',
    minifyWhitespace: mode === 'production',
    // Keep function names in development for better debugging
    keepNames: mode === 'development',
  },
  optimizeDeps: {
    // Pre-bundle dependencies for better tree shaking
    include: ['preact', '@preact/signals'],
    // Exclude large dependencies that might not tree shake well
    exclude: [],
  },
  plugins: [
    preact({
      jsxImportSource: 'preact',
      // Enable additional optimizations
      devtoolsInProd: false,
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
      '@ai': resolve(__dirname, '../src')  // Point to main project's src directory
    }
  }
})); 