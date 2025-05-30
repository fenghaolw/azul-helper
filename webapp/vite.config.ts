import { defineConfig } from 'vite'

export default defineConfig({
  root: '.',
  publicDir: '../static',
  build: {
    outDir: 'dist',
    sourcemap: true
  },
  server: {
    port: 3000,
    host: true
  }
})
