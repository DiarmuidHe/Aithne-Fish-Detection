import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// Every path the FastAPI service owns. The dev server proxies them so the app
// runs against a real API on :8000 without CORS or a second origin.
const API_ROUTES = [
  '/videos',
  '/jobs',
  '/tracks',
  '/system',
  '/batch',
  '/exports',
  '/analytics',
  '/live',
  '/health',
];

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // The build is served from /static/app by FastAPI; the dev server is its own root.
  base: command === 'build' ? '/static/app/' : '/',
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_ROUTES.map((route) => [
        route,
        { target: 'http://localhost:8000', changeOrigin: false },
      ]),
    ),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
}));
