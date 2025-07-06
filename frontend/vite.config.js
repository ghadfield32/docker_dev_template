import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const FASTAPI_URL = env.VITE_FASTAPI_URL || 'http://localhost:8000'   // FastAPI dev server

  return {
    plugins: [react()],
    define: {
      __FASTAPI_URL__: JSON.stringify(FASTAPI_URL),        // FastAPI URL for direct access
    },
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api/v1': {
          target: FASTAPI_URL,
          changeOrigin: true,
          secure: false,
          xfwd: true,
        },
      },
    },
    build: { outDir: 'dist' },
  }
})
