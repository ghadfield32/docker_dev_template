import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
      const API_URL = env.VITE_API_URL || 'http://127.0.0.1:8000'   // Default to local FastAPI

  return {
    plugins: [react()],
    define: {
      __API_URL__: JSON.stringify(API_URL),
    },
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api/v1': {
          target: API_URL,
          changeOrigin: true,
          secure: false,
          xfwd: true,
        },
      },
    },
    build: { outDir: 'dist' },
  }
})
