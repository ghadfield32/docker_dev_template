import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api/v1', // Use proxy
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', {
      status: error.response?.status,
      url: error.config?.url,
      message: error.response?.data?.detail || error.message,
    });
    return Promise.reject(error);
  }
);

// API methods
export const apiService = {
  // Health and system endpoints
  async getHealth() {
    const response = await api.get('/models/health');
    return response.data;
  },

  async getModelMetrics() {
    const response = await api.get('/models/metrics');
    return response.data;
  },

  async listModels() {
    const response = await api.get('/models/list');
    return response.data;
  },

  async reloadModels() {
    const response = await api.post('/models/reload');
    return response.data;
  },

  // Iris endpoints
  async predictIris(data) {
    const response = await api.post('/iris/predict', data);
    return response.data;
  },

  async getIrisModels() {
    const response = await api.get('/iris/models');
    return response.data;
  },

  async getIrisSampleData() {
    const response = await api.get('/iris/sample-data');
    return response.data;
  },

  // Cancer endpoints
  async predictCancer(data) {
    const response = await api.post('/cancer/predict', data);
    return response.data;
  },

  async getCancerModels() {
    const response = await api.get('/cancer/models');
    return response.data;
  },

  async getCancerSampleData() {
    const response = await api.get('/cancer/sample-data');
    return response.data;
  },

  // Retraining endpoints
  async retrainIris(hyperparameters = {}) {
    const response = await api.post('/iris/retrain', {
      model_type: 'rf',
      hyperparameters
    });
    return response.data;
  },

  async retrainCancer(hyperparameters = {}) {
    const response = await api.post('/cancer/retrain', {
      model_type: 'bayes',
      hyperparameters
    });
    return response.data;
  },

  // Utility methods for polling
  async waitForMetrics(modelKey, maxAttempts = 60) {
    let attempts = 0;
    while (attempts < maxAttempts) {
      try {
        const metrics = await this.getModelMetrics();
        if (metrics[modelKey]?.accuracy !== undefined) {
          return metrics[modelKey];
        }
        await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
        attempts++;
      } catch (error) {
        console.error('Error polling metrics:', error);
        attempts++;
      }
    }
    throw new Error(`Timeout waiting for metrics for ${modelKey}`);
  }
};

export default apiService;

