import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell } from 'recharts';
import { Activity, Brain, Database, TrendingUp, AlertCircle, CheckCircle, Play, Upload, Download, Settings, BarChart3, Target } from 'lucide-react';
import apiService from './services/api';

const MLModelFrontend = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedDataset, setSelectedDataset] = useState('iris');
  const [modelStatus, setModelStatus] = useState('idle');
  const [predictions, setPredictions] = useState([]);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [apiHealth, setApiHealth] = useState('unknown');
  const [inputData, setInputData] = useState({});
  const [modelMetrics, setModelMetrics] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);

  // Training parameters
  const [trainingParams, setTrainingParams] = useState({
    iris: { n_trials: 50 },
    cancer: { draws: 800, tune: 400, target_accept: 0.9 }
  });

  // API Base URL - adjust this to match your backend
  const API_BASE = '' // unused – kept for backward compat

  // --- Real API helper  ---------------------------------------------------
  const callApi = async (path, payload = null, opts = {}) => {
    /*
      Direct frontend→FastAPI communication using the `/api/v1` prefix.
      Vite dev server proxies `/api/v1` calls directly to FastAPI.
      This eliminates the Express proxy layer.
    */
    const url = path.startsWith('/api/v1') ? path : `/api/v1${path}`

    // 🔍 DEBUG: Log the exact URL being called
    console.log('🔍 callApi DEBUG:', {
      originalPath: path,
      finalUrl: url,
      hasPayload: payload !== null,
      method: payload !== null ? 'POST' : 'GET'
    })

    const cfg = payload !== null
      ? {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          ...opts,
        }
      : { method: 'GET', ...opts }

    try {
      const res = await fetch(url, cfg)
      console.log('🔍 callApi Response:', {
        status: res.status,
        ok: res.ok,
        url: res.url
      })

      if (!res.ok) {
        const txt = await res.text()
        console.error('❌ callApi Error:', {
          status: res.status,
          statusText: res.statusText,
          body: txt,
          url: url
        })
        throw new Error(`API ${res.status} — ${txt}`)
      }
      return res.json()
    } catch (error) {
      console.error('❌ callApi Exception:', {
        error: error.message,
        url: url,
        payload: payload
      })
      throw error
    }
  }

  // Dataset configurations
  const datasets = {
    iris: {
      name: 'Iris Dataset',
      description: 'Classic iris flower classification (Setosa vs Others)',
      features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
      featureLabels: ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)'],
      targetClasses: ['Setosa', 'Non-Setosa'],
      color: '#8884d8'
    },
    breast_cancer: {
      name: 'Breast Cancer Dataset',
      description: 'Breast cancer diagnosis prediction (Malignant vs Benign)',
      features: ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'],
      featureLabels: ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness'],
      targetClasses: ['Malignant', 'Benign'],
      color: '#82ca9d'
    }
  };

  // Check API health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        console.log('🔍 checkHealth DEBUG: Starting health check')
        const response = await callApi('/health');
        console.log('🔍 checkHealth DEBUG: Health check successful:', response)
        setApiHealth(response.status);
      } catch (error) {
        console.error('❌ checkHealth DEBUG: Health check failed:', error)
        setApiHealth('error');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Initialize input data when dataset changes
  useEffect(() => {
    const initData = {};
    datasets[selectedDataset].features.forEach(feature => {
      initData[feature] = '';
    });
    setInputData(initData);
  }, [selectedDataset]);

  // Train model
  const handleTrainModel = async () => {
    setModelStatus('training');
    try {
      console.log('🔍 handleTrainModel DEBUG: Starting training for dataset:', selectedDataset);

      let result;
      if (selectedDataset === 'iris') {
        result = await apiService.retrainIris(trainingParams.iris);
        console.log('🔍 handleTrainModel DEBUG: Iris training started:', result);

        // Poll for completion
        try {
          const newMetrics = await apiService.waitForMetrics('iris_random_forest');
          setModelMetrics(newMetrics);
          console.log('🔍 handleTrainModel DEBUG: Training completed with metrics:', newMetrics);
        } catch (error) {
          console.warn('Training may still be in progress:', error.message);
        }

      } else if (selectedDataset === 'breast_cancer') {
        result = await apiService.retrainCancer(trainingParams.cancer);
        console.log('🔍 handleTrainModel DEBUG: Cancer training started:', result);

        // Poll for completion
        try {
          const newMetrics = await apiService.waitForMetrics('breast_cancer_bayes');
          setModelMetrics(newMetrics);
          console.log('🔍 handleTrainModel DEBUG: Training completed with metrics:', newMetrics);
        } catch (error) {
          console.warn('Training may still be in progress:', error.message);
        }

      } else {
        alert('Training is only supported for Iris and Breast Cancer models.');
        setModelStatus('idle');
        return;
      }

      // Update training history
      setTrainingHistory((prev) => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          dataset: selectedDataset,
          parameters: trainingParams[selectedDataset],
          status: 'completed'
        },
      ]);

      setModelStatus('trained');
    } catch (error) {
      console.error('❌ handleTrainModel Error:', error);
      setModelStatus('error');
      alert(`Training failed: ${error.message}`);
    }
  };

  // Make prediction
  const handlePredict = async () => {
    try {
      console.log('🔍 handlePredict DEBUG: Starting prediction for dataset:', selectedDataset)
      const features = Object.values(inputData).map(val => parseFloat(val) || 0);
      let result
      if (selectedDataset === 'iris') {
        const payload = {
          rows: [
            {
              sepal_length: parseFloat(inputData.sepal_length) || 0,
              sepal_width: parseFloat(inputData.sepal_width) || 0,
              petal_length: parseFloat(inputData.petal_length) || 0,
              petal_width: parseFloat(inputData.petal_width) || 0,
            },
          ],
        }
        console.log('🔍 handlePredict DEBUG: Calling iris prediction with payload:', payload)
        result = await callApi('/iris/predict', payload)
        console.log('🔍 handlePredict DEBUG: Iris prediction result:', result)
        result.class_name = result.predictions[0] === 0 ? 'Setosa' : 'Non-Setosa'
        result.probability = 1
        result.confidence = 1
      } else {
        const values = features
        const payload = { rows: [{ values }], posterior_samples: 100 }
        console.log('🔍 handlePredict DEBUG: Calling cancer prediction with payload:', payload)
        result = await callApi('/cancer/predict', payload)
        console.log('🔍 handlePredict DEBUG: Cancer prediction result:', result)
        result.class_name = result.predictions[0] > 0.5 ? 'Malignant' : 'Benign'
        result.probability = result.predictions[0]
        result.confidence = 1
      }

      setPredictionResults(result);
      setPredictions(prev => [...prev, {
        id: Date.now(),
        input: { ...inputData },
        result: result,
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('❌ handlePredict Error:', error);
    }
  };

  // Generate sample data for visualization
  const generateSampleData = () => {
    return Array.from({ length: 100 }, (_, i) => ({
      x: Math.random() * 10,
      y: Math.random() * 10,
      class: Math.random() > 0.5 ? 'Class A' : 'Class B'
    }));
  };

  const sampleData = generateSampleData();

  // --- Background Bayesian retrain ---------------------------------------
  const handleRetrainBayes = async () => {
    setModelStatus('training');
    try {
      console.log('🔍 handleRetrainBayes DEBUG: Starting Bayesian retrain');

      const result = await apiService.retrainCancer(trainingParams.cancer);
      console.log('🔍 handleRetrainBayes DEBUG: Bayesian retrain started:', result);

      // Poll for completion
      try {
        const newMetrics = await apiService.waitForMetrics('breast_cancer_bayes');
        setModelMetrics(newMetrics);
        console.log('🔍 handleRetrainBayes DEBUG: Retrain completed with metrics:', newMetrics);
      } catch (error) {
        console.warn('Retraining may still be in progress:', error.message);
      }

      setModelStatus('trained');
    } catch (error) {
      console.error('❌ handleRetrainBayes Error:', error);
      setModelStatus('error');
      alert(`Retraining failed: ${error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">ML Model Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  apiHealth === 'healthy' ? 'bg-green-500' :
                  apiHealth === 'error' ? 'bg-red-500' : 'bg-yellow-500'
                }`} />
                <span className="text-sm text-gray-600">API Status: {apiHealth}</span>
              </div>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.entries(datasets).map(([key, dataset]) => (
                  <option key={key} value={key}>{dataset.name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'training', label: 'Training', icon: Brain },
              { id: 'prediction', label: 'Prediction', icon: TrendingUp },
              { id: 'analysis', label: 'Analysis', icon: BarChart3 }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center px-3 py-4 text-sm font-medium border-b-2 ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Dataset Info Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Current Dataset: {datasets[selectedDataset].name}</h2>
              <p className="text-gray-600 mb-4">{datasets[selectedDataset].description}</p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center">
                  <Database className="h-5 w-5 text-blue-500 mr-2" />
                  <span className="text-sm">Features: {datasets[selectedDataset].features.length}</span>
                </div>
                <div className="flex items-center">
                  <Target className="h-5 w-5 text-green-500 mr-2" />
                  <span className="text-sm">Classes: {datasets[selectedDataset].targetClasses.join(', ')}</span>
                </div>
                <div className="flex items-center">
                  <Settings className="h-5 w-5 text-purple-500 mr-2" />
                  <span className="text-sm">Model: Bayesian LogReg</span>
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <CheckCircle className="h-8 w-8 text-green-500" />
                  <div className="ml-4">
                    <p className="text-sm text-gray-600">Model Status</p>
                    <p className="text-lg font-semibold capitalize">{modelStatus}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <TrendingUp className="h-8 w-8 text-blue-500" />
                  <div className="ml-4">
                    <p className="text-sm text-gray-600">Predictions Made</p>
                    <p className="text-lg font-semibold">{predictions.length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <Activity className="h-8 w-8 text-purple-500" />
                  <div className="ml-4">
                    <p className="text-sm text-gray-600">Training Runs</p>
                    <p className="text-lg font-semibold">{trainingHistory.length}</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center">
                  <BarChart3 className="h-8 w-8 text-orange-500" />
                  <div className="ml-4">
                    <p className="text-sm text-gray-600">Accuracy</p>
                    <p className="text-lg font-semibold">
                      {modelMetrics ? `${(modelMetrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
              <div className="space-y-3">
                {[...trainingHistory, ...predictions].slice(-5).map((item, index) => (
                  <div key={index} className="flex items-center justify-between py-2 border-b border-gray-100">
                    <div className="flex items-center">
                      {item.dataset ? (
                        <Brain className="h-4 w-4 text-blue-500 mr-2" />
                      ) : (
                        <TrendingUp className="h-4 w-4 text-green-500 mr-2" />
                      )}
                      <span className="text-sm">
                        {item.dataset ? `Trained ${item.dataset} model` : 'Made prediction'}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Training Tab */}
        {activeTab === 'training' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Model Training</h2>

              {/* Training Parameters */}
              <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                <h3 className="text-md font-medium mb-3">Training Parameters</h3>

                {selectedDataset === 'iris' && (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Optuna Trials: {trainingParams.iris.n_trials}
                      </label>
                      <input
                        type="range"
                        min="10"
                        max="200"
                        value={trainingParams.iris.n_trials}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          iris: { ...prev.iris, n_trials: parseInt(e.target.value) }
                        }))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>10 (Fast)</span>
                        <span>200 (Thorough)</span>
                      </div>
                    </div>
                  </div>
                )}

                {selectedDataset === 'breast_cancer' && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        MCMC Draws: {trainingParams.cancer.draws}
                      </label>
                      <input
                        type="range"
                        min="100"
                        max="2000"
                        step="100"
                        value={trainingParams.cancer.draws}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          cancer: { ...prev.cancer, draws: parseInt(e.target.value) }
                        }))}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Tune Steps: {trainingParams.cancer.tune}
                      </label>
                      <input
                        type="range"
                        min="100"
                        max="1000"
                        step="100"
                        value={trainingParams.cancer.tune}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          cancer: { ...prev.cancer, tune: parseInt(e.target.value) }
                        }))}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Target Accept: {trainingParams.cancer.target_accept}
                      </label>
                      <input
                        type="range"
                        min="0.7"
                        max="0.99"
                        step="0.01"
                        value={trainingParams.cancer.target_accept}
                        onChange={(e) => setTrainingParams(prev => ({
                          ...prev,
                          cancer: { ...prev.cancer, target_accept: parseFloat(e.target.value) }
                        }))}
                        className="w-full"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="mb-6">
                <button
                  onClick={handleTrainModel}
                  disabled={modelStatus === 'training'}
                  className={`flex items-center px-4 py-2 rounded-md ${
                    modelStatus === 'training'
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700'
                  } text-white`}
                >
                  <Play className="h-4 w-4 mr-2" />
                  {modelStatus === 'training' ? 'Training...' : 'Train Model'}
                </button>

                {/* NEW – retrain Bayesian button */}
                {selectedDataset === 'breast_cancer' && (
                  <button
                    onClick={handleRetrainBayes}
                    disabled={modelStatus === 'training'}
                    className={`ml-4 flex items-center px-4 py-2 rounded-md ${
                      modelStatus === 'training'
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-orange-600 hover:bg-orange-700'
                    } text-white`}
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Retrain Cancer Model
                  </button>
                )}
              </div>

              {modelMetrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {Object.entries(modelMetrics).map(([metric, value]) => (
                    <div key={metric} className="text-center p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600 capitalize">{metric.replace('_', ' ')}</p>
                      <p className="text-lg font-semibold">{(value * 100).toFixed(1)}%</p>
                    </div>
                  ))}
                </div>
              )}

              {trainingHistory.length > 0 && (
                <div className="h-64">
                  <h3 className="text-md font-semibold mb-2">Training History</h3>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleTimeString()} />
                      <YAxis />
                      <Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Prediction Tab */}
        {activeTab === 'prediction' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Make Prediction</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {datasets[selectedDataset].features.map((feature, index) => (
                  <div key={feature}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {datasets[selectedDataset].featureLabels[index]}
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={inputData[feature] || ''}
                      onChange={(e) => setInputData(prev => ({ ...prev, [feature]: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder={`Enter ${feature}`}
                    />
                  </div>
                ))}
              </div>
              <button
                onClick={handlePredict}
                className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md"
              >
                <TrendingUp className="h-4 w-4 mr-2" />
                Predict
              </button>

              {predictionResults && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-semibold mb-2">Prediction Results</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <p className="text-sm text-gray-600">Predicted Class</p>
                      <p className="text-lg font-semibold">{predictionResults.class_name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Probability</p>
                      <p className="text-lg font-semibold">{(predictionResults.probability * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Confidence</p>
                      <p className="text-lg font-semibold">{(predictionResults.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {predictions.length > 0 && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Prediction History</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full table-auto">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Timestamp</th>
                        <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Prediction</th>
                        <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.slice(-10).map((pred) => (
                        <tr key={pred.id} className="border-b border-gray-200">
                          <td className="px-4 py-2 text-sm text-gray-600">
                            {new Date(pred.timestamp).toLocaleString()}
                          </td>
                          <td className="px-4 py-2 text-sm font-medium">
                            {pred.result.class_name}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {(pred.result.confidence * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Feature Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={datasets[selectedDataset].features.map((feature, index) => ({
                      name: feature,
                      value: Math.random() * 100,
                      fill: `hsl(${index * 60}, 70%, 50%)`
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Class Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={datasets[selectedDataset].targetClasses.map((className, index) => ({
                          name: className,
                          value: 50 + Math.random() * 50,
                          fill: ['#8884d8', '#82ca9d', '#ffc658', '#ff7300'][index % 4]
                        }))}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {datasets[selectedDataset].targetClasses.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={['#8884d8', '#82ca9d', '#ffc658', '#ff7300'][index % 4]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-4">Data Visualization</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={sampleData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" />
                    <YAxis type="number" dataKey="y" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter
                      name="Class A"
                      data={sampleData.filter(d => d.class === 'Class A')}
                      fill="#8884d8"
                    />
                    <Scatter
                      name="Class B"
                      data={sampleData.filter(d => d.class === 'Class B')}
                      fill="#82ca9d"
                    />
                    <Legend />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default MLModelFrontend;

