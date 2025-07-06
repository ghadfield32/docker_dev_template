# ML Full Stack Application

A complete machine learning application with FastAPI backend, React frontend, and MLflow integration for model management. Features iris classification and breast cancer prediction with real-time retraining capabilities.

## 🚀 Quick Start

frontend: zesty-starship-5035af.netlify.app
backend: https://docker-dev-template.onrender.com


### Prerequisites
- Docker and Docker Compose
- Invoke (Python task runner)
- Node.js 18+ (for local development)

### 1. Initial Setup

```bash
# Clone and setup the project
git clone <your-repo>
cd <your-repo>

# Start the development environment
invoke up --name ml-fullstack --detach

# Wait for containers to be ready (check with: docker ps)
```

### 2. Start the Application

```bash
# Start both backend and frontend with one command
npm run dev:all
```

This will:
- Start the FastAPI backend on http://localhost:8000
- Start the React frontend on http://localhost:5173
- Configure Vite proxy to forward `/api/v1/*` requests to the backend

### 3. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/v1/docs
- **MLflow UI**: http://localhost:5000

## 📊 Features

### Model Predictions
- **Iris Classification**: Predict iris species using Random Forest or Logistic Regression
- **Breast Cancer Detection**: Predict malignant/benign using Bayesian Logistic Regression
- Real-time predictions with confidence scores and uncertainty estimates

### Model Training & Retraining
- **Iris Models**: Optuna-based hyperparameter optimization for RF and LR
- **Cancer Models**: PyMC-based Bayesian inference with customizable MCMC parameters
- Background training with progress monitoring
- Automatic model promotion to production

### Parameter Controls
- **Iris Training**: Adjust Optuna trials (10-200)
- **Cancer Training**: Customize MCMC parameters:
  - Draws: 100-2000 samples
  - Tune: 100-1000 warmup steps
  - Target Accept: 0.7-0.99 acceptance rate

## 🔧 Development

### Project Structure
```
├── backend/
│   ├── app/
│   │   ├── api/api_v1/endpoints/     # API endpoints
│   │   ├── services/ml/              # ML service layer
│   │   └── schemas/                  # Pydantic models
│   └── src/backend/ML/               # ML training code
├── frontend/
│   ├── src/
│   │   ├── services/api.js          # API client
│   │   └── App.jsx                  # Main React component
│   └── vite.config.js               # Vite configuration
└── docker-compose.yml               # Container orchestration
```

### API Endpoints

#### Health & System
- `GET /api/v1/models/health` - System health status
- `GET /api/v1/models/metrics` - Model performance metrics
- `GET /api/v1/models/list` - Available models

#### Iris Predictions
- `POST /api/v1/iris/predict` - Make iris predictions
- `GET /api/v1/iris/models` - Available iris models
- `GET /api/v1/iris/sample-data` - Sample iris data
- `POST /api/v1/iris/retrain` - Retrain iris models

#### Cancer Predictions
- `POST /api/v1/cancer/predict` - Make cancer predictions
- `GET /api/v1/cancer/models` - Available cancer models
- `GET /api/v1/cancer/sample-data` - Sample cancer data
- `POST /api/v1/cancer/retrain` - Retrain cancer models

### Training from Command Line

```bash
# Iris retraining (default 50 Optuna trials)
curl -X POST http://localhost:8000/api/v1/iris/retrain \
  -H "Content-Type: application/json" \
  -d '{"hyperparameters": {"n_trials": 100}}'

# Cancer retraining with custom MCMC parameters
curl -X POST http://localhost:8000/api/v1/cancer/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "hyperparameters": {
      "draws": 1200,
      "tune": 800,
      "target_accept": 0.95
    }
  }'
```

### Frontend Development

```bash
# Start frontend only
cd frontend
npm run dev

# Build for production
npm run build
```

### Backend Development

```bash
# Start backend only
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/
```

## 🔄 Daily Development Workflow

### Morning Setup
```bash
# 1. Start containers
invoke up --name ml-fullstack --detach

# 2. Start application
npm run dev:all

# 3. Open browser to http://localhost:5173
```

### Development Commands
```bash
# View logs
docker-compose logs -f datascience

# Restart backend
docker-compose restart model-api

# Check MLflow
open http://localhost:5000

# Run quick test
curl http://localhost:8000/api/v1/models/health
```

### Evening Cleanup
```bash
# Stop application
pkill -f "vite\|uvicorn"

# Stop containers (optional)
invoke down --all
```

## 🧪 Testing

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/api/v1/models/health

# Test iris prediction
curl -X POST http://localhost:8000/api/v1/iris/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rf",
    "samples": [{
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }]
  }'

# Test cancer prediction
curl -X POST http://localhost:8000/api/v1/cancer/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "bayes",
    "samples": [{
      "values": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    }],
    "posterior_samples": 100
  }'
```

### Frontend Testing
```bash
# Run frontend tests
cd frontend
npm test

# Check for linting issues
npm run lint
```

## 🐛 Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using the ports
lsof -i :8000
lsof -i :5173

# Kill processes if needed
pkill -f "uvicorn\|vite"
```

**Container issues:**
```bash
# Rebuild containers
invoke down --all
invoke up --name ml-fullstack --detach --build
```

**MLflow connection issues:**
```bash
# Check MLflow is running
curl http://localhost:5000

# Restart MLflow service
docker-compose restart mlflow
```

**Frontend proxy issues:**
```bash
# Check Vite config
cat frontend/vite.config.js

# Restart frontend
cd frontend && npm run dev
```

### Debug Mode

Enable debug logging:
```bash
# Backend
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug

# Frontend
# Open browser dev tools and check console
```

## 📈 Monitoring

### Health Checks
- Backend: http://localhost:8000/health
- Frontend: Check browser console for API status
- MLflow: http://localhost:5000

### Metrics
- Model performance: http://localhost:8000/api/v1/models/metrics
- Training history: Available in frontend dashboard
- MLflow experiments: http://localhost:5000

## 🔧 Configuration

### Environment Variables
```bash
# Backend
MLFLOW_TRACKING_URI=http://mlflow:5000
DEV_AUTOTRAIN=true

# Frontend
VITE_FASTAPI_URL=http://localhost:8000
```

### Docker Configuration
- Backend: Port 8000
- Frontend: Port 5173
- MLflow: Port 5000
- Jupyter: Port 8888

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
