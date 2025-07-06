#!/usr/bin/env node
/**
 * CI validation script for the full-stack setup.
 *
 * This script validates:
 * 1. Environment variables are set correctly
 * 2. Required files exist
 * 3. Basic configuration is valid
 */

const fs = require('fs');
const path = require('path');

// Load .env file if it exists
try {
  const dotenv = require('dotenv');
  dotenv.config();
} catch (error) {
  console.log('dotenv not available, skipping .env loading');
}

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkFileExists(filePath, description) {
  if (fs.existsSync(filePath)) {
    log(`✅ ${description}: ${filePath}`, 'green');
    return true;
  } else {
    log(`❌ ${description}: ${filePath} (missing)`, 'red');
    return false;
  }
}

function checkEnvVar(name, required = true) {
  const value = process.env[name];
  if (value) {
    log(`✅ ${name}: ${value}`, 'green');
    return true;
  } else if (required) {
    log(`❌ ${name}: (missing)`, 'red');
    return false;
  } else {
    log(`⚠️  ${name}: (not set)`, 'yellow');
    return true; // Not required, so this is OK
  }
}

function validateEnvironment() {
  log('🔍 Validating environment setup...', 'blue');

  let allGood = true;

  // Check environment variables
  log('\n📋 Environment Variables:', 'blue');
  allGood &= checkEnvVar('MLFLOW_TRACKING_URI');
  allGood &= checkEnvVar('DEV_AUTOTRAIN');
  allGood &= checkEnvVar('VITE_API_URL');

  // Check required files
  log('\n📁 Required Files:', 'blue');
  allGood &= checkFileExists('.env', 'Environment file');
  allGood &= checkFileExists('package.json', 'Root package.json');
  allGood &= checkFileExists('src/backend/package.json', 'Backend package.json');
  allGood &= checkFileExists('src/frontend/package.json', 'Frontend package.json');
  allGood &= checkFileExists('src/backend/ML/model_api/main.py', 'FastAPI main');
  allGood &= checkFileExists('src/backend/server.js', 'Express server');
  allGood &= checkFileExists('scripts/prune_models.py', 'Model pruning script');
  allGood &= checkFileExists('scripts/test_setup.py', 'Test script');

  // Check MLflow directory
  log('\n🗂️  MLflow Setup:', 'blue');
  const mlrunsDir = 'mlruns_local';
  if (fs.existsSync(mlrunsDir)) {
    log(`✅ MLflow directory exists: ${mlrunsDir}`, 'green');
  } else {
    log(`⚠️  MLflow directory missing: ${mlrunsDir} (will be created on first run)`, 'yellow');
  }

  // Check Docker setup
  log('\n🐳 Docker Setup:', 'blue');
  if (fs.existsSync('docker-compose.yml')) {
    log('✅ Docker Compose file exists', 'green');
  } else {
    log('⚠️  Docker Compose file missing', 'yellow');
  }

  return allGood;
}

function validatePackageScripts() {
  log('\n📦 Package Scripts:', 'blue');

  let allGood = true;

  try {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    const scripts = packageJson.scripts || {};

    const requiredScripts = [
      'dev:all',
      'dev:api',
      'start-api',
      'train',
      'build'
    ];

    for (const script of requiredScripts) {
      if (scripts[script]) {
        log(`✅ ${script}: ${scripts[script]}`, 'green');
      } else {
        log(`❌ ${script}: (missing)`, 'red');
        allGood = false;
      }
    }

  } catch (error) {
    log(`❌ Error reading package.json: ${error.message}`, 'red');
    allGood = false;
  }

  return allGood;
}

function main() {
  log('🚀 CI Validation Script', 'blue');
  log('=' * 50);

  const envValid = validateEnvironment();
  const scriptsValid = validatePackageScripts();

  log('\n' + '=' * 50);
  log('📊 Validation Summary:', 'blue');

  if (envValid && scriptsValid) {
    log('🎉 All validations passed!', 'green');
    log('Your setup is ready for development.', 'green');
    process.exit(0);
  } else {
    log('⚠️  Some validations failed.', 'red');
    log('Please fix the issues above before proceeding.', 'red');
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { validateEnvironment, validatePackageScripts };
