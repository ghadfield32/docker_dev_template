# GPU enabled Docker Image Template

This README provides a comprehensive guide to setting up a robust Docker environment for data science projects with GPU support. It includes detailed explanations and best practices to help you understand each component.

## Recent Updates

### Port Customization with Invoke Commands

The project now supports customizable port mappings through invoke commands, allowing you to run multiple instances with different port configurations:

```bash
# Basic usage with default ports
invoke up --name myproject

# Custom Jupyter port
invoke up --name myproject --jupyter-port 8891

# Multiple custom ports
invoke up --name myproject --jupyter-port 8892 --tensorboard-port 6010 --streamlit-port 8502

# View current port mappings
invoke ports --name myproject
```

Available port options:
- `--jupyter-port`: Jupyter Lab port (default: 8890)
- `--tensorboard-port`: TensorBoard port (auto-assigned by default)
- `--streamlit-port`: Streamlit port (auto-assigned by default)
- `--explainer-port`: Explainer Dashboard port (auto-assigned by default)

This feature is particularly useful when:
- Running multiple project instances simultaneously
- Avoiding port conflicts with other services
- Setting up development environments for team collaboration

## Invoke Commands Reference

This project uses `invoke` tasks to manage Docker containers and development workflows. Here's a comprehensive list of available commands:

### Container Management

```bash
# Start a new container
invoke up [--name PROJECT_NAME] [--jupyter-port PORT] [--tensorboard-port PORT] [--streamlit-port PORT] [--explainer-port PORT]
# Example: invoke up --name myproject --jupyter-port 8891

# Stop a container
invoke stop [--name PROJECT_NAME]
# Example: invoke stop --name myproject

# View port mappings
invoke ports [--name PROJECT_NAME]
# Example: invoke ports --name myproject

# Clean up resources
invoke clean
# Removes stopped containers and dangling images
```

### Development Workflow

```bash
# Run tests
invoke test [--name PROJECT_NAME]
# Example: invoke test --name myproject

# Check GPU status
invoke gpu-check [--name PROJECT_NAME]
# Example: invoke gpu-check --name myproject

# Verify JAGS installation
invoke jags-check [--name PROJECT_NAME]
# Example: invoke jags-check --name myproject
```

### Best Practices

1. **Naming Containers**:
   - Always use meaningful names for your containers with the `--name` flag
   - This helps when running multiple instances
   - Example: `invoke up --name experiment1`

2. **Port Management**:
   - Use custom ports when running multiple containers
   - Keep track of port assignments using `invoke ports`
   - Example: `invoke up --name exp1 --jupyter-port 8891 --tensorboard-port 6006`

3. **Resource Cleanup**:
   - Stop containers when not in use: `invoke stop --name myproject`
   - Regularly run `invoke clean` to remove unused resources
   - This helps maintain system performance and free up disk space

4. **Verification**:
   - Run `invoke gpu-check` to verify GPU accessibility
   - Use `invoke jags-check` to confirm JAGS installation
   - These checks help ensure your environment is properly configured

### PyJAGS Installation Fix

The container now properly supports PyJAGS installation by including the necessary JAGS system libraries. This was fixed by:

1. Adding the `jags` package to the Dockerfile, which includes both runtime and development files
2. Using `pkg-config` to verify the JAGS installation
3. Using the `CPPFLAGS="-include cstdint"` environment variable when installing PyJAGS to fix C++ header issues

The container now successfully runs:
- PyTorch with GPU support
- JAX with GPU detection (with compatibility notes)
- PyJAGS for Bayesian MCMC modeling

You can verify the functionality by running:
```bash
docker compose run --rm datascience python /workspace/tests/test_pytorch_jax_gpu.py
```

## Table of Contents
- [GPU enabled Docker Image Template](#gpu-enabled-docker-image-template)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Project Structure](#project-structure)
  - [Dockerfile](#dockerfile)
  - [Environment Configuration](#environment-configuration)
    - [environment.yml](#environmentyml)
    - [requirements.txt](#requirementstxt)
  - [Docker Compose](#docker-compose)
    - [Explanation:](#explanation)
  - [VS Code Development Container](#vs-code-development-container)
  - [Usage Guide](#usage-guide)
  - [Troubleshooting](#troubleshooting)
  - [Steps to each repo:](#steps-to-each-repo)
- [Usage:](#usage)
- [A Dockerized Python Development Environment Template](#a-dockerized-python-development-environment-template)

## Prerequisites

Before you begin, ensure you have the following installed:
- Docker Engine (version 19.03 or later)
- Docker Compose
- NVIDIA GPU drivers (if using GPU support)
- NVIDIA Container Toolkit
- Visual Studio Code with Remote - Containers extension

Installation commands:

```bash
# Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

project_root/
│
├── .devcontainer/
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── devcontainer.env
│   ├── environment.yml
│   ├── requirements.txt
│   ├── .dockerignore
│   ├── install_dependencies.sh
│   ├── install_quarto.sh
│   └── install_requirements.sh
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── ...
│
├── src/
│   └── features
│   └── models
│   └── visualization
│
├── app/
│   └── ...
│
├── tests/
│   └── test1.py
│
├── README.md
│
└── docker-compose.yml


```

## Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent tzdata from asking for user input
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -tipsy

# Create a new Conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -a

# Activate the Conda environment in all future commands
SHELL ["conda", "run", "-n", "data_science", "/bin/bash", "-c"]

# Install any additional pip packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["bash"]
```

This Dockerfile sets up a CUDA-enabled base image with Miniconda, creates a Conda environment from `environment.yml`, and installs additional pip packages from `requirements.txt`.

## Environment Configuration

### environment.yml

Create an `environment.yml` file to specify your Conda environment:

```yaml
name: data_science
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - ipykernel
  - pytest
  - black
  - flake8
  - mypy
  - pip
```

### requirements.txt

Create a `requirements.txt` file for additional pip packages:

```
tensorflow-gpu==2.11.0
torch==1.13.1
transformers==4.26.0
mlflow==2.1.1
dvc==2.45.1
```

## Docker Compose

Create a `docker-compose.yml` file for easy container management:

```yaml
version: '3.8'

services:
  datascience:
    build: .
    volumes:
      - .:/workspace
    ports:
      - "8888:8888"  # For Jupyter Notebook
      - "6006:6006"  # For TensorBoard
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - ENV_NAME=python_tutorial  # Set environment variables
      - PYTHON_VER=3.10
    command: jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

volumes:
  data:
```

### Explanation:

- **version**: Specifies the version of the Docker Compose file format.
- **services**: Defines the services to be run. In this case, we have a single service named `datascience`.
  - **build**: Specifies the build context for the Docker image. The context is the current directory (`.`).
  - **volumes**: Maps the current directory to `/workspace` inside the container. This allows you to edit files on your host machine and have them immediately reflected inside the container.
  - **ports**: Maps ports on the host machine to ports inside the container. Port 8888 is for Jupyter Notebook, and port 6006 is for TensorBoard.
  - **deploy.resources.reservations.devices**: Configures the service to use NVIDIA GPUs. It specifies that all available GPUs should be used.
  - **environment**: Sets environment variables inside the container. These can be accessed by the application running in the container.
  - **command**: Overrides the default command that the container runs. In this case, it starts Jupyter Lab.
- **volumes**: Defines named volumes. This can be used for persistent storage if needed. Currently, this section is included as a placeholder.

## VS Code Development Container

Create a `.devcontainer/devcontainer.json` file:

```json
{
    "name": "Data Science GPU",
    "dockerComposeFile": "../docker-compose.yml",
    "service": "datascience",
    "workspaceFolder": "/workspace",
    "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-azuretools.vscode-docker",
        "eamodio.gitlens",
        "github.copilot",
        "ms-python.vscode-pylance"
    ],
    "settings": {
        "python.defaultInterpreterPath": "/opt/conda/envs/data_science/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true,
        "python.formatting.provider": "black",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    "postCreateCommand": "pip install -r requirements.txt"
}
```

This configuration sets up a development container with useful VS Code extensions and settings for Python development.

## Usage Guide

1. **Build and start the container**:
   ```bash
   docker-compose up --build
   ```

2. **Access Jupyter Lab**:
   Open a web browser and go to `http://localhost:8888`. The token will be displayed in the console output.

3. **To use VS Code with the container**:
   - Open VS Code
   - Install the "Remote - Containers" extension
   - Press F1, type "Remote-Containers: Open Folder in Container", and select your project folder

4. **Run Python scripts**:
   ```bash
   docker-compose exec datascience python your_script.py
   ```

5. **Open a bash shell in the container**:
   ```bash
   docker-compose exec datascience bash
   ```

## Troubleshooting

### JAX GPU Compatibility Issues

When using JAX with GPU support, you might encounter the following warning:

```
WARNING *** Invoking ptxas with version 12.3.107, which corresponds to a CUDA version <=12.6.2. 
CUDA versions 12.x.y up to and including 12.6.2 miscompile certain edge cases around clamping.
Please upgrade to CUDA 12.6.3 or newer.
```

This occurs because:
1. The container uses CUDA 12.3.2, but JAX recommends CUDA 12.6.3 or newer
2. JAX falls back to the CUDA driver for PTX compilation when it detects this issue

**Solutions:**
- The container will still work with this warning - JAX can detect and use the GPU
- For production environments, consider using a newer CUDA version (12.6.3+)
- If you encounter specific computation errors, you may need to fall back to CPU for those operations

### Windows WSL Issues (VS Code Dev Containers)

**If you see "Command 'Dev Containers: Rebuild and Reopen in Container' resulted in an error" on Windows:**

This is usually a WSL path translation issue, not a Docker problem. Run our diagnostic:

```powershell
# PowerShell version (works even if WSL is broken)
.\.devcontainer\debug_wsl_path.ps1

# Or with Make (if available)
make wsl-check
```

**Common fixes:**
- Update WSL: `wsl --update`  
- Check distro name: `wsl -l -v` and configure VS Code if needed
- Install wslpath: `wsl -d Ubuntu -- sudo apt install -y wslu`

📖 **Detailed guide**: See [`.devcontainer/WSL_TROUBLESHOOTING.md`](.devcontainer/WSL_TROUBLESHOOTING.md)

### Other Issues

1. **GPU not detected**:
   - Ensure NVIDIA drivers are installed and up to date
   - Verify NVIDIA Container Toolkit is installed correctly
   - Check Docker daemon configuration for GPU support

2. **Package conflicts**:
   - Review `environment.yml` and `requirements.txt` for version conflicts
   - Try creating a new Conda environment with minimal dependencies and add packages incrementally

3. **Port conflicts**:
   - Change the port mappings in `docker-compose.yml` if 8888 or 6006 are already in use

4. **VS Code not connecting to container**:
   - Ensure Docker is running
   - Rebuild the container using `docker-compose up --build`
   - Check VS Code logs for any error messages

Remember to adjust paths, versions, and configurations according to your specific needs and system setup.

## Steps to each repo:

1. **Ensure docker desktop/vscode/nvidia gpu drivers are downloaded**:
   - Install Docker and Docker Compose
   - Install NVIDIA Container Toolkit for GPU support
   - Install VS Code and the Remote - Containers extension

2. **Get Dev Containers extension in VS Code marketplace**:
   - Install the "Remote - Containers" extension in VS Code

3. **Create the repository for your project**:
   - Initialize a new Git repository or clone an existing one

4. **Create or import Dockerfile**:
   - Create a new file named `Dockerfile` and populate it with the provided Dockerfile content

5. **Customize your project structure**:
   - Ensure your project follows the specified structure with directories for `data`, `notebooks`, `src`, `app`, and `tests`

6. **Set up the development environment**:
   - Follow the steps in the [Usage Guide](#usage-guide) to build and start the container, access Jupyter Lab, and use VS Code with the container

This setup ensures that you have a robust and reproducible data science environment with GPU support, ready to be cloned and utilized for various projects. Adjust paths, versions, and configurations according to your specific needs and system setup.

# Usage:

## Invoke Commands

The project provides a set of convenient invoke commands for managing your development environment:

### Basic Commands

```bash
# Start a new container
invoke up --name myproject

# Stop the container
invoke stop --name myproject

# Open a shell in the container
invoke shell --name myproject

# View container port mappings
invoke ports --name myproject

# Clean up stopped containers
invoke clean
```

### Port Configuration

You can customize port mappings when starting a new container:

1. **Default Setup**
   ```bash
   invoke up --name myproject
   # Uses default ports: Jupyter (8890), others auto-assigned
   ```

2. **Custom Jupyter Port**
   ```bash
   invoke up --name myproject --jupyter-port 8891
   # Changes only Jupyter port, others remain auto-assigned
   ```

3. **Multiple Custom Ports**
   ```bash
   invoke up --name myproject \
     --jupyter-port 8892 \
     --tensorboard-port 6010 \
     --streamlit-port 8502
   ```

4. **Running Multiple Instances**
   ```bash
   # First instance
   invoke up --name project1 --jupyter-port 8891
   
   # Second instance (different ports)
   invoke up --name project2 --jupyter-port 8892
   ```

5. **Checking Port Assignments**
   ```bash
   invoke ports --name myproject
   # Shows all port mappings for the container
   ```

### Tips for Port Configuration
- Always use different port numbers for multiple instances
- Note down custom port assignments for your projects
- Use the `ports` command to verify current mappings
- Default ports if not specified:
  - Jupyter Lab: 8890
  - TensorBoard: auto-assigned
  - Streamlit: auto-assigned
  - Explainer Dashboard: auto-assigned

# A Dockerized Python Development Environment Template

This repository provides a template for a dockerized Python development environment with VScode and the Dev Containers extension. By default, the template launches a dockerized Python environment and installs add-ins like Quarto and Jupyter. The template is highly customizable with the use of environment variables.

See also:
- [A tutorial for setting this template](https://medium.com/@rami.krispin/setting-a-dockerized-python-development-environment-template-de2400c4812b)
- [Setting up a Python Development Environment with VScode and Docker](https://github.com/RamiKrispin/vscode-python)
- [Setting up an R Development Environment with VScode and Docker](https://github.com/RamiKrispin/vscode-r)
- [Running Python/R with Docker vs. Virtual Environment](https://medium.com/@rami.krispin/running-python-r-with-docker-vs-virtual-environment-4a62ed36900f)
- [Deploy Flexdashboard on Github Pages with Github Actions and Docker](https://github.com/RamiKrispin/deploy-flex-actions)
- [Docker for Data Scientists 🐳](https://github.com/RamiKrispin/Introduction-to-Docker) (WIP)

- [How to Install PyTorch on the GPU with Docker ](https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/)
---

This comprehensive guide covers everything needed to set up and use a dockerized Python development environment with GPU support. Make sure to adapt any paths, versions, and specific configurations to fit your project requirements and system setup.