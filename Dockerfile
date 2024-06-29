# Use Nvidia CUDA image as parent for GPU usage
# The FROM instruction initializes a new build stage and sets the base image for subsequent instructions.
# Options:
#    Different CUDA versions: You can change 11.8.0 to another version if you need a different version of CUDA.
#    Different base OS: You can switch ubuntu22.04 to other supported OS versions like ubuntu20.04.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
# The ENV instruction sets the environment variables for subsequent instructions.
# CONDA_DIR specifies the installation directory for Conda.
# PATH is modified to include Conda binaries, ensuring they are accessible from the command line.
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install dependencies
# The RUN instruction executes commands in a new layer on top of the current image and commits the results.
# apt-get update updates the package list.
# apt-get install -y installs the specified packages: wget, bzip2, ca-certificates, curl, git.
# apt-get clean and rm -rf /var/lib/apt/lists/* clean up temporary files to reduce the image size.
# Options:
#    Additional packages: Add more packages to the list if your project requires other dependencies.
#    Alternative package managers: Use a different package manager if the base image uses one (e.g., yum for CentOS).
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
# The RUN instruction downloads and installs Miniconda, then cleans up the installer and Conda caches.
# wget downloads the Miniconda installer script.
# /bin/bash ~/miniconda.sh -b -p /opt/conda installs Miniconda in the specified directory.
# rm ~/miniconda.sh removes the installer script.
# $CONDA_DIR/bin/conda clean -tipsy cleans up Conda caches to reduce the image size.
# Options:
#    Different Conda versions: Change the URL to download a specific version of Miniconda or Anaconda if needed.
#    Installation location: Change /opt/conda if you prefer a different installation directory.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -tipsy

# Create a new Conda environment
# The COPY instruction copies the environment.yml file from the host to the image.
# The RUN instruction creates a Conda environment using the environment.yml file and cleans up Conda caches.
# Options:
#    Environment file location: Change /tmp/environment.yml to another location if needed.
#    Different environment file: Modify environment.yml to include the dependencies your project requires.
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -a

# Set the default command to bash
# The CMD instruction provides the default command to run when a container is started from the image. Here, it starts a bash shell.
# Options:
#    Different default commands: Change ["bash"] to another command if you want to run a different application or script by default.
CMD ["bash"]
