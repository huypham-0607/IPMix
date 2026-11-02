
# 1. Choose a base environment
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 2. Set the working directory
WORKDIR /app

# 3. Install system libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 4. Install additional Python libraries
RUN pip install --no-cache-dir \
    numpy scipy Pillow \
    opencv-python-headless \
    scikit-learn tqdm

# 5. Copy IPMix files
COPY . .

# 6. Start an interactive shell
CMD ["bash"]
