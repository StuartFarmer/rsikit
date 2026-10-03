ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim
ARG OCEAN=0
RUN apt-get update && apt-get install -y --no-install-recommends g++ swig && rm -rf /var/lib/apt/lists/*
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
RUN pip install --no-cache-dir \
    'numpy>=2.2,<3' 'scipy>=1.16,<2' 'control>=0.10,<0.11' \
    'cvxpy>=1.7,<2' 'osqp>=1,<2' 'clarabel>=0.11,<1' 'scs>=3.2,<4' \
    'scikit-learn>=1.7,<2' \
    'gymnasium[classic-control]>=1.3,<2' 'box2d-py==2.3.8' \
    'cloudpickle>=3.1,<4' 'moviepy>=2,<3'
# A separate CPU-only index prevents the default Linux CUDA dependency bundle.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.9,<3'
WORKDIR /app
COPY pyproject.toml .
# Resolve project dependencies before copying frequently edited source.
RUN python -c "import subprocess, tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; subprocess.check_call(['pip','install','--no-cache-dir',*p['dependencies'],*(d for e in ('openai','box2d','video','dev') for d in p['optional-dependencies'][e])])"
COPY scripts/install_ocean.py /tmp/install_ocean.py
RUN if [ "$OCEAN" = 1 ]; then \
    apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgl1 libx11-6 && \
    rm -rf /var/lib/apt/lists/* && \
    python -c "import subprocess,tomllib; subprocess.check_call(['pip','install','--no-cache-dir',*tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['ocean']])" && \
    python /tmp/install_ocean.py; \
    fi
# Scientific checks depend on installed libraries, not application source.
COPY tests/test_scientific_libraries.py /tmp/check_libraries.py
RUN python /tmp/check_libraries.py && rm /tmp/check_libraries.py
COPY . .
RUN pip install --no-cache-dir --no-deps .
ENV HOME=/tmp
CMD ["python", "-m", "examples.inner_loop", "--help"]
