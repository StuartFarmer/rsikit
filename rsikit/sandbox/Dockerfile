ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim
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
RUN mkdir -p /opt/worker/rsikit/sandbox && printf 'from .policy import Policy\n' > /opt/worker/rsikit/__init__.py && touch /opt/worker/rsikit/sandbox/__init__.py
COPY rsikit/policy.py /opt/worker/rsikit/policy.py
COPY rsikit/episode.py /opt/worker/rsikit/episode.py
COPY rsikit/envs/ /opt/worker/rsikit/envs/
COPY rsikit/sandbox/__init__.py rsikit/sandbox/worker.py rsikit/sandbox/codec.py rsikit/sandbox/evaluate.py rsikit/sandbox/service.py rsikit/sandbox/in_process.py /opt/worker/rsikit/sandbox/
RUN python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'rsikit-worker.pth').write_text('/opt/worker\n')"
COPY rsikit/sandbox/check_libraries.py /opt/worker/rsikit/sandbox/check_libraries.py
RUN python /opt/worker/rsikit/sandbox/check_libraries.py > /opt/worker/libraries.json
USER 65534:65534
ENTRYPOINT ["python", "-I", "-u", "-c", "import sys; sys.path.insert(0, '/opt/worker'); from rsikit.sandbox.worker import main; main()"]
