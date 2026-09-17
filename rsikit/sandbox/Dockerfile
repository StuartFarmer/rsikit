ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim
RUN pip install --no-cache-dir 'numpy>=2.2,<3' 'gymnasium[classic-control]>=1.3,<2' 'cloudpickle>=3.1,<4' 'moviepy>=2,<3'
RUN mkdir -p /opt/worker/rsikit/sandbox && printf 'from .policy import Policy\n' > /opt/worker/rsikit/__init__.py && touch /opt/worker/rsikit/sandbox/__init__.py
COPY rsikit/policy.py /opt/worker/rsikit/policy.py
COPY rsikit/episode.py /opt/worker/rsikit/episode.py
COPY rsikit/envs/ /opt/worker/rsikit/envs/
COPY rsikit/sandbox/__init__.py rsikit/sandbox/worker.py rsikit/sandbox/codec.py rsikit/sandbox/evaluate.py /opt/worker/rsikit/sandbox/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
ENTRYPOINT ["python", "-I", "-u", "-c", "import sys; sys.path.insert(0, '/opt/worker'); from rsikit.sandbox.worker import main; main()"]
