FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 'gymnasium>=1.3,<2'
RUN mkdir -p /opt/worker/rsikit/sandbox && printf 'from .controller import Controller\n' > /opt/worker/rsikit/__init__.py && touch /opt/worker/rsikit/sandbox/__init__.py
COPY rsikit/controller.py /opt/worker/rsikit/controller.py
COPY rsikit/sandbox/worker.py rsikit/sandbox/codec.py /opt/worker/rsikit/sandbox/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
ENTRYPOINT ["python", "-I", "-u", "-c", "import sys; sys.path.insert(0, '/opt/worker'); from rsikit.sandbox.worker import main; main()"]
