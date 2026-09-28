FROM rsikit:local
USER root
RUN pip install --no-cache-dir pokerkit==0.7.6 numba==0.64.0
COPY poker/__init__.py poker/game.py poker/worker.py poker/service.py poker/candidate.py poker/codec.py /opt/poker/
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ENTRYPOINT ["python", "-u", "-m", "poker.service"]
CMD []
