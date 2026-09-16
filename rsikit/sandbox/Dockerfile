FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3
COPY worker.py /worker.py
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
ENTRYPOINT ["python", "-I", "-u", "/worker.py"]
