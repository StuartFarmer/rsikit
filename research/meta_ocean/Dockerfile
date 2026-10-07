FROM python:3.14-slim
RUN pip install --no-cache-dir 'numpy>=2.2,<3' 'gymnasium==1.3.0'
WORKDIR /opt
# Only the policy contract is available; no environments, data, research code or secrets.
COPY rsikit/policy.py /opt/rsikit/__init__.py
COPY research/meta_ocean/worker.py /opt/worker.py
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
ENTRYPOINT ["python", "-u", "/opt/worker.py"]
