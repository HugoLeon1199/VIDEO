FROM vastai/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install Python deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install -q --no-cache-dir -r /workspace/requirements.txt

# Copy the klein worker package
COPY . /workspace/vast_worker_klein/

# Model will be downloaded at runtime by model_loader; leave /workspace/model_klein empty
RUN mkdir -p /workspace/model_klein

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

EXPOSE 8081

CMD ["/usr/bin/python3", "-m", "vast_worker_klein.gpu_worker", "--port", "8081", "--host", "0.0.0.0", "--model-path", "/workspace/model_klein"]
