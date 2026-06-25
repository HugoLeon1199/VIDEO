FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
EXPOSE 8080
CMD ["python", "server.py", "--port", "8080", "--preload"]
