FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
RUN pip install --no-cache-dir flask
COPY app.py analyst.html /app/
EXPOSE 5000
CMD ["python", "/app/app.py"]
