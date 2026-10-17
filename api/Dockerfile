FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
RUN pip install --no-cache-dir flask
COPY app.py /app/app.py
EXPOSE 5000
CMD ["python", "/app/app.py"]
