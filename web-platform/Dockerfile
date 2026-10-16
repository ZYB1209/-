FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MFL_STORAGE_DIR=/var/lib/mfl PORT=8080
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home mfl && mkdir -p /var/lib/mfl && chown mfl:mfl /var/lib/mfl
COPY --chown=mfl:mfl engine ./engine
COPY --chown=mfl:mfl web ./web
COPY --chown=mfl:mfl platform_server.py .
USER mfl
EXPOSE 8080
CMD ["python", "platform_server.py"]
