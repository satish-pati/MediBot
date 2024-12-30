FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app
  
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
ENV XDG_CACHE_HOME=/app/.cache
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip espeak -y && pip install -r requirements.txt

EXPOSE 7860
CMD ["python3", "app.py","--host", "0.0.0.0", "--port", "7860"]