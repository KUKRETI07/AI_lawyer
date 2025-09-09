FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential wget unzip curl \
&& pip install --upgrade pip \
&& pip install -r requirements.txt \
&& apt-get remove -y build-essential \
&& rm -rf /var/lib/apt/lists/*
ENV PORT=8000
EXPOSE $PORT
CMD ["bash", "start.sh"]
