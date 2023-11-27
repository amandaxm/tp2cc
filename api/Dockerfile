FROM python:3.9-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "-m" , "flask", "run","--host=0.0.0.0","--port=32165"]

