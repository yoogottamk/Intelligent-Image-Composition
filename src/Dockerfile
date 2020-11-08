FROM python:3.8-slim

RUN apt update && \
    apt install -y libgl1-mesa-glx libglib2.0-0

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app/

ENV PYTHONPATH /app
WORKDIR /app

ENTRYPOINT ["python", "-u", "friendblend/main.py"]
