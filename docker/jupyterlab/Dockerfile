FROM jupyter/datascience-notebook:latest

WORKDIR /app

COPY /requirements.txt /

RUN pip install --upgrade pip &&\
    pip install -r /requirements.txt
