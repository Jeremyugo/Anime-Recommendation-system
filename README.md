# Anime Recommender

This repository contains code for building recommendation systems using various techniques.

## Project Organization

This project is organized into the following directories: `app`, `data`, `docker`, `model`, `notebooks`, and `scripts`.
- `app`: holds python files for streamlit app
- `data`: contains both raw and processed data 
- `docker`: contains DockerFiles for docker containers 
- `model`: contains model artifacts
- `notebooks`: holds notebooks for data preparation and model building
- `scripts`: contains python files for model inferencing


# Getting Started

Please ensure you have: 
1. docker installed in your machine. If not, you can check [Get Docker](https://docs.docker.com/get-docker/) to install docker
2. anaconda / miniconda in your machine. If not, please check the [installation documentation](https://docs.anaconda.com/free/anaconda/install/index.html)

## Setting up Environment
It is recommended to use conda for this project. To create a conda environment, run the following:
```
conda create --name <env-name> python=3.11
```

replace `<env-name>` with the name of your conda environment. Then run

```
conda activate <env-name>
```

## Building docker containers

Ensure you are in the home directory containing the `dockercompose.yml` file, then run the following in your terminal:

```
docker compose build --parallel
docker compose up
```

After the successful build, we can access the running services using the following links:

|Service |URL|
|-----|--------|
|JupyterLab|http://localhost:8888/|
|Streamlit  |http://localhost:8501/|


# Anime Recommender App

![image](https://github.com/Jeremyugo/Anime-Recommender/assets/36512525/85b91cfa-1a33-4d5e-b377-c31caabba1db)

### TensorFlow Recommenders

![image](https://github.com/Jeremyugo/Anime-Recommender/assets/36512525/e1dc02ad-7f85-4295-bdb7-a5985ad67275)

### jaccard

![image](https://github.com/Jeremyugo/Anime-Recommender/assets/36512525/f9c27061-fa0a-4821-a249-d2101f0ecbd4)


