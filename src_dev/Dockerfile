FROM ubuntu:latest
ENV SHELL=/bin/bash

WORKDIR /app/

COPY . /app

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git 

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install pandas==1.4.1 tqdm
RUN pip install torch==1.11.0 torchaudio==0.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install Flask==2.1.1 Flask-Cors==3.0.10



# EXPOSE 80
ENTRYPOINT ["python"]
CMD ["app.py"]