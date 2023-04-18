FROM python:3.7-slim-buster

RUN apt-get update \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app
COPY ./MASK_RCNN /app

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

CMD [ "python", "mask_rcnn_prediction.py" ]
