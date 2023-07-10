FROM python:3.10
WORKDIR .
COPY requirements.txt requirements.txt
RUN apt-get update -y
RUN apt-get install libblas-dev liblapack-dev -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scipy
RUN pip install llvmlite
RUN pip install scikit-learn
RUN pip install tensorboard
RUN pip install tensorboard-data-server
RUN pip install tensorflow-estimator
RUN pip install tensorflow-io-gcs-filesystem
RUN pip install --upgrade pip && pip install --no-cache-dir torch
RUN pip install --upgrade pip && pip install --no-cache-dir tensorflow
COPY . .
ENV TZ Europe/Moscow
CMD ["python3", "-u", "app.py"]
