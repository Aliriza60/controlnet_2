FROM nvidia/cuda:12.1.0-base-ubuntu20.04
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /controlnet
COPY controlnet.py /controlnet/
COPY requirements.txt /controlnet/
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install -r requirements.txt
CMD ["python","controlnet.py"]
