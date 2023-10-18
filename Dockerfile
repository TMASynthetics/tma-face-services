# 
FROM python:3.10

RUN apt-get update && apt-get install curl ffmpeg libsm6 libxext6  -y

ENV QT_DEBUG_PLUGINS=1
ENV QT_QPA_PLATFORM=xcb

RUN pip install -U pip wheel cmake

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./face_services /code/face_services
COPY ./gfpgan /code/gfpgan
# 
CMD ["uvicorn", "face_services.app:app", "--host", "0.0.0.0", "--port", "80"]
