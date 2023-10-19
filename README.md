Face Services
==========
> State-of-the-art human face processing.
> - [x] Face detection and analysis
> - [x] Face anonymization
> - [x] Face swapping
> - [x] Face enhancement
> - [ ] Face reenactment
> - [ ] Visual dubbing

> [!NOTE]
> This version works only with image file.

Installation
------------
Please use Python 3.10.

1. Install the requirements:
```
pip install -r requirements.txt
```
2. Download the pretrained models from [here](https://jwsite.sharepoint.com/:f:/r/sites/WHQ-MEPS-TMASyntheticMedia-Team/Shared%20Documents/Products/Face%20Services%20API/models?csf=1&web=1&e=ea1zHa).
Unzip and place the _**.assets**_ folder in the _**face_services**_ folder

Usage
-----
Run the command:
```
uvicorn face_services.app:app --host 0.0.0.0 --port 80 
```

Docker
-----
```
docker build . -t face_services
docker run -d --name face_services_container -p 80:80 face_services
```
