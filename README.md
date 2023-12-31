Face Services API
==========
> State-of-the-art human face processing.
> - [ ] Face detection and analysis
> - [ ] Face anonymization
> - [ ] Face swapping
> - [ ] Face enhancement
> - [ ] Face reenactment
> - [x] Visual dubbing

Installation
------------
Please use Python 3.10.

1. Install the requirements:
```
pip install -r requirements.txt
```
2. Download the pretrained models from [here](https://jwsite.sharepoint.com/:u:/r/sites/WHQ-MEPS-TMASyntheticMedia-Team/Shared%20Documents/Products/Face%20Services%20API/Face%20Services%20API%20-%20v1.1/models.zip?csf=1&web=1&e=Hscuxo).
Unzip and place the _**models**_ folder in the _**face_services**_ folder

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
