Face Services
==========
> State-of-the-art human face processing capabilities.


Installation
------------
Please use Python 3.10.

1. Install the requirements:
```
pip install -r requirements.txt
```
2. Download the pretrained models from [here](https://docs.facefusion.io](https://jwsite.sharepoint.com/:f:/r/sites/WHQ-MEPS-TMASyntheticMedia-Team/Shared%20Documents/Products/Face%20Services%20API/models?csf=1&web=1&e=lglSBe)).
- Place the **.assets** folder in the face_services folder
- Place the **gfpgan** folder at the root of the project

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
