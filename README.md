# tma-face-services
 
docker build . -t face_services
docker run -d --name face_services_container -p 80:80 face_services