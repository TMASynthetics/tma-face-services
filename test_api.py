import uvicorn
from face_services.app import app
uvicorn.run(app, host='0.0.0.0', port=80)