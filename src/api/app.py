#\src\api\app.py


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import forecast 

app = FastAPI(title="Quake Forecast API")

# CORS (modo desarrollo: permitir todo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # permite cualquier origen (incluye file:// -> origin null)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
