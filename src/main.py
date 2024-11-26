from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Configuration du CORS
app.add_middleware(
    CORSMiddleware,
    # Liste des origines autorisées (vous pouvez ajuster selon vos besoins)
    allow_origins=["http://localhost:5173", "http://57.128.24.53:8294"],  # Origine de votre app Vue.js
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les headers
)

@app.get("/")
async def read_root():
  return {"status": "ok", "message": "API is working fine."}

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)