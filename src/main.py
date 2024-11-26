from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
  return {"status": "ok", "message": "API is working fine."}

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)