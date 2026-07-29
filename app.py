import json
import os
import time
import webbrowser

import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq

# Load environment variables from a local .env file when present.
load_dotenv(override=True)

app = FastAPI(title="RoboStrategy — FTC Optimizer")
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_landing():
    """Serves landing.html as the homepage."""
    if not os.path.exists("landing.html"):
        raise HTTPException(status_code=404, detail="landing.html not found.")
    with open("landing.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/index.html")
async def serve_index():
    return FileResponse("index.html")


@app.get("/app", response_class=HTMLResponse)
async def get_main_app():
    """Serves index.html when the user opens the main app."""
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/randomize-obstacles")
async def randomize_obstacles(rx: float = 0.5, ry: float = 0.5, tx: float = 3.0, ty: float = 3.0):
    np.random.seed(int(time.time() * 1000) % 1000000)
    random_obs = []
    while len(random_obs) < 10:
        obs_x = float(np.random.uniform(0.5, 3.1))
        obs_y = float(np.random.uniform(0.5, 3.1))
        obs_r = float(np.random.uniform(0.15, 0.35))
        if np.hypot(obs_x - rx, obs_y - ry) > 0.5 and np.hypot(obs_x - tx, obs_y - ty) > 0.5:
            random_obs.append([round(obs_x, 2), round(obs_y, 2), round(obs_r, 2)])
    return {"obstacles": random_obs}


@app.post("/api/upload-heuristics")
async def upload_heuristics(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        return {"status": "success", "columns": list(df.columns), "preview": df.head(5).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat")
async def chat_copilot(payload: dict):
    current_key = os.getenv("GROQ_API_KEY", "").strip()
    if not current_key:
        async def err_gen():
            yield "data: " + json.dumps({"error": "GROQ_API_KEY is missing."}) + "\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    client = Groq(api_key=current_key)
    messages = payload.get("messages", [])

    def stream_generator():
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                stream=True,
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    is_render = bool(os.getenv("RENDER")) or bool(os.getenv("PORT"))
    host = os.getenv("HOST", "0.0.0.0" if is_render else "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    url = f"http://{host}:{port}"

    print(f"\nRoboStrategy Landing Page live at: {url}")
    if not is_render:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, reload=not is_render)

