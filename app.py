from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import uvicorn

# Load model
model = joblib.load("sales_model.joblib")

# Initialize FastAPI
app = FastAPI(title="Sales Prediction App")

# Templates folder
templates = Jinja2Templates(directory="templates")

# 1️⃣ HTML Form route (GET)
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": None,
        "tv": None,
        "radio": None,
        "newspaper": None
    })

# 2️⃣ HTML Form submission (POST)
@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request,
                    tv: float = Form(...),
                    radio: float = Form(...),
                    newspaper: float = Form(...)):

    # Convert inputs to numpy array
    data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(data)[0]

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "prediction": round(float(prediction), 2),
            "tv": tv,
            "radio": radio,
            "newspaper": newspaper
        }
    )

# 3️⃣ JSON API endpoint
@app.post("/predict")
async def predict_api(tv: float, radio: float, newspaper: float):
    data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(data)[0]
    return {
        "tv": tv,
        "radio": radio,
        "newspaper": newspaper,
        "prediction": round(float(prediction), 2)
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
