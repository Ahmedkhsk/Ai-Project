from fastapi import FastAPI, File, UploadFile
from model_predict import predict_pneumonia

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_pneumonia(contents)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)