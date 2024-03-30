import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results




@app.post("/predict")
async def root(raw_text: str):
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)

    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']
    probability_dict = {emotion: prob for emotion, prob in zip(emotions, probability[0])}

    return JSONResponse(content={})