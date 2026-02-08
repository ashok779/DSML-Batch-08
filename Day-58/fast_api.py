import json
from fastapi import FastAPI
from utils import pred_pipe


MODEL_PATH  = "/home/shailesh/Desktop/education/DSML-Batch-08/Day-57/sentiment_classifier.pkl"

app = FastAPI(title="Testing Fast API", version="1.0.0")


@app.get("/")
def welcome():
    return {
        "Hello" : "World"
    }
    

@app.post("/classify")
def classify_review(
    user_input : str,
) -> str:
    """
    Docstring for classify_review
    """
    sentiment = pred_pipe(user_input, MODEL_PATH)
    
    return json.dumps({"sentiment": sentiment})



