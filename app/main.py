from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.requests import Request

# Define amino acid list and encoding function
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_dict = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode_sequence(sequence):
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)))
    for idx, aa in enumerate(sequence):
        if aa in amino_acid_dict:
            one_hot_matrix[idx, amino_acid_dict[aa]] = 1
    return one_hot_matrix

# Custom padding function used during model training
def pad_sequences_custom(sequences, maxlen, padding_value=0):
    padded_sequences = np.full((len(sequences), maxlen), padding_value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded_sequences[i, :length] = seq[:length]  # truncate if too long
        
    return padded_sequences

# Load your model
model = joblib.load("./model/epitope_model.joblib")

app = FastAPI()

# Request model for FastAPI input
class SequenceInput(BaseModel):
    sequence: str

def featurize(sequence: str, maxlen=4096):
    encoded_sequence = one_hot_encode_sequence(sequence)
    flattened_features = encoded_sequence.flatten()
    padded_features = pad_sequences_custom([flattened_features], maxlen=maxlen)
    return padded_features[0]  # return as a 1D array

# FastAPI endpoint for predicting
@app.post("/predict/")
def predict(input: SequenceInput):
    features = featurize(input.sequence)
    prediction = model.predict(features.reshape(1, -1))
    return {"prediction": int(prediction[0])}

# FastAPI endpoint for HTML form
@app.get("/", response_class=HTMLResponse)
def form_get():
    return """
    <html>
        <head>
            <title>Epitope Predictor</title>
        </head>
        <body style="font-family: sans-serif;">
            <h2>Enter Protein Sequence</h2>
            <form action="/predict_html" method="post">
                <textarea name="sequence" rows="4" cols="50" placeholder="Enter sequence here..."></textarea><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

# FastAPI endpoint for processing HTML form data
@app.post("/predict_html", response_class=HTMLResponse)
async def form_post(request: Request):
    form = await request.form()
    sequence = form.get("sequence")
    features = featurize(sequence)
    prediction = model.predict(features.reshape(1, -1))
    return f"<h2>Prediction: {int(prediction[0])}</h2><br><a href='/'>Try another</a>"
