from fastapi import FastAPI
import joblib
import numpy as np

# Define the amino acids and their corresponding indices
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_dict = {aa: idx for idx, aa in enumerate(amino_acids)}

# Function to one-hot encode a sequence
def one_hot_encode_sequence(sequence):
    # Initialize an empty matrix for the sequence
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)))
    
    for idx, aa in enumerate(sequence):
        if aa in amino_acid_dict:  # Ensure the amino acid is valid
            one_hot_matrix[idx, amino_acid_dict[aa]] = 1
    
    return one_hot_matrix

# Load the trained model using joblib
model = joblib.load("../model/epitope_model.joblib")

# Create FastAPI app
app = FastAPI()

# Featurization logic for incoming sequences
def featurize(sequence: str):
    # One-hot encode the sequence and flatten it to 1D
    encoded_sequence = one_hot_encode_sequence(sequence)
    flattened_features = encoded_sequence.flatten()  # Flatten to 1D
    return flattened_features

@app.post("/predict/")
def predict(sequence: str):
    # Featurize the input protein sequence
    features = featurize(sequence)
    
    # Predict using the trained model
    prediction = model.predict(features.reshape(1, -1))  # Reshape for model input
    
    # Return the prediction
    return {"prediction": int(prediction[0])}
