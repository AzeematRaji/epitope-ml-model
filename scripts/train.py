import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# pad sequences to a fixed length
def pad_sequences_custom(sequences, maxlen, padding_value=0):
    padded_sequences = np.full((len(sequences), maxlen), padding_value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded_sequences[i, :length] = seq[:length]  # truncate if too long
        
    return padded_sequences

X_padded_custom = pad_sequences_custom(X, maxlen=4096)

X_train, X_test, y_train, y_test = train_test_split(X_padded_custom, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = xgb.XGBClassifier(scale_pos_weight=len(y) / (2 * np.bincount(y)[1]))
model.fit(X_train_res, y_train_res)

joblib.dump(model, "./models/epitope_model.pkl")

print("Model training complete and saved as epitope_model.pkl")
