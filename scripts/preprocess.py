import numpy as np

epitope_sequences = []
non_epitope_sequences = []

for seq in sequences:
    if seq.isupper():
        epitope_sequences.append(seq)
    else:
        non_epitope_sequences.append(seq)

# Create dataset with labels
data = []
labels = []

# Add epitope sequences with label 1
for epitope in epitope_sequences:
    data.append(epitope)
    labels.append(1)

# Add non-epitope sequences with label 0
for non_epitope in non_epito_sequences:
    data.append(non_epitope)
    labels.append(0)

# Define the amino acids and their corresponding indices for one-hot encoding
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_dict = {aa: idx for idx, aa in enumerate(amino_acids)}

# Function to one-hot encode a sequence
def one_hot_encode_sequence(sequence):
    # Initialize an empty matrix for the sequence
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)))
    
    # Fill the one-hot matrix with 1s for corresponding amino acids
    for idx, aa in enumerate(sequence):
        if aa in amino_acid_dict:  # Ensure the amino acid is valid
            one_hot_matrix[idx, amino_acid_dict[aa]] = 1
    
    return one_hot_matrix

# Create feature arrays (X) and labels (y)
X = []
y = []

# One-hot encode epitope sequences and append them to X, and set their labels to 1
for seq in epitope_sequences:
    encoded = one_hot_encode_sequence(seq)
    X.append(encoded.flatten())  # Flatten to 1D array
    y.append(1)

# One-hot encode non-epitope sequences and append them to X, and set their labels to 0
for seq in non_epitope_sequences:
    encoded = one_hot_encode_sequence(seq)
    X.append(encoded.flatten())
    y.append(0)

# Convert X and y into numpy arrays
X = np.array(X, dtype=object) 
y = np.array(y)

# Save the preprocessed data to disk
np.save('./data/X_featurized.npy', X)
np.save('./data/y_labels.npy', y)

print(f"Preprocessing complete. {len(X)} sequences processed.")
