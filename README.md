## Epitope Prediction Model

Cloud-hosted ML model for predicting epitope regions on protein sequences, making it ccessible, fast, and researcher-friendly.

A model that predicts the epitope regions on a protein sequence, helping researchers identify which parts of an antigen can trigger an immune response, which is of primary importance in vaccine and antibody development.

### Table of Contents
1. [Project Description](#project-description)
2. [Setting the Environment](#setting-the-environment)
3. [Project Structure](#project-structure)
4. [Download a Dataset](#download-a-dataset)
5. [Featurise the Data](#featurise-the-data)
6. [Training the Model](#training-the-model)
7. [Model Evaluation](#model-evaluation)
8. [Model Summary](#model-summary)
9. [Using the Model Later](#using-the-model-later)
10. [Host the Model Directly in the Cloud](#host-the-model-directly-in-the-cloud)
11. [Containerize the Model](#containerize-the-model)
12. [Automate the Deployment with GitHub Actions](#automate-the-deployment-with-gitHub-actions)
13. [Conclusion](#conclusion)

### Project Description

This project provides a ML model that predicts epitope regions within protein sequences, the specific parts of an antigen that can trigger an immune response. Such predictions are vital in vaccine design, antibody development, and immunodiagnostics. Built with XGBoost and deployed on the cloud, this tool is to ensure accessibility for non-technical users, providing a simple interface for making predictions without needing data science expertise.

The goal is to streamline epitope discovery and accelerate immunological research in both academic and low-resource settings.

### Setting the Environment:
#### Prerequisites
- linux OS
- gcc compiler, `sudo apt install build-essential`
- python 3.8 and above, [install here](https://www.python.org/).
- conda installed, use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Installer
- git installed, [install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- github cli installed, [Github CLI](https://cli.github.com/)
- gitlfs installed and activated, [Git LFS](https://git-lfs.github.com/)
- docker installed, [Docker](https://docs.docker.com/engine/install/)

#### Create an isolated environment

All the prerequisites must be satisfied. 

create a conda environment:
  
`conda create -n eptiope python=3.12`

activate the environment:
  
`conda activate epitope`

### Project Structure
``` text
epitope-ml-model/
│
├── app/                                    
│   └── main.py                  
│
├── model/                                                    
│   ├── epitope_model.joblib               
│   └── epitope_model.pkl              
│
├── data/                        
│   ├── epipred_data.fasta
│   ├── X_featurized.npy
│   └── y_labels.npy
│
├── notebooks/                    
│   └── epitope_exploration.ipynb
│
├── scripts/                        
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train.py
│
├── Dockerfile                    
├── requirements.txt              
├── .gitignore                   
├── README.md                     
└── .github/
    └── workflows/
        └── deployment.yaml
```

### Download a dataset

#### Background of Data

__Dataset__: _IEDB Linear Epitope Data (Reduced)_
This dataset focuses on identifying regions within antigen protein sequences where antibodies can bind and trigger an immune response.

__Task__: Given a protein sequence, predict the epitope regions.

- Epitope - Uppercase
- Non epitopee - Lowercase
  
__Size__: 3560 sequences

__Source__: [BepiPred 3.0 - DTU Health Tech](https://services.healthtech.dtu.dk/services/BepiPred-3.0/)

__Model Output__:

Binary prediction for each amino acid in the sequence:

- Epitope - 1
- Non epitopee - 0

#### Steps to downloading dataset

1- Used Jupyter Notebook for Interactive Exploration

Check the notebook: notebooks/epitope_exploration.ipynb

Set up Jupyter Notebook if not installed already:
```
# using conda
conda install -c conda-forge notebook

# using pip
pip install notebook
```
Launch Jupyter:

`jupyter notebook`

Then run the following code to download and explore the dataset

2- To retrieve dataset from BepiPred, 

```
import requests
url = 'https://services.healthtech.dtu.dk/services/BepiPred-3.0/Data/IEDB/IEDBSeqsNotSharedAt20ID.fasta'
response = requests.get(url)
with open('../data/epipred_data.fasta', 'wb') as file:
    file.write(response.content)
print('Download complete!')
```

load the dataset:
```
from Bio import SeqIO
fasta_file = '../data/epipred_data.fasta'
sequences = []
for record in SeqIO.parse(fasta_file, 'fasta'):
    sequences.append(str(record.seq))
```

3- Use Python Script for Automation

To make workflow reproducible for local setup, use the script at: scripts/retrieve.py.

To run it:

`python ./scripts/retrieve.py`

### Featurise the data

__Featuriser__: _One-hot encoding_

One-hot encoding was used to featurize the protein sequences because it provides a straightforward, non-biased representation of each amino acid. It allows the model to treat each residue equally without assuming any ordinal relationship, making it ideal for sequence-based tasks where the identity of each amino acid matters more than its "rank" or magnitude.

#### Steps to featurise the data:

1. Sequence Labeling and Dataset Construction
```
epitope_sequences = []
non_epitope_sequences = []

for seq in sequences:
    if seq.isupper():
        epitope_sequences.append(seq)
    else:
        non_epitope_sequences.append(seq)

# create a dataset with labels
data = []
labels = []

for epitope in epitope_sequences:
    data.append(epitope)
    labels.append(1)

for non_epitope in non_epitope_sequences:
    data.append(non_epitope)
    labels.append(0)
```

1. One-Hot Encoding of Amino Acid Sequences
```
import numpy as np

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_dict = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode_sequence(sequence):
    # Initialize an empty matrix for the sequence
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)))
    
    for idx, aa in enumerate(sequence):
        if aa in amino_acid_dict:  # Ensure the amino acid is valid
            one_hot_matrix[idx, amino_acid_dict[aa]] = 1
    
    return one_hot_matrix
```

1. Encoding and Dataset Finalization
```
# Combine and label
X = []
y = []

for seq in epitope_sequences:
    encoded = one_hot_encode_sequence(seq)
    X.append(encoded.flatten())  # flatten to 1D
    y.append(1)

for seq in non_epitope_sequences:
    encoded = one_hot_encode_sequence(seq)
    X.append(encoded.flatten())
    y.append(0)
```

1. Prepare Data for Modeling
```
X = np.array(X, dtype=object)  # Use object type because sequences is of different lengths
y = np.array(y)
```

1. Save the featurised data for reference
```
np.save('../data/X_featurized.npy', X)
np.save('../data/y_labels.npy', y)
```
1. Use Python Script for Automation

To make workflow reproducible for local setup, use the script at: scripts/preprocess.py.

To run it:

`python ./scripts/preprocess.py`


### Training the model

Label Distribution:
- Label 0 (Non-epitope): 3,522 sequences
- Label 1 (Epitope): 38 sequences

Given the highly imbalanced nature of the dataset, XGBoost was chosen for its robustness in handling class imbalance through techniques like scale weighting and built-in regularization. Additionally, scikit-learn was used for data preprocessing, model evaluation, and saving the trained model.

#### Steps to train a model
1. Sequence Padding for Uniform Input Size
```
def pad_sequences_custom(sequences, maxlen, padding_value=0):
    padded_sequences = np.full((len(sequences), maxlen), padding_value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded_sequences[i, :length] = seq[:length]  # truncate if too long
        
    return padded_sequences

X_padded_custom = pad_sequences_custom(X, maxlen=4096)
```

1. Split train and test data
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_padded_custom, y, test_size=0.2, random_state=42, stratify=y)
```
Training set size: 2848
Test set size: 712

1. SMOTE Oversampling & Model Training
   
Due to the extreme class imbalance, SMOTE was applied only to the training set to generate synthetic examples of the minority class

```
import xgboost as xgb
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = xgb.XGBClassifier(scale_pos_weight=len(y) / (2 * np.bincount(y)[1]))
model.fit(X_train_res, y_train_res)
```

1. Use Python Script for Automation

To make workflow reproducible for local setup, use the script at: scripts/train.py.

To run it:

`python ./scripts/train.py`

### Model Evaluation

Once the model is trained, it’s crucial to evaluate its performance using different metrics. Accuracy, precision, recall, F1 score, and AUC is used. 

#### Steps to evaluate the model

1. Model prediction
```
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

1. Model Evaluation Metrics
```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

auroc = roc_auc_score(y_test, y_prob)
```
Accuracy: 0.9860

Precision: 0.2500

Recall: 0.1250

F1-score: 0.1667

AUROC: 0.9664


Classification Report:

              precision    recall  f1-score   support

           0       0.99      1.00      0.99       704
           1       0.25      0.12      0.17         8

    accuracy                           0.99       712
    macro avg       0.62      0.56      0.58       712
    weighted avg    0.98      0.99      0.98       712


1. Save the trained the model

```
import joblib
joblib.dump(model, "../models/epitope_model.pkl")
```
### Model Summary

After training the XGBoost model, it achieved:

Accuracy: 98.60%

Precision: 25.00%

Recall: 12.50%

F1-score: 16.67%

ROC-AUC Score: 0.7078

To address class imbalance, the following adjustments were made:
- Stratified sampling while splitting the dataset
- Increased scale_pos_weight to favor the minority class
- SMOTE was used to generate synthetic samples for the minority class in the training set.

#### Improvement:
- To experiment with more advanced balancing techniques
- Exploring other models or ensemble methods could provide further performance gains, especially for the minority class.

### Using the model later

To use the model, the saved model has to be loaded;

```
import joblib
model = joblib.load("../model/epitope_model.pkl")
```

Make predictions

`y_pred = model.predict(x_test)`

### Host the Model Directly in the Cloud
1.Launch an EC2 Instance on AWS

- Choose an appropriate instance type (e.g., t3.medium because of data size).

- Configure the security group to allow inbound traffic on port 8000 or 0.0.0.0 for testing.

- SSH into the instance and set up the [environment](#setting-the-environment)

1. Set Up FastAPI Server:
   
Create a simple FastAPI application that loads the saved model and exposes an endpoint for predictions.

`uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

3. Access the API from the Browser
   
Navigate to:

`http://<ec2-public-ip>:8000/docs`

This opens the interactive Swagger UI where you can test your API.

⚠️ While this method works for quick testing, it's not scalable or production-ready.
Let's take it a step further by containerizing the application, making it portable, reproducible, and easier to deploy at scale.

### Containerize the Model
1. Create a `requirements.txt`

```
fastapi
uvicorn
joblib
numpy
scikit-learn
xgboost
```

2. Create a `Dockerfile`

```
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. Build and Run the Docker Image in Cloud environment
```
docker build -t epitope-model-api .
docker run -p 8000:8000 epitope-model-api
```

4. Access the API from the Browser
   
Navigate to:

`http://<ec2-public-ip>:8000/docs`

App is containerized making it portable and consistent across environments. It successfully eliminated the need to manually set up Python environments in different systems. Now Lets automate this!

### Automate the Deployment with GitHub Actions
1. Create a `github/workflows/deployment.yaml`
```
name: Build and Deploy to EC2

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest  
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}  
        password: ${{ secrets.DOCKER_PASSWORD }}  

    - name: Build Docker image
      run: docker build -t ${{ secrets.DOCKER_USERNAME }}/epitope-model-api:latest .

    - name: Push Docker image
      run: docker push ${{ secrets.DOCKER_USERNAME }}/epitope-model-api:latest

    - name: Deploy to EC2 via SSH
      uses: appleboy/ssh-action@v1.0.0
      with:
        host: ${{ secrets.EC2_PUBLIC_IP }}
        username: ubuntu
        key: ${{ secrets.EC2_SSH_KEY }}
        script: |
          docker pull ${{ secrets.DOCKER_USERNAME }}/epitope-model-api:latest
          docker stop epitope-api || true
          docker rm epitope-api || true
          docker run -d --name epitope-api -p 8000:8000 ${{ secrets.DOCKER_USERNAME }}/epitope-model-api:latest
```

This workflow triggers everytime there is a push to the repository. It will automatically containerize the app, pushed to dockerhub and deploy it on EC2 that is already provisioned. This eliminate the need to deploy manually deploy everytime there is change to the code

#### Suggestions
- Automate EC2 Provisioning: While it is currently deploying to an existing EC2 instance, we could automate the provisioning of the EC2 instance using Terraform or AWS CloudFormation. This ensures that infrastructure is version-controlled and reproducible.
- GitHub secrets must be properly configured in the repository to handle sensitive data like DockerHub credentials and EC2 SSH keys. This is crucial for security.
  
### Conclusion
This project aims to provide a solution that helps researchers in low-resource settings predict epitope regions in protein sequences, a crucial task in vaccine and antibody development. The trained XGBoost model, which predicts whether a part of a protein sequence is an epitope or not, has been deployed as an easy-to-use API. This allows researchers, even without data science knowledge, to quickly make predictions using only a protein sequence as input. By automating this process and making it accessible via the cloud, researchers in under-resourced areas can save time and resources that would otherwise be spent on manual analysis.

Also, the use of cloud automation in this project ensures that the model is scalable, reliable, and available to users anywhere. By hosting the model in the cloud, we can provide researchers with seamless access to the predictions without the need for expensive local infrastructure. Cloud-based solutions also make it easier to maintain and update the model as new data becomes available, ensuring that the tool remains effective over time. Automating the deployment of the model using GitHub Actions ensures that the latest version of the model is always deployed efficiently, reducing the chances of errors and making the system more robust.

Lastly, the combination of machine learning model deployment in the cloud and the automation of the process helps make advanced research tools available to everyone, regardless of their location or available resources, ultimately supporting the advancement of science and healthcare.




