import requests
from Bio import SeqIO

url = 'https://services.healthtech.dtu.dk/services/BepiPred-3.0/Data/IEDB/IEDBSeqsNotSharedAt20ID.fasta'
fasta_path = './data/epipred_data.fasta' 

def download_fasta_file():
    response = requests.get(url)
    if response.status_code == 200:
        with open(fasta_path, 'wb') as file:
            file.write(response.content)
        print('FASTA file download complete!')
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def parse_fasta_sequences():
    sequences = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        sequences.append(str(record.seq))
    return sequences

if __name__ == '__main__':
    download_fasta_file()
    sequences = parse_fasta_sequences()
    print(f"Parsed {len(sequences)} sequences.")
