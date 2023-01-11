import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs

df = pd.read_csv('D:/Doctorado/Articulo Raul/TF_DB_clean_pathway.csv')

matrix = np.zeros((5390,5390))
fps = []
for i in df['SMILES']:
    m=Chem.MolFromSmiles(i)
    fps.append(Chem.RDKFingerprint(m, fpSize = 512))
for i in range(len(fps)):
    for j in range(len(fps)):
        matrix[i,j] = DataStructs.FingerprintSimilarity(fps[i],fps[j])

disimilar = []
for i in range(len(fps)):
    mini = np.ones((10,))
    mini_idx = np.zeros((10,))
    for j in range(len(fps)):
        if matrix[i][j] < np.max(mini):
            a = np.where(mini == np.max(mini))[0][0]
            mini[a] = matrix[i][j]
            mini_idx[a] = j
    b = np.random.choice(mini_idx)
    disimilar.append(b)

fpm = np.load('D:/Doctorado/Articulo Raul/fingerprints_matrix_noneg.npy')
fp_m = np.zeros((len(fpm)*2,512))

for i in range(len(fpm)):
    fp_m[2*i] = fpm[i]
    fp_m[2*i+1] = fpm[int(disimilar[i])]
np.save('D:/Doctorado/Articulo Raul/new_fingerprints_matrix.npy',fp_m)