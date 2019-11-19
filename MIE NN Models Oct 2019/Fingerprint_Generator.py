# -*- coding: utf-8 -*-
"""
Fingerprint_Generator.py

@author: Elena Gelzintye & Timothy E H Allen
"""
#%%

# Import modules

import pandas as pd 
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdMolDescriptors

# Define paths and variables

'''
chemicals_path= binary activity file (.csv)
fingerprints_path= location for output (.csv)
fingerprint_length = length of genrerated fingerprint
fingerprint_radius = radius of gernerated fingerprint
'''

chemicals_path="/content/drive/My Drive/data/AR.csv"
fingerprints_path="/content/drive/My Drive/data/AR fingerprints ECFP6.csv"
fingerprint_length = 5000
fingerprint_radius = 3

smiles=pd.read_csv(chemicals_path)

# Define fingerprinting procedure and execute

def get_fingerprint(smiles):
    '''smiles dataframe'''
    
    bit_infos=[]
    rdkit_molecules=[Chem.MolFromSmiles(x) for x in smiles['SMILES']]
    rdkit_fingerprint=[]
    for mol in rdkit_molecules:
        bit_info={}
        fp=rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=fingerprint_radius, nBits=fingerprint_length, \
                                                                      bitInfo=bit_info).ToBitString() 
        bit_infos.append(bit_info)
        rdkit_fingerprint.append(fp)
        
    fingerprint_df=pd.DataFrame([np.array(list(x)).astype(int) for x in rdkit_fingerprint])
    
    return fingerprint_df, bit_infos

print('getting fingerprints')
fingerprints, substruct_lib=get_fingerprint(smiles)

fingerprints = pd.concat([fingerprints,smiles.drop(['SMILES'], axis = 1)], axis=1)

# Outputs fingerprints

fingerprints.to_csv(fingerprints_path, index = False)

#Endgame

print("END")

#%%
