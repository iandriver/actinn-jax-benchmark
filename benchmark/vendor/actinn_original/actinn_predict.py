

import pandas as pd
import scanpy as sc
import os

import numpy as np


from . import actinn_utils as au


from anndata import AnnData


sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)

def celltype_predict_actinn(adata:AnnData, train_data_path:str, outpath:str, train_label_name:str='celltype',output_label_name:str='celltype', output_h5ad:bool=False):
    """\
    Takes anndata object and a training dataset.
    Parameters
    ----------
    adata
        AnnData object.
    train_data_path
        Full path to h5ad object with raw training data in X and train_label_name in .obs.
    outpath
        Path to output directory
    train_label_name
        Name of column present in train_data with labels for training.
    output_label_name
        Name to add to input adata.obs with the predicted celltype names
    output_h5ad
        If True a h5ad objected with predicted celltypes will be saved to the outpath with the file name "predicted_label.h5ad"

    Returns
    -------
    adata
        AnnData object with doublet score calculated.
    parameters
        model parameters
    """
    test_set = adata.to_df().T

    test_set = test_set.loc[test_set.sum(axis=1)>0, :]
    test_set.index = [s.upper() for s in test_set.index]
    uniq_index = np.unique(test_set.index, return_index=True)[1]
    test_set = test_set.iloc[uniq_index,]
    train_adata = sc.read_h5ad(train_data_path)
    train_set = train_adata.to_df().T
    train_set.index = [s.upper() for s in train_set.index]
    uniq_index = np.unique(train_set.index, return_index=True)[1]
    train_set = train_set.iloc[uniq_index,]
    train_label = train_adata.obs[train_label_name].values

    upper_dups = [a for a in test_set.index if (a.upper() in test_set.index and a !=a.upper())]
    uppers_to_remove = [a.upper() for a in upper_dups]
    test_set.drop(uppers_to_remove, axis=0, inplace=True)
    test_set.index = [s.upper() for s in test_set.index]
    barcode = list(test_set.columns)
    nt = len(set(train_label))
    train_set, test_set = au.scale_sets([train_set, test_set])
    type_to_label_dict = au.type_to_label_dict(train_label)
    label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
    print("Cell Types in training set:", type_to_label_dict)
    print("# Trainng cells:", train_label.shape[0])
    train_label = au.convert_type_to_label(train_label, type_to_label_dict)
    train_label = au.one_hot_matrix(train_label, nt)
    parameters = au.model(train_set, train_label, test_set, 0.0001, 50, 128, True)
    train_data_save_name = train_data_path.split('/')[-1].split('.h5ad')[0]
    test_predict = pd.DataFrame(au.predict_probability(test_set, parameters))
    test_predict.index = [label_to_type_dict[x] for x in range(test_predict.shape[0])]
    test_predict.columns = barcode
    test_predict.to_csv(os.path.join(outpath,train_data_save_name+"_predicted_probabilities.txt"), sep="\t")
    pp_max_df = pd.DataFrame(test_predict.max(), columns=['celltype_probability'])
    # Preserve original adata.obs order by using loc to align indices
    adata.obs[output_label_name+'_probability'] = pp_max_df.loc[adata.obs.index, 'celltype_probability']
    test_predict = au.predict(test_set, parameters)
    predicted_label = []
    for i in range(len(test_predict)):
        predicted_label.append(label_to_type_dict[test_predict[i]])
    predicted_label = pd.DataFrame({output_label_name:predicted_label}, index=barcode)
    predicted_label.to_csv(os.path.join(outpath,output_label_name+"_predicted_label.txt"), sep="\t", index=False)
    # Preserve original adata.obs order by using loc to align indices
    adata.obs[output_label_name] = predicted_label.loc[adata.obs.index, output_label_name]
    if output_h5ad:
        adata.write_h5ad(os.path.join(outpath,"predicted_label.h5ad"))

    return adata, parameters