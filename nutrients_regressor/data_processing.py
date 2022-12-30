
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def split_idsintwo(ndata, ids = None, percentage = None, fixedids = None, seed = 123):

    if ids is None:
        ids = list(range(len(ndata)))

    if percentage is not None:
        if fixedids is None:
            idsremaining = pd.Series(ids).sample(int(ndata*percentage), random_state= seed).tolist()
        else:
            idsremaining = fixedids
        
        main_ids = [i for i in ids if i not in idsremaining]
    
    else:
        idsremaining = None
        main_ids = ids

    return main_ids, idsremaining


def retrieve_datawithids(data, ids):
    if len(ids) > 0:
        subset  = data.iloc[ids]
    else:
        subset = None

    return subset

def split_dataintotwo(data, idsfirst, idssecond):

    subset1 = data.iloc[idsfirst]
    subset2 = data.iloc[idssecond]

    return subset1, subset2


class SplitIds(object):

    
    def _ids(self):
        ids = list(range(self.ids_length))
        if self.shuffle:
            ids = pd.Series(ids).sample(n = self.ids_length, random_state= self.seed).tolist()

        return ids


    def _split_test_ids(self, test_perc):
        self.training_ids, self.test_ids = split_idsintwo(self.ids_length, self.ids, test_perc,self.test_ids, self.seed)


    def kfolds(self, kfolds, shuffle = True):
        kf = KFold(n_splits=kfolds, shuffle = shuffle, random_state = self.seed)

        idsperfold = []
        for train, test in kf.split(self.training_ids):
            idsperfold.append([list(np.array(self.training_ids)[train]),
                               list(np.array(self.training_ids)[test])])

        return idsperfold
    
    def __init__(self, ids_length = None, ids = None,val_perc =None, test_perc = None,seed = 123, shuffle = True, testids_fixed = None) -> None:
        
        
        self.shuffle = shuffle
        self.seed = seed
        if ids is None and ids_length is not None:
            self.ids_length = ids_length
            self.ids = self._ids()
            
        else:
            self.ids = ids
        
        self.val_perc = val_perc

        if testids_fixed is not None:
            self.test_ids = [i for i in testids_fixed if i in self.ids]
        else:
            self.test_ids = None

        self.training_ids, self.test_ids = split_idsintwo(self.ids_length, self.ids, test_perc,self.test_ids, self.seed)
        self.training_ids, self.val_ids = split_idsintwo(len(self.training_ids), self.training_ids, val_perc, seed = self.seed)
        
            

class SplitData(object):

    @property
    def test_data(self):
        return retrieve_datawithids(self.data, self.ids_partition.test_ids) 
    
    @property
    def training_data(self):
        return retrieve_datawithids(self.data, self.ids_partition.training_ids) 
    
    @property
    def validation_data(self):
        return retrieve_datawithids(self.data, self.ids_partition.val_ids) 

    def kfold_data(self, kifold):
        tr, val = None, None
        if self.kfolds is not None:
            if kifold <= self.kfolds:
                tr, val = split_dataintotwo(self.data, 
                                            idsfirst = self.ids_partition.kfolds(self.kfolds)[kifold][0], 
                                            idssecond = self.ids_partition.kfolds(self.kfolds)[kifold][1])

        return tr, val
        
    def __init__(self, df, splitids, kfolds = None) -> None:

        self.data = df
        self.ids_partition = splitids
        self.kfolds = kfolds


class ReadData(object):
    @property
    def data(self):
        return self.read_data()

    def read_data(self):

        if os.path.exists(self.path):
            spdata = pd.read_csv(self.path)

        else:
            raise ValueError("This path {} does not exists".format(self.path))
        
        return spdata

    def __init__(self, path) -> None:
        
        self.path = path

        
class ElementsData(ReadData):
    
    #@property
    def data_elements(self,elements:list = None):
        
        indextoselect = [list(self.data.columns).index(i) for i in elements if i in self.data.columns]
        
        if len(indextoselect) > 0:
            df = self.data.iloc[:,indextoselect]
        else:
            df = self.data

        return df

    def __init__(self, path, elements = None) -> None:
        self.elements = elements
        super().__init__(path)
        