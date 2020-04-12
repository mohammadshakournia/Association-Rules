import scipy.io
from nltk import flatten
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

number_of_rows=1000

def Read_data(file):
    mat = scipy.io.loadmat(file)
    D=flatten(list(mat.values()))
    Data=np.array(D[2])
    del D,mat
    return Data

def array_to_df(array):
    df=pd.DataFrame(data=array[0:number_of_rows,1:],columns=array[0,1:])
    return df


def binarize_dataframe(df):
    df1 = (pd.get_dummies(df.astype(str), prefix='', prefix_sep='')
             .max(level=0, axis=1))
    
    df1.columns = df1.columns.astype(int)    
    
    df1 = (df1.reindex(columns=range(1, df1.columns.max() + 1), fill_value=0)
              .add_prefix('Product '))
    return df1

Data=Read_data('Data.mat')
Data=np.delete(Data, 0, axis=0)
Data=np.delete(Data, 0, axis=1)

Basket=array_to_df(Data)

Basket_bin=binarize_dataframe(Basket)

frequent_itemsets = apriori(Basket_bin, min_support=0.008, use_colnames=True) 

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

Apriori_Reslut=rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ] 

for e in ['confidence','lift','support','leverage','conviction']:
    d=Apriori_Reslut[e].idxmax() #return index of maximum value 
    m=Apriori_Reslut.at[d,'antecedents']#return name of product of maximum value(index) 
    m1=Apriori_Reslut.at[d,'consequents']#return consequnet name of product of maximum value(index) which might bought after antecedents
    print('\n#####################################\n')
    print('MAX Value of ',e,' :')
    print(m,m1,'\n')
    
