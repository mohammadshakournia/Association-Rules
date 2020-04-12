import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

#open the source dataset file
f= open("marketBasket.txt","r")
f1=f.readlines()
#convert dataset file to a dataframe:
col=['Fruitveg','Freshmeat','Dairy','Cannedveg','Cannedmeat','Frozenmeal','Beer','Wine','Softdrink','Fish','Confectionery']
df=pd.DataFrame(columns=col);

#function for return each T or F
def first_char(s,f,n):
    return s[f:n]
#function for appending each boolean value of features to corresponding list
def add(s,f,e):
    for n in f1:
        m=first_char(n,f,e)
        s.append(m)
#create empty lists
Fruitveg=[]
Freshmeat=[]
Dairy=[]
Cannedveg=[]
Cannedmeat=[]
Frozenmeal=[]
Beer=[]
Wine=[]
Softdrink=[]
Fish=[]
Confectionery=[]
#append data to lists
add(Freshmeat,0,1)
add(Fruitveg,2,3)
add(Dairy,4,5)
add(Cannedveg,6,7)
add(Cannedmeat,8,9)
add(Frozenmeal,10,11)
add(Beer,12,13)
add(Wine,14,15)
add(Softdrink,16,17)
add(Fish,18,19)
add(Confectionery,20,21)
#convert lists to series to add them to dataframe
Fruitveg=pd.Series(Fruitveg)
Freshmeat=pd.Series(Freshmeat)
Dairy=pd.Series(Dairy)
Cannedveg=pd.Series(Cannedveg)
Cannedmeat=pd.Series(Cannedmeat)
Frozenmeal=pd.Series(Frozenmeal)
Beer=pd.Series(Beer)
Wine=pd.Series(Wine)
Softdrink=pd.Series(Softdrink)
Fish=pd.Series(Fish)
Confectionery=pd.Series(Confectionery)
#add series to dataframe
df['Fruitveg']=Fruitveg.values
df['Freshmeat']=Freshmeat.values
df['Dairy']=Dairy.values
df['Cannedveg']=Cannedveg.values
df['Cannedmeat']=Cannedmeat.values
df['Frozenmeal']=Frozenmeal.values
df['Beer']=Beer.values
df['Wine']=Wine.values
df['Softdrink']=Softdrink.values
df['Fish']=Fish.values
df['Confectionery']=Confectionery.values
#change boolean data to int as 0,1
for n in col:
    df[n]=(df[n]=='T').astype(int)
    
#summation of count of each products
print(df.sum(axis = 0, skipna = True))

def apriori_function(input_df,min_sup,min_conf,min_lift):
    
    # make an apriori model with min_sup 0.1
    frequent_itemsets = apriori(input_df, min_support=min_sup, use_colnames=True) 
    # add lenght of each rules
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    #frequent_itemsets=frequent_itemsets[frequent_itemsets['length'] >= 3]
    #create an association rule and uses lift metric to shows the interest of it 
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules.head()
    #show just rules with lift>=1 and cofidence value more than 0.5
    Apriori_Result=rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_conf) ] 
    
    #show maximum of rules in each case of support ,lift ,confidence , leverage and confidence
    for e in ['confidence','lift','support','leverage','conviction']:
        d=Apriori_Result[e].idxmax() #return index of maximum value 
        m=Apriori_Result.at[d,'antecedents']#return name of product of maximum value(index) 
        m1=Apriori_Result.at[d,'consequents']#return consequnet name of product of maximum value(index) which might bought after antecedents
        print('\n#####################################\n')
        print('MAX Value of ',e,' :')
        print(m,m1,'\n')
    return Apriori_Result

Apriori_Result=apriori_function(df,0.1,0.5,1)


def fp_growth_function(input_df,min_sup):
    FP_Growth_Result=fpgrowth(input_df, min_support=0.1, use_colnames=True)
    FP_Growth_Result['length'] = FP_Growth_Result['itemsets'].apply(lambda x: len(x))
    return FP_Growth_Result

FP_Growth_Result=fp_growth_function(df,0.1)

del Beer,Cannedmeat,Cannedveg,Confectionery,Dairy,Fish,Frozenmeal,Freshmeat,Fruitveg,Softdrink,Wine,col,f1,n