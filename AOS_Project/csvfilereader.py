import  pandas as pd
#https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
df= pd.read_csv("data_set/happiness.csv")
df=df[["Entity","Year","WorldHappinessReport"]]
df=df.loc[df['Year'] == 2017]
df.to_csv(r'data_set\update_happiness.csv', index=False)
print(df.head(5))