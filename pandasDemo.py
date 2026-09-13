import pandas as p
ages = p.Series([10,20,30],
               index = ["john","Alice","Bob"])
print(ages)
# print(ages[0])
# print(ages[1:4])
print(ages.index)
print(ages.values)


data={"name" : ['a' , 'b' , 'c'],
       "age" : [10 , 20,30],
        "salary":[30000 , 45000,10000]}
df = p.DataFrame(data)
print(df)

print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())

dc = p.read_csv("demo.csv")
print(dc)
print()
print(dc.head())
print()
print(dc.tail())
print()
print(dc.shape)

print(dc['Name'])
print()
print(dc[["Name" , "Score"]])
print()
print(dc.iloc[3,3])
print(dc.loc[2,"Course"])
print()
print()
print(dc["Score"].sum())
print()

print(dc["Score"].mean())
print()
print(dc["Score"].min())
print()
print(dc["Score"].max())
print()
print(dc["Score"].count())
print()
print(dc["Score"].median())
print()
print(dc["City"].value_counts())
print()
print(dc["City"].unique())
print()
print(dc["City"].nunique())
print()
print(dc["Score"] > 88)
print(dc["Name"] == "Aarav")


