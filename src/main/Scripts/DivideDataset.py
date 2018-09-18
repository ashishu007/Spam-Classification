import pandas as pd

df = pd.read_csv("../Resources/sms_spam.csv")

df1 = df.sample(4500)
df2 = df.sample(1000)

df1.to_csv("../Resources/sms_spam_train.csv", index=0)
df2.to_csv("../Resources/sms_spam_test.csv", index=0)