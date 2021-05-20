import pandas as pd

descriptions = pd.read_csv("repos_description_full.csv")
descriptions.index = descriptions["repo"]

correct = pd.read_csv("correct.csv")
correct.index = correct["repo"]
correct = correct[["label"]]

check = pd.read_csv("to_check.csv")
check.index = check["repo"]
check = check[["label"]]

new = correct.append(check)
new = new.join(descriptions)[["repo", "text", "label"]].drop_duplicates()

print(len(new))
print(new.columns)

new.to_csv("new.csv", index=False)
print(len(new))
labelled = pd.read_csv("labelled.csv")
labelled.index = labelled["repo"]
print(len(labelled))
labelled.to_csv("labelled_old.csv", index=False)

labelled = labelled.append(new)
print(len(labelled))
print(labelled)

labelled.to_csv("labelled.csv", index=False)
