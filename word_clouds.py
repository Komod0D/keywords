import re

import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud

#%%
words = stopwords.words("english")
import matplotlib.pyplot as plt

labelled = pd.read_csv("data/labelled.csv").fillna("")
by_label = labelled.groupby("label")
text_by_label = by_label["text"].apply(lambda x: ",".join(x))
word_clouds = {}
for i, text in text_by_label.iteritems():
    if i.strip() != "":
        word_cloud = WordCloud(stopwords={"https", "ID", "github", "use"}.union(words),
                               collocations=False,
                               background_color="white",
                               max_font_size=60)
        word_clouds[i] = word_cloud.generate(text)
#%%
for label, cloud in word_clouds.items():
    print(label)
    fixed = re.sub("/", "_", label)
    if label == "Data/ML":
        label = "data_ml"
    cloud.to_image().save("images/" + label.lower() + "_cloud.png")

