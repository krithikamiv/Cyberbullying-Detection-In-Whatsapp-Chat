#!/usr/bin/env python
# coding: utf-8

# In[34]:


from __future__ import division
import codecs
import re
import copy
import collections

import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


nltk.download('stopwords')


# In[36]:


from nltk.corpus import stopwords


# In[37]:


with codecs.open("E:/SUBJECTS/bagofwords.csv") as f :
    jane=f.read()
with codecs.open("E:/SUBJECTS/chat.csv") as f:
    wuthering=f.read()


# In[38]:


esw=stopwords.words('english')
esw.append("would")


# In[39]:


word_pattern=re.compile("^\w+$")


# In[40]:


def get_text_counter(text):
    tokens=WordPunctTokenizer().tokenize(PorterStemmer().stem(text))
    tokens=list(map(lambda x: x.lower(), tokens))
    tokens=[token for token in tokens if re.match(word_pattern,token) and token not in esw]
    return collections.Counter(tokens), len(tokens)


# In[41]:


def make_df(counter,size):
    abs_freq=np.array([el[1] for el in counter])
    rel_freq=abs_freq/size
    index=[el[0] for el in counter]
    df=pd.DataFrame(data=np.array([abs_freq,rel_freq]).T,index=index,columns=["Absolute frequency", "Relative Frequency"])
    df.index.name="Most common words"
    return df


# In[42]:


je_counter,je_size=get_text_counter(jane)
make_df(je_counter.most_common(10),je_size)


# In[43]:


je_df=make_df(je_counter.most_common(1000),je_size)
je_df.to_csv("ih.csv")


# In[44]:


wh_counter,wh_size=get_text_counter(wuthering)
make_df(wh_counter.most_common(10),wh_size)


# In[45]:


wh_df=make_df(wh_counter.most_common(1000),wh_size)
wh_df.to_csv("il.csv")


# In[46]:


all_counter=wh_counter+je_counter
all_df=make_df(wh_counter.most_common(1000),1)
most_common_words=all_df.index.values


# In[47]:


df_data=[]
for word in most_common_words:
    je_c=je_counter.get(word,0)/je_size
    wh_c=wh_counter.get(word,0)/wh_size
    d=abs(je_c-wh_c)
    df_data.append([je_c,wh_c,d])
    
dist_df= pd.DataFrame(data=df_data, index=most_common_words,
                      columns=["bagofwords_relative_frequency","whatsappchat_relative_frequency",
                              "Relative_Frequency_difference"])
dist_df.index.name="Most common words"
dist_df.sort_values("Relative_Frequency_difference",ascending=False,inplace=True)


# In[48]:


dist_df.head(30)


# In[49]:


dist_df.to_csv("oh.csv")


# In[50]:


dist_df[dist_df!=0.000000	]


# In[51]:


dist_df=dist_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


# In[52]:


dist_df.head(25)


# In[59]:


# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib histogram
plt.hist(dist_df['Relative_Frequency_difference'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))

# seaborn histogram
sns.distplot(dist_df['Relative_Frequency_difference'], hist=True, kde=False, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram')
plt.xlabel('I')
plt.ylabel('W')


# In[ ]:




