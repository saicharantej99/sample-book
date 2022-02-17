#!/usr/bin/env python
# coding: utf-8

# ## Discover Food Bundles ##
# 
# A food confectionary is undergoing a tough time in sales and wants use machine learning to turn their fortunes around. They have captured their daily transaction data from 30th Oct 2016 to 9th Apr 2017 and want to understand what are all the food bundles that are popular among their customers. 
# 
# Your job is to perform <b>Market Basket Analysis</b> using the Apriori algorithm and recommend the confectionary the food bundles that they should be marketing more. 

# <b>Below is the list of questions you need to crack as part of this assignment </b>

# 1. Read the data from the csv file into a dataframe.
# 
# 2. Convert the `Datetime` object column to a datetime column and set it as index.
# 
# 3. Identify the top-10 best-selling and worst-selling items in the confectionary.
# 
# 4. Find the day and the timeslot in which most of the sales are happening in the confectionary. <b>Hint:You need to extract the hour and day details from the `Datetime` column.</b>
# 
# 5. Convert the data into a one-hot encoded format containing the count of items sold in a transaction. <b>Hint: You need to create the `Count` column to track the number of items sold in a transaction. You will then use the `pivot_table` method to create the one-hot encoded format.</b>
# 
# 6. Encode the data into 1's and 0's to apply the apriori algorithm. <b>Hint:Result obtained in the above step will contain the actual number of items sold in a transaction. In order to apply Apriori, we just need to know whether an item is present in a transaction or not. </b>
# 
# 7. Generate <b>frequent itemsets</b> and <b>association rules</b> with a minimum support threshold of 1% and a minimum lift value of 1. Sort them by <b>confidence values</b>.
# 
# 8. What <b>percentage of transactions</b> involved customers buying 'Sandwich' and 'Coffee'? <b>Hint: Select only those association rules that have lift value greater than 1 and confidence value greater than 0.5 to generate the insights.</b>
# 
# 9. What is the <b>probability</b> that a customer 'Coffee' given that he bought a 'Juice'?
# 
# 10. Which item **lifts** the Coffee's purchase by the most?
# 

# # 1. Read the data from the csv file into a dataframe.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


# load the data into a pandas dataframe and take a look at the first 10 rows
confec = pd.read_csv("confectionary.csv")
confec.head(10)


# In[ ]:


confec.info()


# ## Step 2: Convert the `Datetime` object column to a datetime column and set it as index. ##

# In[ ]:


confec['Datetime'] = pd.to_datetime(confec['Datetime'])
confec = confec[["Datetime", "Transaction", "Item"]].set_index("Datetime")
confec.head(10)


# # Step 3: Identify the top-10 best-selling and worst-selling items in the confectionary.

# In[ ]:


#List the top 10 best-selling items
confec.Item.value_counts()[:10]


# In[ ]:


#List the bottom 10 selling items
confec.Item.value_counts()[-10:-1]


# In[ ]:





# # Step 4: Find the day and the timeslot in which most of the sales are happening in the confectionary.

# In[ ]:


#Extract the hour and Weekday values from the index
confec["Hour"] = confec.index.hour
confec["Weekday"] = confec.index.weekday


# In[ ]:


#Find the peak hours by grouping the item count
confec_groupby_hour = confec.groupby("Hour").agg({"Item": lambda item: item.count()})


# In[ ]:


confec_groupby_hour


# You can see from the above result that the most sales happen between <b>11 am - 12 noon</b>. Sales actually pick up between 9 am - 4 pm and are pretty down outside these hours. Most of the sales happen during the lunch hours.

# In[ ]:


#Find the most popular weekday by grouping the item count. Weekday 0 corresponds to Monday.
confec_groupby_day = confec.groupby("Weekday").agg({"Item": lambda item: item.count()})
confec_groupby_day


# You can see from the above result that most sales happen on a <b>Saturday</b>. Weekend sales seem to be far higher than compared to weekdays. Marketing efforts are needed to boost the sales during weekdays.

# ## Step 5: Convert the data into a one-hot encoded format containing the count of items sold in a transaction.##

# The **Apriori** function in the MLxtend library expects data in a one-hot encoded pandas DataFrame. This means that all the data for a transaction must be included in one row and the items must be one-hot encoded. Example below:
# 
# |   | Coffee | Cake | Bread | Cookie | Muffin | Tea | Milk | Juice | Sandwich |
# |---|--------|------|-------|--------|--------|-----|------|-------|----------|
# | 0 | 0      | 1    | 1     | 0      | 0      |0    |0     |1      |0         |
# | 1 | 1      | 0    | 0     | 0      | 1      |0    |0     |0      |0         |
# | 2 | 0      | 0    | 0     | 1      | 0      |0    |0     |0      |1         |
# | 3 | 1      | 0    | 0     | 0      | 0      |1    |0     |0      |1         |
# | 4 | 1      | 1    | 0     | 0      | 0      |0    |0     |0      |0         |

# In[ ]:


confec.head()


# In[ ]:


#Create a 'Count' column to get the number of items sold per transaction 
confec = confec.groupby(["Transaction","Item"]).size().reset_index(name="Count")
confec.head()


# In[ ]:


#Pivot the data into a one-hot encoded format
confec_basket = pd.pivot_table(confec, index='Transaction', columns='Item',values='Count', fill_value=0)
confec_basket.shape


# In[ ]:


confec_basket.head()


# **Note:** At this stage, the one-hot encoded table shows the count of items purchased as result. We just need to know whether an item is present in a transaction or not.

# # Step 6: Encode the data into 1's and 0's to apply the apriori algorithm.

# In[ ]:


# the encoding function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# In[ ]:


confec_basket_sets = confec_basket.applymap(encode_units)

confec_basket_sets.head()


# After applying the encoding function, the cell value for all the items has become either a "1" or a "0", which is what we need for the **Apriori** function.

# ## Step 7:Generate <b>frequent itemsets</b> and <b>association rules</b> with a minimum support threshold of 1% and a minimum lift value of 1. Sort them by confidence values. ##

# In[ ]:


frequent_itemsets = apriori(confec_basket_sets, min_support=0.01, use_colnames=True)


# In[ ]:


#generate association rules and sort them by confidence in ascending order
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values("confidence", ascending = False, inplace = True)
rules.head(10)


# # Step 8: What percentage of transactions involved customers buying 'Sandwich' and 'Coffee'?
# 

# In[ ]:


#select only those rules which have lift > 1 and confidence > 0.5
rules[ (rules['lift'] >= 1) &
       (rules['confidence'] >= 0.5) ]


# From the above table, you can observe that the <b>support</b> value for the association rule involving Sandwich and Coffee is 0.038. This means out of all the transactions, <b>3.8%</b> of the transactions involved customers buying Sandwich and Coffee

# # Step 9: What is the <b>probability</b> that a customer 'Coffee' given that he bought a 'Juice'?

# From the above association rule table, you can observe that the <b>confidence</b> value for the association rule involving Juice and Coffee is 0.532. This implies that whenever a customer buys Juice, there is a <b>53.2%</b> chance that he will also buy 'Coffee'. 

# # Step 10:Which item **lifts** the Coffee's purchase by the most?

# If you observe all the association rules in the above table, you can notice that the <b>lift</b> value(1.47) is the highest for the rule having 'Toast' as the antecedent and 'Coffee' as the consequent. This implies that the purchase of <b>'Toast'</b> lifts the chances of someone buying 'Coffee' by 1.47 times.

# In[ ]:




