#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


ab_data = pd.read_csv('ab_data.csv')


# In[3]:


ab_data.head()


# In[4]:


ab_data.group.unique(), ab_data.landing_page.unique()


# b. Use the below cell to find the number of rows in the dataset.

# In[5]:


ab_data.shape[0]


# c. The number of unique users in the dataset.

# In[6]:


ab_data.user_id.nunique()


# d. The proportion of users converted.

# In[7]:


(ab_data.converted == 1).mean()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[8]:


mismatch1 = ab_data.query('group == "treatment" and landing_page != "new_page"').shape[0]
mismatch2 = ab_data.query('group == "control" and landing_page != "old_page"').shape[0]
mismatch1 + mismatch2, type(mismatch1)


# f. Do any of the rows have missing values?

# In[9]:


ab_data.info()


# This tells us there's no rows with missing values!

# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[10]:


df2 = ab_data.copy()


# In[11]:


drop_rows1 = df2.query('group == "treatment" and landing_page != "new_page"').index
drop_rows2 = df2.query('group == "control" and landing_page != "old_page"').index
drop_rows = drop_rows1.append(drop_rows2)
df2.drop(drop_rows,inplace=True)
df2.head()


# In[12]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# In[13]:


df2[((df2['group'] == 'control') == (df2['landing_page'] == 'old_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[14]:


df2.user_id.unique().shape[0]


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[15]:


df2[df2.user_id.duplicated()]


# In[16]:


# alternate
df2[df2.user_id.duplicated(keep=False)]


# In[17]:


df2.dtypes


# c. What is the row information for the repeat **user_id**? 

# In[18]:


df2[df2.user_id == 773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[19]:


dup_user = df2[df2['user_id'] == 773192].index
dup_user


# In[20]:


df2.drop_duplicates(subset='user_id', inplace=True)

#Alternative
#df2.drop(1899, inplace=True)
#df2.drop(dup_user[0], inplace=True)


# In[21]:


df2[df2['user_id'] == 773192]


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[22]:


convert_prop = (df2.converted == 1).mean()
convert_prop


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[23]:


control_convert = df2.query('group == "control" and converted == 1').shape[0] / df2.query('group == "control"').shape[0]
control_convert


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[24]:


treatment_convert = df2.query('group == "treatment" and converted == 1').shape[0] /  df2.query('group == "treatment"').shape[0]
treatment_convert


# d. What is the probability that an individual received the new page?

# In[25]:


prop_newpage = (df2.landing_page == "new_page").mean()
prop_newpage


# In[26]:


observed_diff = control_convert - treatment_convert
observed_diff


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# Not enough evidence yet as the conversion rates appear to be higher for the control group if we're looking at the entire dataset.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **$p_{old}$**: The conversion rate, on average, for the new landing page is equal to or less than the old landing page.
# 
# **$p_{new}$** The conversion rate, on average, for the new landing page is greater than the old landing page.

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[27]:


# Since we're assuming both to be equal and have 'true' success rates
p_new = (df2.converted == 1).mean()
p_new

#p_old = df2.query('group == "control" and converted == 1').shape[0] / df2.query('group == "control"').shape[0]
#p_new = df2.query('group == "treatment" and converted == 1').shape[0] / df2.query('group == "treatment"').shape[0]
#(p_new + p_old) / 2


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# Same as $p_{new}$ as we assume both to be equal

# In[28]:


p_old = (df2.converted == 1).mean()
p_old


# c. What is $n_{new}$?

# In[29]:


n_new = df2.query('group == "treatment"').shape[0]
n_new


# d. What is $n_{old}$?

# In[30]:


n_old = df2.query('group == "control"').shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[31]:


#new_page_converted = np.random.choice([0,1], size=n_new, p=[(1-p_new),p_new])
#new_page_converted.mean()

new_page_converted = np.random.binomial(n_new,p_new)
new_page_converted / n_new


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[32]:


old_page_converted = np.random.binomial(n_old,p_old)
old_page_converted / n_old


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[33]:


new_page_converted / n_new - old_page_converted / n_old


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[34]:


p_diffs = []

for _ in range(10000):
    sample_new = np.random.binomial(n_new,p_new)
    sample_old = np.random.binomial(n_old,p_old)
    diff = sample_new/n_new - sample_old/n_old
    p_diffs.append(diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[35]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[36]:


np.mean(p_diffs)


# In[37]:


#(observed_diff < p_diffs).mean()
obs_diff = df2.query('group == "treatment"')['converted'].mean() - df2.query('group == "control"')['converted'].mean()
(obs_diff < p_diffs).mean()


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# ** We just calcualted the p-value, a p-value of ~0.91. Assuming an alpha of 0.05 (95% confidence level) this does not provide enough evidence to reject our null hypothesis; therefore, it looks like the conversion rate with our new page is less than or equal to our old page.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[44]:


import statsmodels.api as sm

convert_old = df2.query('landing_page == "old_page" and converted == 1').shape[0]
convert_new = df2.query('landing_page == "new_page" and converted == 1').shape[0]
n_old = df2.query('landing_page == "old_page"').shape[0]
n_new = df2.query('landing_page == "new_page"').shape[0]


# In[45]:


convert_old, convert_new


# In[46]:


n_old,n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[47]:


import statsmodels.api as sm


# In[48]:


z_stat, p_value = sm.stats.proportions_ztest([convert_new,convert_old], [n_new,n_old], alternative='larger')
z_stat, p_value


# In[49]:


from scipy.stats import norm
norm.cdf(z_stat)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **The z-score, which is not 3 standard deviations from the mean, and the high p-value do not provide enough evidence to reject the null hypothesis in this case. This seems to agree with the earlier findings.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic since it's categorical (yes or no conversion).**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[50]:


df2['intercept'] = 1

df2[['control','treatment']] = pd.get_dummies(df2['group'])
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[51]:


logit = sm.Logit(df2['converted'], df2[['intercept', 'treatment']])
                                   
# fit the model
result = logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[52]:


# results
print (result.summary())


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **We get the p-value of 0.190 which is still too high for a 95% confidence level (0.05), so we still do not have enough evidence to reject the null hypothesis. It's different from our previous value as our new hypothesis relates to the alternative hypothesis being not equal to the average conversion rate of the null rather than just being greater than it.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **There could be tons of other factors involved like how much time was spent on the site, where the user accessed the website from, what platform they used. At the same time, some factors may not actually be significant and may skew the data towards one end.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[53]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[54]:


df_new.head()


# In[55]:


df_new.country.unique()


# In[56]:


### Create the necessary dummy variables
df_new['intercept'] = 1

df_new[['UK', 'CA']] = pd.get_dummies(df_new['country'])[['UK', 'CA']]
df_new.head()


# In[57]:


logit = sm.Logit(df_new['converted'],df_new[['intercept','UK','CA']])
result = logit.fit()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[58]:


### Fit Your Linear Model And Obtain the Results
result.summary()


# It looks like adding the 'country' factor did not provide enough evidence to reject our null hypothesis as our p-value is still quite high in regards to a 95% confidence level.

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by beginning the next module in the program.

# In[ ]:




