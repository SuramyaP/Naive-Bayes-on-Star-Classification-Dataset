#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Star_Dataset.csv')


# In[3]:


df


# In[4]:


unique_values = df['Star color'].unique()

print(unique_values)


# In[5]:


value_mapping = {'Blue White': 'Blue White', 'Whitish': 'White', 'Blue white':'Blue White','Blue-white': 'Blue White','Whitish':'White','Yellowish White':'Yellow White', 'yellow-white': 'Yellow White','White-Yellow':'Yellow White','white':'White', 'Blue ':'Blue', 'yellowish':'Yellow', 'Yellowish':'Yellow','Blue white ':'Blue White', 'Blue-White':'Blue White','Orange-Red':'Orange Red'  }
df['Star color'].replace(value_mapping, inplace=True)


# In[6]:


x = df['Star type'].value_counts()


# In[7]:


x


# In[97]:


plt.figure(figsize=(8,4))  
sns.countplot(x="Star type", data=df)
plt.title('CountPlot of Star Dataset')
plt.xticks(ticks = [0,1,2,3,4,5], labels = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence' , 'SuperGiants', 'HyperGiants'])
plt.savefig('1.png')
plt.show()


# In[9]:


star_type_by_color = df.groupby('Star color')['Star type'].mean()


# In[10]:


star_type_by_color


# In[11]:


df.groupby(['Star color','Star type']).size()


# In[12]:


table = pd.pivot_table(df,values ='Star type',index =['Star color'], columns = ['Spectral Class'] )


# In[13]:


table


# In[99]:


table.plot.barh()
plt.savefig('2.png')
plt.title('Star Type Rate By Star Color And Spectral Class')
plt.show()


# In[100]:


star_type_by_specclass = df.groupby('Spectral Class')['Star type'].mean()


# In[101]:


star_type_by_specclass


# In[102]:


df.groupby(['Spectral Class','Star type']).size()


# In[103]:


df['Temperature (K)'].max() 


# In[104]:


df['Temperature (K)'].min()


# In[105]:


df['Temperature Range'] = pd.cut(df['Temperature (K)'], bins=[0, 8000, 16000, 24000 , 32000, 40000], labels=['0 - 8000', '8001-16000','16001-24000', '24001-32000', '32001-40000'])


# In[106]:


df


# In[107]:


newtable = pd.pivot_table(df,values ='Star type',index =['Star color', 'Temperature Range'], columns = ['Spectral Class'] )


# In[108]:


newtable


# In[109]:


plt.scatter(df['Temperature (K)'], df['Star color'])
plt.title('Visualization of Temperature vs Star Color')
plt.savefig('3.png')


# In[110]:


plt.scatter(df['Temperature (K)'], df['Spectral Class'])
plt.title('Visualization of Temperature vs Spectral Class')
plt.savefig('4.png')


# In[111]:


plt.scatter(df['Luminosity(L/Lo)'], df['Star color'])
plt.title('Visualization of Liminosity vs Star Color')
plt.savefig('5.png')


# In[ ]:





# In[112]:


nf = df.copy()


# In[113]:


nf


# In[114]:


y = nf["Star type"]
nf.drop(columns = ['Star type','Temperature Range'], inplace = True, axis = 1)


# In[115]:


nf.head(5)


# In[116]:


y


# In[117]:


# xyz = pd.get_dummies(nf,columns = ['Star color','Spectral Class'])


# In[118]:


# Initialize the LabelEncoder
xyz = nf.copy()
label_encoder = LabelEncoder()

# Fit and transform the categorical column
xyz['Star_color_en'] = label_encoder.fit_transform(xyz['Star color'])
xyz['Spectral_Class_en'] = label_encoder.fit_transform(xyz['Spectral Class'])

xyz.drop(labels = ['Star color', 'Spectral Class'],axis = 1, inplace= True)
xyz


# In[119]:


xyz.head(5)


# In[120]:


xyz.describe()


# In[121]:


correlation_matrix = xyz.corr(method='pearson')


# In[122]:


correlation_matrix


# In[123]:


plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Star Dataset')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)
plt.savefig('6.png')
# Show the plot
plt.show()


# In[124]:


sns.histplot(xyz['Temperature (K)'],kde=True)
plt.title("Histogram plot of Temperature")
plt.savefig('7.png')


# In[125]:


sns.histplot(xyz['Luminosity(L/Lo)'],kde=True)
plt.title("Histogram plot of Luminosity")
plt.savefig('8.png')


# In[126]:


sns.histplot(xyz['Radius(R/Ro)'],kde=True)
plt.title("Histogram plot of Radius")
plt.savefig('9.png')


# In[127]:


sns.histplot(xyz['Absolute magnitude(Mv)'],kde=True)
plt.title("Histogram plot of Magnitude")
plt.savefig('10.png')


# In[128]:


X_shuffled, y_shuffled = shuffle(xyz, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size = 0.2, random_state = 0)


# In[129]:


#X_train, X_test, Y_train, Y_test = train_test_split(xyz, y, test_size = 0.2, random_state= 100)


# In[130]:


X_train


# In[131]:


classifier = GaussianNB()


# In[132]:


model = classifier.fit(X_train, y_train)


# In[133]:


prediction = model.predict(X_test )


# In[134]:


prediction


# In[135]:


y_test


# In[136]:


probabilities = classifier.predict_proba(X_test)


# In[137]:


# probabilities


# In[138]:


cm = confusion_matrix( prediction, y_test)


# In[139]:


cm


# In[140]:


sns.heatmap(cm, annot=True, fmt='', cmap='Blues',xticklabels = ['0','1','2','3','4','5'],yticklabels = ['0','1','2','3','4','5'] )

# Add labels, title, and adjust layout
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix-Gaussian Naive Bayes')
plt.tight_layout()
plt.savefig('11.png')

# Display the heatmap
plt.show()


# In[141]:


report = classification_report(y_test, prediction)


# In[142]:


print(report)


# In[143]:


print("Naive Bayes score: ",model.score(X_test, y_test))


# In[144]:


X_shuffled.head(5)


# Equal Width Binning (Catagorical)

# In[145]:


#Equal width Binning

# Define continuous features to discretize
continuous_features = [ 'Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
X_shuffled2=X_shuffled.copy()
# Number of desired bins
num_bins = 10

# Apply equal width binning to each continuous feature
for feature in continuous_features:
    # bin_width = (X_shuffled2[feature].max() - X_shuffled2[feature].min()) / num_bins
    # bins = np.arange(X_shuffled2[feature].min(), X_shuffled2[feature].max() + bin_width, bin_width)
    # X_shuffled2[feature + '_bins'] = pd.cut(X_shuffled2[feature], bins=bins, labels=False)
    
    X_shuffled2[feature + '_bins'], bins = pd.cut(X_shuffled2[feature], bins=num_bins, labels=False, retbins= True)
    # print(f"Bins for {feature} feature \nBin width: {bin_width}\nBins are:")
    # print(bins)
    # print("")
    print("Bin Width: ", bins[1] - bins[0])
    print(bins)
    print("")

# Drop the original continuous features from the dataset
X_shuffled2.drop(continuous_features, axis=1, inplace=True)


# In[146]:


X_shuffled2


# In[147]:


X_shuffled2.describe()


# In[148]:


X_shuffled2.isnull().any()


# In[149]:


y_shuffled.describe()


# In[150]:


X_trainn, X_testn, y_trainn, y_testn = train_test_split(X_shuffled2, y_shuffled, test_size = 0.2, random_state = 0)

model1 = CategoricalNB()
model1.fit(X_trainn, y_trainn)


# In[151]:


prediction1 = model1.predict(X_testn)


# In[152]:


prediction1


# In[153]:


y_testn


# In[154]:


cm = confusion_matrix(prediction1, y_testn)
cm


# In[156]:


sns.heatmap(cm, annot=True, fmt='', cmap='Blues',xticklabels = ['0','1','2','3','4','5'],yticklabels = ['0','1','2','3','4','5'] )

# Add labels, title, and adjust layout
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Equal Width Binning')
plt.tight_layout()
plt.savefig('13.png')

# Display the heatmap
plt.show()


# In[71]:


print("Categorical Naive Bayes score(Equal Width Binning): ",model1.score(X_testn, y_testn))


# In[72]:


report1 = classification_report(y_testn, prediction1, digits=4)
print(report1)


# Equal Frequency Binning (Catagorical)

# In[157]:


#####Equal Frequency Binning

X_shuffled3=X_shuffled.copy()
# Define continuous features to discretize
continuous_features = [ 'Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']

# Number of desired bins
num_bins = 10

# Create a new DataFrame to store the binned features
data_binned = X_shuffled3.copy()

# Apply equal frequency binning to each continuous feature
for feature in continuous_features:
    # Calculate the bin edges based on quantiles
    data_binned[feature], bins = pd.qcut(X_shuffled3[feature], q=num_bins, labels=False, retbins=True, duplicates='drop')
    print("Bin Width: ", bins[1] - bins[0])
    print(bins)
    print("")



# In[158]:


data_binned.isna().any()


# In[159]:


data_binned


# In[160]:


X_shuffled3=data_binned.copy()


# In[161]:


nX_train, nX_test, ny_train, ny_test = train_test_split(X_shuffled3, y_shuffled, test_size = 0.2, random_state = 0)

model2 = CategoricalNB()
model2.fit(nX_train, ny_train)


# In[162]:


prediction2 = model2.predict(nX_test)


# In[163]:


prediction2


# In[164]:


ny_test


# In[165]:


cm = confusion_matrix(prediction2, ny_test)
cm


# In[166]:


sns.heatmap(cm, annot=True, fmt='', cmap='Blues',xticklabels = ['0','1','2','3','4','5'],yticklabels = ['0','1','2','3','4','5'] )

# Add labels, title, and adjust layout
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Equal Frequency Binning')
plt.tight_layout()
plt.savefig('15.png')

# Display the heatmap
plt.show()


# In[167]:


print("Categorical Naive Bayes score(Equal Frequency Binning): ",model2.score(nX_test, ny_test))


# In[168]:


report2 = classification_report(ny_test, prediction2, digits=4)
print(report2)


# HYBRID NAIVE BAYES

# In[169]:


X_shuffled.head(5)


# In[170]:


X = X_shuffled.copy()


# In[171]:


X


# In[172]:


y = y_shuffled.copy()
y


# In[173]:


X.columns


# In[174]:


cata_features = ['Star_color_en', 'Spectral_Class_en']
cont_features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)','Absolute magnitude(Mv)']


# In[175]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Gaussian Naive Bayes for continuous features
gnb = GaussianNB()
gnb.fit(X_train[cont_features], y_train)

# Train Categorical Naive Bayes for categorical features
cnb = CategoricalNB()
cnb.fit(X_train[cata_features], y_train)

# Predict probabilities for each class
gnb_probs = gnb.predict_proba(X_test[cont_features])
cnb_probs = cnb.predict_proba(X_test[cata_features])

# Combine predictions using weighted probabilities or other methods
weight_continuous = 4 / 6  # Since there are 7 continuous features
weight_categorical = 2 / 6  # Since there are 2 categorical features

hybrid_probs = weight_continuous * gnb_probs + weight_categorical * cnb_probs

# Make the final prediction by selecting the class with the highest probability
hybrid_predictions = np.argmax(hybrid_probs, axis=1)

# Evaluate the model
from sklearn.metrics import accuracy_score
# Calculate the accuracy of the Hybrid Naive Bayes model
accuracy = accuracy_score(y_test, hybrid_predictions)
print("Hybrid Naive Bayes Accuracy: {:.4f}".format(accuracy))


# In[ ]:




