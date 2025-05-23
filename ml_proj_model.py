#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


# Configure visualization styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("/kaggle/input/salary-data/Salary_Data.csv")
print("Data loaded successfully.")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


print(df.isnull().sum())


# In[8]:


# Drop rows with any missing values
df.dropna(inplace=True)
df.isnull().sum()


# In[9]:


print("\nValue counts for Job Title (Top 20):")
print(df['Job Title'].value_counts().head(20))
print(f"\nTotal unique Job Titles before consolidation: {df['Job Title'].nunique()}")


# In[10]:


job_title_counts = df['Job Title'].value_counts()
titles_to_group = job_title_counts[job_title_counts <= 25].index

df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in titles_to_group else x)

print(f"\nTotal unique Job Titles after consolidation: {df['Job Title'].nunique()}")
print("\nValue counts for Job Title after consolidation (Top 20):")
print(df['Job Title'].value_counts().head(20))


# In[11]:


print("\nValue counts for Education Level before consolidation:")
print(df['Education Level'].value_counts())

# Standardize education level names
df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD" 
}, inplace=True)

print("\nValue counts for Education Level after consolidation:")
print(df['Education Level'].value_counts())


# In[12]:


print("\nValue counts for Gender:")
print(df['Gender'].value_counts())


# # Exploratory Data Analysis

# In[13]:


# Distribution of Categorical Features
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(x='Gender', data=df, ax=ax[0], palette='viridis', order=df['Gender'].value_counts().index)
ax[0].set_title('Distribution of Gender')
ax[0].set_xlabel('Gender')
ax[0].set_ylabel('Count')

sns.countplot(y='Education Level', data=df, ax=ax[1], palette='magma', order=df['Education Level'].value_counts().index)
ax[1].set_title('Distribution of Education Level')
ax[1].set_xlabel('Count')
ax[1].set_ylabel('Education Level')

plt.tight_layout()
plt.show()


# In[14]:


# Top 10 Highest Paying Job Titles (using consolidated titles)
plt.figure(figsize=(12, 7))
top_paying_jobs = df.groupby('Job Title')['Salary'].mean().nlargest(10).sort_values()

sns.barplot(y=top_paying_jobs.index, x=top_paying_jobs.values, palette='crest')
plt.title('Top 10 Highest Average Salaries by Job Title')
plt.xlabel('Average Salary')
plt.ylabel('Job Title')
plt.xticks(rotation=0) # Keep x-axis labels horizontal for salary
plt.tight_layout()
plt.show()


# In[15]:


# Distribution of Continuous Variables
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Age'], ax=ax[0], color='skyblue', kde=True)
ax[0].set_title('Age Distribution')
ax[0].set_xlabel('Age')

sns.histplot(df['Years of Experience'], ax=ax[1], color='lightcoral', kde=True)
ax[1].set_title('Years of Experience Distribution')
ax[1].set_xlabel('Years of Experience')

sns.histplot(df['Salary'], ax=ax[2], color='lightgreen', kde=True)
ax[2].set_title('Salary Distribution')
ax[2].set_xlabel('Salary')

plt.tight_layout()
plt.show()


# In[16]:


# Salary vs. Categorical Features
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Gender vs Salary
sns.boxplot(x='Gender', y='Salary', data=df, ax=ax[0], palette='viridis', order=['Male', 'Female', 'Other'])
ax[0].set_title('Salary Distribution by Gender')
ax[0].set_xlabel('Gender')
ax[0].set_ylabel('Salary')

# Education Level vs Salary
# Define order for better visualization
edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
sns.boxplot(x='Education Level', y='Salary', data=df, ax=ax[1], palette='magma', order=edu_order)
ax[1].set_title('Salary Distribution by Education Level')
ax[1].set_xlabel('Education Level')
ax[1].set_ylabel('Salary')
ax[1].tick_params(axis='x', rotation=0) # Keep labels horizontal

plt.tight_layout()
plt.show()


# In[17]:


# Interaction: Education Level, Salary, and Gender
plt.figure(figsize=(12, 7))
sns.barplot(x='Education Level', y='Salary', data=df, hue='Gender', palette='muted', order=edu_order)
plt.title('Average Salary by Education Level and Gender')
plt.xlabel('Education Level')
plt.ylabel('Average Salary')
plt.legend(title='Gender')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[18]:


# Salary vs. Continuous Features (Age, Experience)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Age vs Salary
sns.scatterplot(x='Age', y='Salary', data=df, alpha=0.5, ax=ax[0], color='darkcyan')
sns.regplot(x='Age', y='Salary', data=df, scatter=False, ax=ax[0], color='red') # Add regression line
ax[0].set_title('Age vs. Salary')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Salary')

# Experience vs Salary
sns.scatterplot(x='Years of Experience', y='Salary', data=df, alpha=0.5, ax=ax[1], color='purple')
sns.regplot(x='Years of Experience', y='Salary', data=df, scatter=False, ax=ax[1], color='red') # Add regression line
ax[1].set_title('Years of Experience vs. Salary')
ax[1].set_xlabel('Years of Experience')
ax[1].set_ylabel('Salary')

plt.tight_layout()
plt.show()


# In[19]:


# Correlation Heatmap 
df_encoded = df.copy()

# Map Education Level: Use ordinal mapping as there's a clear order
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df_encoded['Education Level'] = df_encoded['Education Level'].map(education_mapping)

# Label Encode Gender
le = LabelEncoder()
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])
# Print gender mapping for clarity
print("\nGender Label Encoding Mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name} -> {i}")

# One-Hot Encode Job Title (since it's nominal)
# Calculate correlation matrix on relevant numeric columns
numeric_cols = ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary']
correlation_matrix = df_encoded[numeric_cols].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Key Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# #  Data Preparation

# In[20]:


# Map Education Level
df['Education Level'] = df['Education Level'].map(education_mapping)

# Label Encode Gender
df['Gender'] = le.transform(df['Gender'])

# One-Hot Encode 'Job Title'
df = pd.get_dummies(df, columns=['Job Title'], drop_first=True, prefix='Job')

print("\nDataFrame columns after One-Hot Encoding Job Titles:")
print(df.columns)
print("\nFirst 5 rows of the final preprocessed data:")
df.head()


# In[21]:


# Define Features (X) and Target (y)
X = df.drop('Salary', axis=1)
y = df['Salary']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")


# In[22]:


# Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")


# # Model Training and Hyperparameter Tuning

# In[23]:


# Define models and parameter grids for GridSearchCV
model_params = {
    'Linear_Regression': {
        'model': LinearRegression(),
        'params': {
             'fit_intercept': [True, False]
        }
    },
    'Decision_Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],     
            'min_samples_leaf': [1, 3, 5]
        }
    },
    'Random_Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150], 
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 3]
        }
    },
    'XGBoost': { # Added XGBoost
        'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1.0] # Fraction of samples used per tree
        }
    }
}


# In[24]:


# Perform GridSearchCV to find the best parameters for each model
scores = []
best_estimators = {}

print("\nStarting Hyperparameter Tuning (this may take a while)...")

for model_name, mp in model_params.items():
    print(f"--- Tuning {model_name} ---")
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=False)
    clf.fit(X_train, y_train)
    
    scores.append({
        'Model': model_name,
        'Best Params': clf.best_params_,
        'Best Negative MSE': clf.best_score_
    })
    best_estimators[model_name] = clf.best_estimator_ # Store the best model instance
    print(f"Best parameters for {model_name}: {clf.best_params_}")
    print(f"Best cross-validation score (Negative MSE): {clf.best_score_:.4f}\n")


# In[25]:


# Evaluate the best models found by GridSearchCV on the unseen test set
evaluation_results = []

print("\n--- Evaluating Best Models on Test Set ---")

for model_name, model in best_estimators.items():
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    evaluation_results.append({
        'Model': model_name,
        'R-squared (R2)': r2,
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse
    })
    
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")


# In[26]:


# Display evaluation results in a DataFrame
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df = evaluation_df.sort_values(by='R-squared (R2)', ascending=False)

print("\n--- Test Set Evaluation Summary (Sorted by R-squared) ---")
evaluation_df


# In[27]:


# Analyze feature importances for tree-based models (DT, RF, XGB)

models_for_importance = {
    'Decision_Tree': best_estimators.get('Decision_Tree'),
    'Random_Forest': best_estimators.get('Random_Forest'),
    'XGBoost': best_estimators.get('XGBoost')
}

num_features_to_show = 15

for model_name, model in models_for_importance.items():
    if model is None or not hasattr(model, 'feature_importances_'):
        print(f"\nSkipping feature importance for {model_name} (model not found or doesn't support it).")
        continue
        
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Create a DataFrame for better handling
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Sort features by importance and select top N
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(num_features_to_show)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top {num_features_to_show} Feature Importances - {model_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


# # Visualizing the Decision Tree

# In[28]:


from sklearn.tree import plot_tree

best_rf_model = best_estimators['Random_Forest']
n_trees_to_plot = 5
plot_max_depth = 2
fig_width = 20  
fig_height = 10      

feature_names = X_train.columns.tolist()
n_trees_to_plot = min(n_trees_to_plot, len(best_rf_model.estimators_))

print(f"\nVisualizing the first {n_trees_to_plot} trees from the Random Forest model (up to depth {plot_max_depth})...")

for i in range(n_trees_to_plot):
    tree_to_plot = best_rf_model.estimators_[i]
    plt.figure(figsize=(fig_width, fig_height))
    plot_tree(
        tree_to_plot,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        precision=2,
        fontsize=10,
        max_depth=plot_max_depth
    )
    plt.title(f'Decision Tree {i + 1} from Random Forest (Top {plot_max_depth} Levels)', fontsize=16)
    plt.show()


# # Saving the Model

# In[29]:


import joblib
import os


# In[30]:


model_to_save = best_estimators.get('Random_Forest')
label_encoder_to_save = le
education_map_to_save = education_mapping
model_columns_to_save = X_train.columns.tolist()


# In[31]:


output_dir = 'salary_model_files'
os.makedirs(output_dir, exist_ok=True)

model_filename = os.path.join(output_dir, 'random_forest_salary_model.joblib')
encoder_filename = os.path.join(output_dir, 'gender_label_encoder.joblib')
edu_map_filename = os.path.join(output_dir, 'education_mapping.joblib')
columns_filename = os.path.join(output_dir, 'model_columns.joblib')


# In[32]:


# Check if model exists before saving
if model_to_save:
    print(f"Saving model to: {model_filename}")
    joblib.dump(model_to_save, model_filename)
    
    print(f"Saving label encoder to: {encoder_filename}")
    joblib.dump(label_encoder_to_save, encoder_filename)
    
    print(f"Saving education mapping to: {edu_map_filename}")
    joblib.dump(education_map_to_save, edu_map_filename)
    
    print(f"Saving model columns to: {columns_filename}")
    joblib.dump(model_columns_to_save, columns_filename)
    
    print("\nModel and preprocessing objects saved successfully.")
else:
    print("\nError: Best Random Forest model not found. Cannot save.")


# # Loading the model for inference

# In[33]:


import joblib
import pandas as pd
import numpy as np


# In[34]:


output_dir = 'salary_model_files' 
model_filename = os.path.join(output_dir, 'random_forest_salary_model.joblib')
encoder_filename = os.path.join(output_dir, 'gender_label_encoder.joblib')
edu_map_filename = os.path.join(output_dir, 'education_mapping.joblib')
columns_filename = os.path.join(output_dir, 'model_columns.joblib')


# In[35]:


try:
    loaded_model = joblib.load(model_filename)
    loaded_encoder = joblib.load(encoder_filename)
    loaded_edu_map = joblib.load(edu_map_filename)
    loaded_model_columns = joblib.load(columns_filename)
    print("Model and preprocessing objects loaded successfully.")
except FileNotFoundError:
    print("Error: One or more model files not found. Please ensure the files exist in the specified directory.")
    # Exit or handle error appropriately
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    # Exit or handle error appropriately
    exit()


# In[36]:


# Define custom input data
custom_data = {
    'Age': [32],
    'Gender': ['Male'],
    'Education Level': ["Master's"],
    'Job Title': ['Data Scientist'],
    'Years of Experience': [8]
}

custom_df = pd.DataFrame(custom_data)
print("\nCustom Input Data:")
custom_df


# In[37]:


# Preprocess the Custom Data EXACTLY as the Training Data
processed_df = custom_df.copy()

# 1. Apply Education Level Mapping
processed_df['Education Level'] = processed_df['Education Level'].map(loaded_edu_map)

# Check if any values failed to map (resulting in NaN)
if processed_df['Education Level'].isnull().any():
     print("Warning: Unknown 'Education Level' encountered in custom data. Prediction might be inaccurate.")

# 2. Apply Gender Label Encoding
try:
    processed_df['Gender'] = loaded_encoder.transform(processed_df['Gender'])
except ValueError as e:
     print(f"Warning: Unknown 'Gender' encountered: {e}. Prediction might be inaccurate.")
     processed_df['Gender'] = 0 # Example: Assigning the code for 'Female' or 'Male' depending on encoding


# In[38]:


# 3. Apply One-Hot Encoding for Job Title and Align Columns
processed_df = pd.get_dummies(processed_df, columns=['Job Title'], prefix='Job', drop_first=False) 


# In[39]:


processed_df = processed_df.reindex(columns=loaded_model_columns, fill_value=0)


# In[40]:


print("\nProcessed Custom Data (Ready for Model):")
processed_df


# In[41]:


# Make Prediction
try:
    prediction = loaded_model.predict(processed_df)
    predicted_salary = prediction[0] 
    print(f"\nPredicted Salary: ${predicted_salary:,.2f}") 
except Exception as e:
     print(f"\nError during prediction: {e}")


# In[ ]:




