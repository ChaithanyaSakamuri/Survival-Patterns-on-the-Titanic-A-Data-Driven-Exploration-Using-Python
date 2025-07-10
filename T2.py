import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    gender_submission = pd.read_csv('gender_submission.csv')
    # Combine train and test data for comprehensive analysis
    full_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    # Data Cleaning
    # 1. Handle missing values
    print("Missing values before cleaning:")
    print(full_df.isnull().sum())
    # Age - fill with median age grouped by title
    # Fixed the escape sequence by using raw string (r prefix)
    full_df['Title'] = full_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    full_df['Title'] = full_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                              'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    full_df['Title'] = full_df['Title'].replace('Mlle', 'Miss')
    full_df['Title'] = full_df['Title'].replace('Ms', 'Miss')
    full_df['Title'] = full_df['Title'].replace('Mme', 'Mrs')
    # Calculate median age by title and fill missing values
    title_median_age = full_df.groupby('Title')['Age'].median()
    for title in full_df['Title'].unique():
        full_df.loc[(full_df['Age'].isna()) & (full_df['Title'] == title), 'Age'] = title_median_age[title]
    # Fixed the inplace operations warnings
    embarked_mode = full_df['Embarked'].mode()[0]
    full_df['Embarked'] = full_df['Embarked'].fillna(embarked_mode)
    fare_median = full_df['Fare'].median()
    full_df['Fare'] = full_df['Fare'].fillna(fare_median)
    # Cabin - create new feature indicating whether cabin was known
    full_df['Has_Cabin'] = full_df['Cabin'].notna().astype(int)
    full_df = full_df.drop('Cabin', axis=1)
    # Feature engineering
    full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1
    full_df['IsAlone'] = 0
    full_df.loc[full_df['FamilySize'] == 1, 'IsAlone'] = 1
    full_df['AgeBin'] = pd.cut(full_df['Age'].astype(int), 5)
    full_df['FareBin'] = pd.qcut(full_df['Fare'], 4)
    # Convert categorical variables
    full_df = pd.get_dummies(full_df, columns=['Sex', 'Embarked', 'Title'], 
                            prefix=['Sex', 'Emb', 'Title'])
    # Drop unnecessary columns
    cols_to_drop = ['Name', 'Ticket', 'PassengerId']
    full_df = full_df.drop([col for col in cols_to_drop if col in full_df.columns], axis=1)
    print("\nMissing values after cleaning:")
    print(full_df.isnull().sum())
    # EDA
    # 1. Survival rate overview
    survival_rate = full_df['Survived'].mean()
    print(f"\nOverall survival rate: {survival_rate:.2%}")
    # 2. Survival by gender
    if 'Sex_female' in full_df.columns:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Sex_female', y='Survived', data=full_df)
        plt.title('Survival Rate by Gender')
        plt.xlabel('Female (1) vs Male (0)')
        plt.ylabel('Survival Rate')
        plt.show()
    # 3. Survival by passenger class
    if 'Pclass' in full_df.columns:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Pclass', y='Survived', data=full_df)
        plt.title('Survival Rate by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Survival Rate')
        plt.show()
    # [Include the rest of your visualization code...]
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all Titanic dataset files (train.csv, test.csv) are in your working directory.")
except Exception as e:
    print(f"An error occurred: {e}")
