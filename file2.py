from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Combine train_features and train_labels for easier preprocessing
train_data = train_features.merge(train_labels, on="respondent_id")

# Identify categorical and numerical columns
categorical_columns = [
    'age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 
    'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 
    'employment_industry', 'employment_occupation'
]

numerical_columns = [
    'xyz_concern', 'xyz_knowledge', 'behavioral_antiviral_meds', 
    'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands', 
    'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 
    'doctor_recc_xyz', 'doctor_recc_seasonal', 'chronic_med_condition', 
    'child_under_6_months', 'health_worker', 'health_insurance', 
    'opinion_xyz_vacc_effective', 'opinion_xyz_risk', 'opinion_xyz_sick_from_vacc', 
    'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 
    'household_adults', 'household_children'
]

# Preprocessing pipeline for numerical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numerical and categorical pipelines into a single ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

# Split the data into training and validation sets
X = train_data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y = train_data[['xyz_vaccine', 'seasonal_vaccine']]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))

# Create a pipeline that combines preprocessing and model training
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_val_pred = pipeline.predict_proba(X_val)
y_val_pred = np.column_stack([y_val_pred[0][:,1], y_val_pred[1][:,1]])  # Extract probabilities for the positive class

roc_auc = roc_auc_score(y_val, y_val_pred, average="macro")

roc_auc
