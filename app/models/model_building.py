import pandas as pd
df_od_raw = pd.read_csv('orders.csv')
df_ps_raw = pd.read_csv('product-supplier.csv')

# Copy
df_od = df_od_raw.copy()
df_ps = df_ps_raw.copy()

# Replace customer status to upper cases
df_od['Customer Status'] = df_od['Customer Status'].str.upper()
# Add Item Retail Value features
df_od['Item Retail Value'] = df_od['Total Retail Price for This Order'] / df_od['Quantity Ordered']
# Convert dates to datetime objects
df_od['date'] = pd.to_datetime(df_od['Date Order was placed'])
df_od['Delivery Date'] = pd.to_datetime(df_od['Delivery Date'])
# Segment to year, month, and day
df_od['date_year'] = pd.to_datetime(df_od.date).dt.year
df_od['date_month'] = pd.to_datetime(df_od.date).dt.month
df_od['date_day'] = pd.to_datetime(df_od.date).dt.day
df_od['date_day_of_week'] = pd.to_datetime(df_od.date).dt.dayofweek + 1

# Merge
df_merge = df_od.merge(df_ps, how='left', left_on='Product ID', right_on='Product ID')

# Filter features
print(f'Initial features: {list(df_merge.columns)}')
features2keep = ['date_month', 'date_day', 'date_day_of_week',
                 'Cost Price Per Unit','Item Retail Value', #'Product Line', 'Product Group',
                 'Product Category','Supplier Country']
print(f'Features used: {features2keep}')
# Target Variable
target = 'Quantity Ordered'
print(f'target: {target}')

X = df_merge[features2keep]
y = df_merge[target]

# Define loss function to calculate metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
def loss(data, X, y_true, model):
    y_pred = model.predict(X)
    print(data, "MAPE: {0:.2e}".format(mean_absolute_percentage_error(y_true, y_pred)))
    print(data, "MAE: {0:.2e}".format(mean_absolute_error(y_true, y_pred)))
    print(data, "MSE: {0:.2e}".format(mean_squared_error(y_true, y_pred)))
    print(data, "R2 score : {0:.3f}".format(r2_score(y_true, y_pred)))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Preprocessing and model pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


feat_num = list(X_train.select_dtypes(exclude='object').columns)
feat_cat = list(X_train.select_dtypes(include='object').columns)

# To rescale numerical and categorical data separately
colTransformer = ColumnTransformer([
    ('cat_cols', OneHotEncoder(handle_unknown='ignore'), feat_cat),
    ('num_cols', MinMaxScaler(), feat_num)
])

# Define the pipeline
steps = [
    ("col_tf", colTransformer),
    ("lr", LinearRegression())
]

model_lr = Pipeline(steps)
model_lr.fit(X_train, y_train)

# Model Performance
loss("Train", X_train, y_train, model_lr)
loss("Test", X_test, y_test, model_lr)


from xgboost.sklearn import XGBRegressor
steps = [("preprocess", colTransformer),
         ('xgbr', XGBRegressor())]
model_xgb = Pipeline(steps)
model_xgb.fit(X_train, y_train)

# Model Performance
loss("Train", X_train, y_train, model_xgb)
loss("Test", X_test, y_test, model_xgb)


# Export the model
import joblib
# save the model to disk
joblib.dump(model_xgb, "xgb_model.sav")
