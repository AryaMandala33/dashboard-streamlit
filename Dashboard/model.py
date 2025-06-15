import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv('cars.csv')

# Pilih 10 fitur utama
selected_features = [
    'odometer_value', 'year_produced', 'engine_capacity', 'has_warranty',
    'drivetrain', 'transmission', 'engine_fuel',
    'number_of_photos', 'duration_listed', 'up_counter', 'price_usd'
]
df_selected = df[selected_features].dropna()

# Pisahkan fitur dan target
X = df_selected.drop(columns=['price_usd'])
y = df_selected['price_usd']

# Encoding kategorikal
X_encoded = pd.get_dummies(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# Simpan file .pkl
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)

with open('model_dt.pkl', 'wb') as f:
    pickle.dump(model_dt, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Semua model dan scaler berhasil disimpan.")
