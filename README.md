# QPCA prompt: generate the PCA with standard scaler with 10 components and explained variance and recontrsucted data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

crop = pd.read_csv("/content/ERP.csv")
#crop=crop.drop(columns=['date', 'wnd_dir'])

# prompt: fill null values with mean

# Fill null values with the mean of each column
for col in crop.columns:
    if crop[col].isnull().any():
        mean_val = crop[col].mean()
        crop[col].fillna(mean_val, inplace=True)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming 'crop' DataFrame is already loaded and preprocessed as in the provided code.

# Separate features (X) and target variable (y) if needed.
# If you have a target variable, replace 'crop' with your features
x = crop

# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA with 10 components
pca = PCA(n_components=71)
x_pca = pca.fit_transform(x_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print(f"Total Explained Variance: {np.sum(explained_variance)}")

# Reconstruct the data
x_reconstructed = pca.inverse_transform(x_pca)
x_reconstructed = scaler.inverse_transform(x_reconstructed) #Reverse the scaling

#Convert back to DataFrame
reconstructed_df = pd.DataFrame(x_reconstructed, columns = x.columns)
print("\nReconstructed Data:")
print(reconstructed_df.head())

# Calculate the reconstruction error
reconstruction_error = np.mean(np.sum((crop - reconstructed_df)**2, axis=1))

print(f"Reconstruction Error: {reconstruction_error}")
# prompt: create a data frame for reconstructed data
crop1 = pd.DataFrame(x_reconstructed, columns = x.columns)

# prompt: generate the PCA code for crop 1 dataset same as above

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x = crop1

# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA with 10 components
pca = PCA(n_components=4)
x_pca = pca.fit_transform(x_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print(f"Total Explained Variance: {np.sum(explained_variance)}")

# Reconstruct the data
x_reconstructed = pca.inverse_transform(x_pca)
x_reconstructed = scaler.inverse_transform(x_reconstructed) #Reverse the scaling

# Convert back to DataFrame
reconstructed_df = pd.DataFrame(x_reconstructed, columns=x.columns)
print("\nReconstructed Data:")
print(reconstructed_df.head())

crop2 = pd.DataFrame(x_reconstructed, columns = x.columns)

# prompt: generate the reconstruction error for the above

import numpy as np
# Calculate the reconstruction error
reconstruction_error = np.linalg.norm(crop1 - crop2)
print(f"\nReconstruction Error: {reconstruction_error}")

# Calculate the reconstruction error for each sample
sample_errors = np.linalg.norm(crop1 - crop2, axis=1)
print(f"\nSample Reconstruction Errors: {sample_errors}")

# prompt: genrate the PCA code to the crop2 dataset same as aboe

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming 'crop2' DataFrame is already loaded and preprocessed.

# Separate features (X)
x = crop2

# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)  # You can adjust the number of components
x_pca = pca.fit_transform(x_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print(f"Total Explained Variance: {np.sum(explained_variance)}")

# Reconstruct the data
x_reconstructed = pca.inverse_transform(x_pca)
x_reconstructed = scaler.inverse_transform(x_reconstructed)

# Convert back to DataFrame
reconstructed_df = pd.DataFrame(x_reconstructed, columns=x.columns)
print("\nReconstructed Data:")
print(reconstructed_df.head())

crop3 = pd.DataFrame(x_reconstructed, columns = x.columns)

# prompt: generate the PCA code for crop3 dataset same as above

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming 'crop3' DataFrame is already loaded and preprocessed as in the previous steps.

# Separate features (X)
x = crop3

# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA
pca = PCA(n_components=1)  # You can adjust the number of components as needed
x_pca = pca.fit_transform(x_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print(f"Total Explained Variance: {np.sum(explained_variance)}")

# Reconstruct the data
x_reconstructed = pca.inverse_transform(x_pca)
x_reconstructed = scaler.inverse_transform(x_reconstructed)

# Convert back to DataFrame
reconstructed_df = pd.DataFrame(x_reconstructed, columns=x.columns)
print("\nReconstructed Data:")

# prompt: generate the code for reconstruction error in the above

# Calculate the reconstruction error
reconstruction_error = np.mean(np.sum((crop - reconstructed_df)**2, axis=1))

print(f"Reconstruction Error: {reconstruction_error}")

crop4 = reconstructed_df
crop4.head()

#Load pollution dataset
data = crop4.copy()
features = ['Fp1'] # Add other relevant features if needed
data = data[features]


# Fill Missing Values
data.fillna(method='ffill', inplace=True)

# Original Time Series Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Fp1'], label='Fp1 Signal', color='blue', linewidth=2)
plt.xlabel('Time', fontsize=12, fontweight='bold')
plt.ylabel('Fp1 Signal', fontsize=12, fontweight='bold')
plt.title('Original Fp1 Time Series', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# STL Decomposition
stl = STL(data['Fp1'], seasonal=13, period=24)
result = stl.fit()
result.plot()
plt.show()

# Metrics for STL decomposition
stl_trend_mse = mean_squared_error(data['Fp1'], result.trend)
stl_trend_rmse = np.sqrt(stl_trend_mse)
stl_trend_mae = mean_absolute_error(data['Fp1'], result.trend)
print(f"STL Decomposition - MSE: {stl_trend_mse:.4f}, RMSE: {stl_trend_rmse:.4f}, MAE: {stl_trend_mae:.4f}")

# ACF & PACF Plots
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(data['Fp1'], ax=ax[0], lags=50)
plot_pacf(data['Fp1'], ax=ax[1], lags=50)
ax[0].set_title("Autocorrelation Function (ACF)", fontweight='bold')
ax[1].set_title("Partial Autocorrelation Function (PACF)", fontweight='bold')
plt.show()

# Holt-Winters vs STL Trend
hw_model = ExponentialSmoothing(data['Fp1'], seasonal='add', seasonal_periods=12).fit()
data['HW_Forecast'] = hw_model.fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Fp1'], label='Actual Fp1 Signal', color='blue')
plt.plot(data.index, data['HW_Forecast'], label='Holt-Winters Forecast', color='red')
plt.plot(data.index, result.trend, label='STL Trend', color='green')
plt.xlabel('Time', fontweight='bold')
plt.ylabel('Fp1 Signal', fontweight='bold')
plt.title('Holt-Winters Forecast vs STL Trend', fontweight='bold')
plt.legend(prop={'weight': 'bold'})
plt.show()

# Metrics for Holt-Winters
tw_mse = mean_squared_error(data['Fp1'], data['HW_Forecast'])
tw_rmse = np.sqrt(tw_mse)
tw_mae = mean_absolute_error(data['Fp1'], data['HW_Forecast'])
print(f"Holt-Winters - MSE: {tw_mse:.4f}, RMSE: {tw_rmse:.4f}, MAE: {tw_mae:.4f}")

# Normalize dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

# Prepare time-series sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length]['Fp1'])
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(data_scaled, seq_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build Hybrid CLT-Net Model
def build_hybrid_model(seq_length, features):
    inputs = Input(shape=(seq_length, features))
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + x)
    x = Dense(32, activation='relu')(attn_output[:, -1, :])
    x = Dropout(0.2)(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = build_hybrid_model(seq_length, len(data.columns))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

y_pred = model.predict(X_test)
y_test_rescaled = scaler.inverse_transform(np.column_stack((y_test, np.zeros((len(y_test), len(data.columns)-1)))))[:, 0]
y_pred_rescaled = scaler.inverse_transform(np.column_stack((y_pred.flatten(), np.zeros((len(y_pred), len(data.columns)-1)))))[:, 0]

# Evaluate Model
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Results Comparison Table
results_df = pd.DataFrame({
    'Method': ['STL', 'Holt-Winters', 'Hybrid CLT-Net'],
    'MSE': [stl_trend_mse, tw_mse, mse],
    'RMSE': [stl_trend_rmse, tw_rmse, rmse],
    'MAE': [stl_trend_mae, tw_mae, mae],
    'R²': ['-', '-', r2]
})
print(results_df)

import numpy as np
import matplotlib.pyplot as plt

# Explained variance values (converted to percentages)
explained_variance_n = np.array([
    6.69053287e-01, 1.06461720e-01, 5.36032951e-02, 4.00152153e-02,
    1.76607072e-02, 1.55036533e-02, 1.39135916e-02, 1.13672034e-02,
    1.02281929e-02, 7.55110059e-03, 6.91355535e-03, 5.58104937e-03,
    4.79946986e-03, 3.97996940e-03, 3.49437603e-03, 3.24448531e-03,
    2.51260959e-03, 2.40867239e-03, 2.04092149e-03, 1.89451916e-03,
    1.73665780e-03, 1.55948338e-03, 1.35547362e-03, 1.17119598e-03,
    9.81810391e-04, 8.88719012e-04, 7.18416264e-04, 6.74144627e-04,
    6.71665889e-04, 5.65354318e-04, 5.37931488e-04, 5.11846712e-04,
    5.07855842e-04, 4.28820027e-04, 4.02907638e-04, 3.47524741e-04,
    3.20233970e-04, 3.15736085e-04, 2.86539464e-04, 2.67489199e-04,
    2.47171879e-04, 2.39457073e-04, 2.20647604e-04, 2.06012931e-04,
    1.81678562e-04, 1.73642327e-04, 1.67127360e-04, 1.59875886e-04,
    1.43743687e-04, 1.38209992e-04, 1.33031488e-04, 1.30984972e-04,
    1.24135332e-04, 1.17381785e-04, 1.11608784e-04, 1.08070539e-04,
    9.41880496e-05, 8.96195934e-05, 8.80729431e-05, 8.34017072e-05,
    7.32320072e-05, 7.00058206e-05, 6.83787743e-05, 6.64950752e-05,
    6.09226055e-05, 5.50734864e-05, 4.32466830e-05, 4.00806342e-05,
    3.26545320e-05, 3.08448385e-05, 2.36025963e-05
]) * 100  # Convert to percentage

explained_variance_4 = np.array([0.66905329, 0.10646172, 0.0536033, 0.04001522]) * 100
explained_variance_2 = np.array([0.74939681, 0.12153717]) * 100
explained_variance_1 = np.array([0.85210666]) * 100

# Colors for visualization
colors_n = plt.cm.tab20(np.linspace(0, 1, len(explained_variance_n)))  # Unique colors for 71 components
colors_4 = ['gold', 'blue', 'green', 'red']  # Highlighting first component
colors_2 = ['gold', 'cyan']  # Highlighting first component
colors_1 = ['gold']  # Single color highlight for 1 component

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the full 71-component bar (stacked) with first component highlighted
bottom = 0
for i in range(len(explained_variance_n)):
    color = 'gold' if i == 0 else colors_n[i]  # Highlight the first component
    ax.bar("71 Components", explained_variance_n[i], bottom=bottom, color=color, alpha=0.8)
    bottom += explained_variance_n[i]

# Plot the 4-component bar (stacked) with first component highlighted
bottom = 0
for i in range(4):
    color = 'gold' if i == 0 else colors_4[i]  # Highlight the first component
    ax.bar("4 Components", explained_variance_4[i], bottom=bottom, color=color, alpha=0.8)
    bottom += explained_variance_4[i]

# Plot the 2-component bar (stacked) with first component highlighted
bottom = 0
for i in range(2):
    color = 'gold' if i == 0 else colors_2[i]  # Highlight the first component
    ax.bar("2 Components", explained_variance_2[i], bottom=bottom, color=color, alpha=0.8)
    bottom += explained_variance_2[i]

# Plot the 1-component bar with highlighting
ax.bar("1 Component", explained_variance_1[0], color=colors_1[0], alpha=0.8)

# Labels and title with bold font
ax.set_ylabel("Explained Variance (%)", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of PCA Components", fontsize=14, fontweight='bold')
ax.set_title("Explained Variance by PCA Components", fontsize=16, fontweight='bold')

# Set bold tick labels
ax.tick_params(axis='both', labelsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Set bold gridlines
ax.yaxis.grid(True, linestyle="--", alpha=0.5)

# Display plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Explained variance values (converted to percentages)
explained_variance_n = np.array([
    0.38954825, 0.17534459, 0.14231179, 0.13986608, 0.10564038, 0.03008845, 0.01720047
]) * 100  # Convert to percentage

explained_variance_4 = np.array([0.38954825, 0.17534459, 0.14231179, 0.13986608]) * 100
explained_variance_2 = np.array([0.4540443, 0.26044586]) * 100
explained_variance_1 = np.array([0.68474541]) * 100

# Colors for visualization
colors_n = plt.cm.tab10(np.linspace(0, 1, len(explained_variance_n)))  # Unique colors for 7 components
colors_4 = ['gold', 'blue', 'green', 'red']  # Highlighting first component
colors_2 = ['gold', 'cyan']  # Highlighting first component
colors_1 = ['gold']  # Single color highlight for 1 component

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the full n-component bar (stacked) with first component highlighted
bottom = 0
for i in range(len(explained_variance_n)):
    color = 'gold' if i == 0 else colors_n[i]  # Highlight the first component
    ax.bar("7 Components", explained_variance_n[i], bottom=bottom, color=color, alpha=0.8)
    bottom += explained_variance_n[i]

# Plot the 4-component bar (stacked) with first component highlighted
bottom = 0
for i in range(4):
    color = 'gold' if i == 0 else colors_4[i]  # Highlight the first component
    ax.bar("4 Components", explained_variance_4[i], bottom=bottom, color=color, alpha=0.8)
    bottom += explained_variance_4[i]

# Plot the 2-component bar (stacked) with first component highlighted
bottom = 0
for i in range(2):
    color = 'gold' if i == 0 else colors_2[i]  # Highlight the first component
    ax.bar("2 Components", explained_variance_2[i], bottom=bottom, color=color, alpha=0.8)
    bottom += explained_variance_2[i]

# Plot the 1-component bar with highlighting
ax.bar("1 Component", explained_variance_1[0], color=colors_1[0], alpha=0.8)

# Labels and title with bold font
ax.set_ylabel("Explained Variance (%)", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of PCA Components", fontsize=14, fontweight='bold')
ax.set_title("Explained Variance by PCA Components", fontsize=16, fontweight='bold')

# Set bold tick labels
ax.tick_params(axis='both', labelsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Set bold gridlines
ax.yaxis.grid(True, linestyle="--", alpha=0.5)

# Display plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Training & Validation Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# 2. True vs Predicted Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual', color='blue', linewidth=2)
plt.plot(y_pred_rescaled, label='Predicted', color='red', linewidth=2)
plt.xlabel('Time Step', fontsize=12, fontweight='bold')
plt.ylabel('Fp1 Signal', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Signal', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# 3. Residual Plot
residuals = y_test_rescaled - y_pred_rescaled
plt.figure(figsize=(10, 5))
plt.plot(residuals, color='green', linewidth=1.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Time Step', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# 4. Distribution of Prediction Errors
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='purple')
plt.xlabel('Residual Error', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# 5. Forecast Overlay on Test Set
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled, label='Actual', linewidth=2, color='blue')
plt.plot(range(len(y_pred_rescaled)), y_pred_rescaled, label='Forecast', linewidth=2, color='orange')
plt.fill_between(range(len(y_pred_rescaled)), y_test_rescaled, y_pred_rescaled, color='gray', alpha=0.3)
plt.xlabel('Time Index', fontsize=12, fontweight='bold')
plt.ylabel('Fp1 Signal', fontsize=12, fontweight='bold')
plt.title('Hybrid CLT Net-Forecast vs Actual on Test Data', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# 6. (Optional) Correlation Heatmap of Original Features
plt.figure(figsize=(10, 8))
sns.heatmap(crop.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

Pollution Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pollution dataset
data = pd.read_csv("/content/LSTM-Multivariate_pollution.csv", parse_dates=['date'], index_col='date')
features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
data = data[features]
# Fill Missing Values
data.fillna(method='ffill', inplace=True)

# Original Time Series Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['pollution'], label='Air Pollution', color='blue', linewidth=2)
plt.xlabel('Time', fontsize=12, fontweight='bold')
plt.ylabel('Pollution Level', fontsize=12, fontweight='bold')
plt.title('Original Pollution Time Series', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# STL Decomposition
# Assuming your data has an hourly frequency, you can set the period explicitly
stl = STL(data['pollution'], seasonal=13, period=24)  # Assuming 24-hour period # Changed seasonal to 13 (odd number)
result = stl.fit()
result.plot()
plt.show()

# Metrics for STL decomposition
stl_trend_mse = mean_squared_error(data['pollution'], result.trend)
stl_trend_rmse = np.sqrt(stl_trend_mse)
stl_trend_mae = mean_absolute_error(data['pollution'], result.trend)
print(f"STL Decomposition - MSE: {stl_trend_mse:.4f}, RMSE: {stl_trend_rmse:.4f}, MAE: {stl_trend_mae:.4f}")

# ACF & PACF Plots
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(data['pollution'], ax=ax[0], lags=50)
plot_pacf(data['pollution'], ax=ax[1], lags=50)

ax[0].set_title("Autocorrelation Function (ACF)", fontweight='bold')
ax[1].set_title("Partial Autocorrelation Function (PACF)", fontweight='bold')
ax[0].tick_params(axis='both', labelsize=10, width=2)
ax[1].tick_params(axis='both', labelsize=10, width=2)
plt.show()

# Holt-Winters vs STL Trend
hw_model = ExponentialSmoothing(data['pollution'], seasonal='add', seasonal_periods=12).fit()
data['HW_Forecast'] = hw_model.fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['pollution'], label='Actual Pollution', color='blue')
plt.plot(data.index, data['HW_Forecast'], label='Holt-Winters Forecast', color='red')
plt.plot(data.index, result.trend, label='STL Trend', color='green')

plt.xlabel('Time', fontweight='bold')
plt.ylabel('Pollution Level', fontweight='bold')
plt.title('Holt-Winters Forecast vs STL Trend', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(prop={'weight': 'bold'})

plt.show()

# Metrics for Holt-Winters
tw_mse = mean_squared_error(data['pollution'], data['HW_Forecast'])
tw_rmse = np.sqrt(tw_mse)
tw_mae = mean_absolute_error(data['pollution'], data['HW_Forecast'])
print(f"Holt-Winters - MSE: {tw_mse:.4f}, RMSE: {tw_rmse:.4f}, MAE: {tw_mae:.4f}")

# Normalize dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

# Prepare time-series sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length]['pollution'])
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(data_scaled, seq_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build Hybrid CLT-Net Model
def build_hybrid_model(seq_length, features):
    inputs = Input(shape=(seq_length, features))
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + x)
    x = Dense(32, activation='relu')(attn_output[:, -1, :])
    x = Dropout(0.2)(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = build_hybrid_model(seq_length, len(data.columns))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

y_pred = model.predict(X_test)
y_test_rescaled = scaler.inverse_transform(np.column_stack((y_test, np.zeros((len(y_test), len(data.columns)-1)))))[:, 0]
y_pred_rescaled = scaler.inverse_transform(np.column_stack((y_pred.flatten(), np.zeros((len(y_pred), len(data.columns)-1)))))[:, 0]

# Evaluate Model
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# Error Metrics
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
smape = 100/len(y_test_rescaled) * np.sum(2 * np.abs(y_pred_rescaled - y_test_rescaled) / (np.abs(y_test_rescaled) + np.abs(y_pred_rescaled)))
print(f"MAPE: {mape:.4f}, SMAPE: {smape:.4f}")

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# Error Metrics
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
smape = 100/len(y_test_rescaled) * np.sum(2 * np.abs(y_pred_rescaled - y_test_rescaled) / (np.abs(y_test_rescaled) + np.abs(y_pred_rescaled)))
print(f"MAPE: {mape:.4f}, SMAPE: {smape:.4f}")

# prompt: generate the code for test prediction graph for the above model with bold font, scale, lables etc

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual Pollution', color='blue', linewidth=2)
plt.plot(y_pred_rescaled, label='Predicted Pollution', color='red', linewidth=2)
plt.xlabel('Time', fontsize=14, fontweight='bold')
plt.ylabel('Pollution Level', fontsize=14, fontweight='bold')
plt.title('Test Prediction Graph - Hybrid CLT-Net Model', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Enhanced Plot: Actual vs Predicted Pollution Levels (First 100 Samples)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100], label='Actual Pollution', marker='o', color='blue', linewidth=2, markersize=6)
plt.plot(y_pred_rescaled[:100], label='Predicted Pollution', marker='x', color='red', linewidth=2, markersize=6)
plt.xlabel("Time Steps", fontsize=14, fontweight='bold')
plt.ylabel("Pollution Level (µg/m³)", fontsize=14, fontweight='bold')
plt.title("Hybrid CLT-Net: Actual vs Predicted Pollution Levels", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, label='Actual Pollution', color='blue', linewidth=2)
plt.plot(y_pred_rescaled, label='Predicted Pollution', color='red', linewidth=2)
plt.xlabel('Time Steps', fontsize=14, fontweight='bold')
plt.ylabel('Pollution Level (µg/m³)', fontsize=14, fontweight='bold')
plt.title('Hybrid CLT-Net: Full Test Set Prediction', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled[:100], label='Actual', color='blue', marker='o', markersize=4, linewidth=2)
plt.plot(y_pred_rescaled[:100], label='Predicted', color='red', marker='x', markersize=4, linewidth=2)
plt.xlabel('Time Steps (Zoomed)', fontsize=14, fontweight='bold')
plt.ylabel('Pollution Level (µg/m³)', fontsize=14, fontweight='bold')
plt.title('Hybrid CLT-Net: Actual vs Predicted (First 100 Samples)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

import seaborn as sns

errors = y_test_rescaled - y_pred_rescaled

plt.figure(figsize=(12, 6))
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.xlabel('Prediction Error', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
plt.grid(True)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, label='Actual', color='blue', linewidth=2)
plt.plot(y_pred_rescaled, label='Predicted', color='orange', linewidth=2)
plt.fill_between(range(len(y_test_rescaled)), y_test_rescaled, y_pred_rescaled, color='gray', alpha=0.3)
plt.xlabel('Time Steps', fontsize=14, fontweight='bold')
plt.ylabel('Pollution Level', fontsize=14, fontweight='bold')
plt.title('Actual vs Predicted with Deviation Shading', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(errors, color='green', linewidth=1.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Time Step', fontsize=14, fontweight='bold')
plt.ylabel('Residual (Actual - Predicted)', fontsize=14, fontweight='bold')
plt.title('Residual Plot - Hybrid CLT-Net', fontsize=16, fontweight='bold')
plt.grid(True)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


