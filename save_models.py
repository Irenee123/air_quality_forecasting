import tensorflow as tf
import os
import json
from datetime import datetime

# Create models folder
os.makedirs('models', exist_ok=True)
print('✅ Models folder created')

# Create and save a basic LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(24, 17)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Save model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model.save(f'models/best_lstm_model_{timestamp}.keras')
print(f'✅ Model saved: best_lstm_model_{timestamp}.keras')

# Create different model variants
configs = [
    {'name': 'fast_model', 'units': 32, 'layers': 1},
    {'name': 'medium_model', 'units': 64, 'layers': 2},
    {'name': 'complex_model', 'units': 128, 'layers': 3}
]

for config in configs:
    model = Sequential()
    
    if config['layers'] > 1:
        model.add(LSTM(config['units'], return_sequences=True, input_shape=(24, 17)))
        model.add(Dropout(0.3))
        
        for i in range(1, config['layers'] - 1):
            model.add(LSTM(config['units'] // (i + 1), return_sequences=True))
            model.add(Dropout(0.3))
        
        model.add(LSTM(config['units'] // config['layers']))
    else:
        model.add(LSTM(config['units'], input_shape=(24, 17)))
    
    model.add(Dropout(0.3))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model_path = f"models/{config['name']}_{timestamp}.keras"
    model.save(model_path)
    print(f'✅ Saved: {model_path}')

# Save all configs
all_configs = {
    'timestamp': timestamp,
    'models': [
        {'name': 'best_lstm_model', 'lstm_units': 64, 'layers': 2, 'dropout': 0.3},
        {'name': 'fast_model', 'lstm_units': 32, 'layers': 1, 'dropout': 0.3},
        {'name': 'medium_model', 'lstm_units': 64, 'layers': 2, 'dropout': 0.3},
        {'name': 'complex_model', 'lstm_units': 128, 'layers': 3, 'dropout': 0.3}
    ]
}

with open(f'models/all_configs_{timestamp}.json', 'w') as f:
    json.dump(all_configs, f, indent=2)

print('✅ All configs saved')
print('🎯 ALL MODELS SAVED SUCCESSFULLY!')
print(f'📁 Check the models folder: {len(configs) + 1} models saved')