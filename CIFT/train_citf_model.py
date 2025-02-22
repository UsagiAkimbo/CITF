import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Paths to preprocessed data
DATA_DIR = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "citf_features.npy")
LABELS_PATH = os.path.join(DATA_DIR, "citf_labels.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "citf_ml_dataset.csv")

# Load and prepare data
def load_data():
    """Load preprocessed CITF data."""
    features = np.load(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    full_data = pd.read_csv(FULL_DATA_PATH)
    
    # Ensure consistency
    assert features.shape[0] == labels.shape[0] == full_data.shape[0]
    
    # Extract dataset tags for routing
    dataset_tags = full_data['dataset'].values
    
    return features, labels, dataset_tags

def preprocess_data(features, labels, dataset_tags):
    """Split and prepare data for multi-output model."""
    # Define output columns
    output_cols = ['citf_torsion', 'citf_velocity_anomaly', 'citf_shift', 'citf_energy', 
                   'citf_scaling', 'citf_temperature_anomaly', 'citf_conduction_factor', 'citf_anomaly']
    
    # Filter valid outputs per dataset
    output_dict = {
        'exoplanet': ['citf_torsion', 'citf_velocity_anomaly'],
        'gw_event': ['citf_shift', 'citf_energy'],
        'cmb_data': ['citf_scaling', 'citf_temperature_anomaly'],
        'uhecr': ['citf_energy', 'citf_conduction_factor'],
        'stellar_motion': ['citf_torsion', 'citf_anomaly'],
        'planetary_data': ['citf_torsion', 'citf_velocity_anomaly']
    }
    
    # Create output arrays
    y_dict = {}
    for dataset in output_dict.keys():
        mask = dataset_tags == dataset
        valid_cols = [col for col in output_cols if col in output_dict[dataset]]
        y_dict[dataset] = labels[valid_cols][mask].values
    
    # Split data
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    for dataset in output_dict.keys():
        mask = dataset_tags == dataset
        X_subset = features[mask]
        y_subset = y_dict[dataset]
        X_tr, X_te, y_tr, y_te = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        X_train[dataset] = X_tr
        X_test[dataset] = X_te
        y_train_dict[dataset] = y_tr
        y_test_dict[dataset] = y_te
    
    return X_train, X_test, y_train_dict, y_test_dict, output_dict

# Build multi-output model
def build_model(input_dim, output_dict):
    """Create a multi-output neural network for CITF predictions."""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Shared layers
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Output branches for each dataset
    outputs = {}
    for dataset, output_cols in output_dict.items():
        branch = tf.keras.layers.Dense(len(output_cols), activation='linear', name=dataset)(x)
        outputs[dataset] = branch
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with multi-output loss
    loss_dict = {dataset: 'mse' for dataset in output_dict.keys()}
    model.compile(optimizer='adam', loss=loss_dict, metrics={'exoplanet': 'mae', 'gw_event': 'mae', 
                                                            'cmb_data': 'mae', 'uhecr': 'mae', 
                                                            'stellar_motion': 'mae', 'planetary_data': 'mae'})
    
    return model

def train_model(model, X_train, X_test, y_train_dict, y_test_dict, output_dict):
    """Train the CITF model."""
    # Prepare training and validation data
    train_data = {dataset: X_train[dataset] for dataset in output_dict.keys()}
    train_labels = {dataset: y_train_dict[dataset] for dataset in output_dict.keys()}
    val_data = {dataset: X_test[dataset] for dataset in output_dict.keys()}
    val_labels = {dataset: y_test_dict[dataset] for dataset in output_dict.keys()}
    
    # Train
    history = model.fit(train_data, train_labels, epochs=50, batch_size=32, 
                        validation_data=(val_data, val_labels), verbose=1)
    
    return history

def save_model(model, history):
    """Save the trained model and history."""
    model.save("models/citf_model.h5")
    pd.DataFrame(history.history).to_csv("models/training_history.csv")
    print("Model and history saved to 'models/' directory")

def main():
    # Load data
    features, labels, dataset_tags = load_data()
    
    # Preprocess for multi-output
    X_train, X_test, y_train_dict, y_test_dict, output_dict = preprocess_data(features, labels, dataset_tags)
    
    # Determine input dimension
    input_dim = next(iter(X_train.values())).shape[1]
    
    # Build and train model
    model = build_model(input_dim, output_dict)
    history = train_model(model, X_train, X_test, y_train_dict, y_test_dict, output_dict)
    
    # Save results
    save_model(model, history)

    # Evaluate
    eval_results = model.evaluate(X_test, y_test_dict, verbose=0)
    print("Evaluation results:")
    for dataset, result in zip(output_dict.keys(), eval_results):
        print(f"{dataset} loss: {result}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
