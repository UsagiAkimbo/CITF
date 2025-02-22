import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Database connection (matches Flask DATABASE_URL)
DATABASE_PATH = os.getenv('DATABASE_URL', 'sqlite:///Database.sqlite').replace('sqlite:///', '')

# CITF constants from archived conversation
T_N = 1e-6  # Base cosmic torsion
ALPHA = 2e-8  # Planetary coupling constant
S_LOCAL = 1  # Local scaling (Flyby)
S_GW = 5e-13  # GW detector scaling
S_UHECR = 1e-17  # Non-local UHECR scaling
S_COSMIC = 2e-7  # Cosmic scaling (CMB)

def connect_db():
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_data(table_name):
    """Fetch all rows from a specified table."""
    conn = connect_db()
    if conn is None:
        return None
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_exoplanet_data(df):
    """Preprocess Exoplanet data for CITF ML."""
    # Drop rows with missing critical data
    df = df.dropna(subset=['orbital_period', 'density'])
    
    # Features
    df['omega_p'] = 2 * np.pi / (df['orbital_period'] * 86400)  # rad/s
    df['rho'] = df['density'] * 1000  # kg/m^3
    df['mass_approx'] = 1000  # Placeholder mass (kg) for consistency
    
    # Labels (already calculated in Flask, refine here if needed)
    df['citf_torsion'] = T_N + ALPHA * df['omega_p'] * df['rho']
    df['citf_velocity_anomaly'] = ((2 * df['citf_torsion'] * df['mass_approx'] * 5e5) / df['mass_approx']) ** 0.5
    
    # Select features and labels
    features = df[['omega_p', 'rho', 'mass_approx']]
    labels = df[['citf_torsion', 'citf_velocity_anomaly']]
    return features, labels

def preprocess_gw_data(df):
    """Preprocess GW data for CITF ML."""
    df = df.dropna(subset=['strain_sample'])
    
    # Features
    df['detector_mass'] = 40  # LIGO mirror mass (kg)
    df['distance'] = 4000  # Arm length (m)
    
    # Labels (refine with Earth CITF and S)
    df['citf_shift'] = ((2 * (T_N * S_GW) * df['detector_mass'] * df['distance']) / 1e15) ** 0.5 / df['distance']
    df['citf_energy'] = T_N * S_GW * df['detector_mass'] * df['distance']
    
    features = df[['detector_mass', 'distance', 'strain_sample']]
    labels = df[['citf_shift', 'citf_energy']]
    return features, labels

def preprocess_cmb_data(df):
    """Preprocess CMB data for CITF ML."""
    df = df.dropna(subset=['map_value'])
    
    # Features
    df['nside'] = df['nside']
    
    # Labels (refine with cosmic S)
    df['citf_scaling'] = S_COSMIC
    df['citf_temperature_anomaly'] = df['map_value'] * S_COSMIC
    
    features = df[['nside', 'map_value']]
    labels = df[['citf_scaling', 'citf_temperature_anomaly']]
    return features, labels

def preprocess_uhecr_data(df):
    """Preprocess UHECR data for CITF ML."""
    df = df.dropna(subset=['energy'])
    
    # Features
    df['proton_mass'] = 1.6726e-27  # kg
    df['distance'] = 1e24  # m (100 Mpc)
    
    # Labels (refine with source CITF and S)
    T_p_source = 8.994e12  # Neutron star CITF
    df['citf_energy'] = T_p_source * S_UHECR * df['proton_mass'] * df['distance']
    df['citf_conduction_factor'] = S_UHECR
    
    features = df[['proton_mass', 'distance', 'energy', 'ra', 'dec']]
    labels = df[['citf_energy', 'citf_conduction_factor']]
    return features, labels

def preprocess_stellar_motion_data(df):
    """Preprocess Stellar Motion data for CITF ML."""
    df = df.dropna(subset=['ra', 'dec'])
    
    # Features
    df['omega_p'] = np.where(df['source'] == 'SDSS', 
                            df['redshift'] * 3e8 / 3.0857e19,  # Approximate from redshift
                            ((df['pmra']**2 + df['pmdec']**2)**0.5) * 4.74e-6 / 3.0857e19)  # Gaia proper motion
    df['rho'] = 1410  # Solar-like density (kg/m^3)
    df['mass_approx'] = 1  # kg
    
    # Labels
    df['citf_torsion'] = T_N + ALPHA * df['omega_p'] * df['rho']
    df['citf_anomaly'] = ((2 * df['citf_torsion'] * df['mass_approx'] * 1) / df['mass_approx']) ** 0.5
    
    features = df[['omega_p', 'rho', 'mass_approx', 'ra', 'dec']]
    labels = df[['citf_torsion', 'citf_anomaly']]
    return features, labels

def preprocess_planetary_data(df):
    """Preprocess Planetary data for CITF ML."""
    df = df.dropna(subset=['rotation_period', 'density'])
    
    # Features
    df['omega_p'] = 2 * np.pi / (df['rotation_period'] * 86400)
    df['rho'] = df['density'] * 1000
    df['mass_approx'] = 1000
    
    # Labels
    df['citf_torsion'] = T_N + ALPHA * df['omega_p'] * df['rho']
    df['citf_velocity_anomaly'] = ((2 * df['citf_torsion'] * df['mass_approx'] * 5e5) / df['mass_approx']) ** 0.5
    
    features = df[['omega_p', 'rho', 'mass_approx']]
    labels = df[['citf_torsion', 'citf_velocity_anomaly']]
    return features, labels

def preprocess_all_data():
    """Preprocess all datasets and save for ML."""
    tables = {
        'exoplanet': preprocess_exoplanet_data,
        'gw_event': preprocess_gw_data,
        'cmb_data': preprocess_cmb_data,
        'uhecr': preprocess_uhecr_data,
        'stellar_motion': preprocess_stellar_motion_data,
        'planetary_data': preprocess_planetary_data
    }
    
    all_features = []
    all_labels = []
    
    for table, preprocess_func in tables.items():
        df = fetch_data(table)
        if df is None or df.empty:
            print(f"No data found for {table}")
            continue
        
        features, labels = preprocess_func(df)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Combine features and labels
        combined = pd.concat([pd.DataFrame(features_scaled, columns=features.columns), 
                             labels.reset_index(drop=True)], axis=1)
        combined['dataset'] = table  # Track source
        
        all_features.append(features_scaled)
        all_labels.append(labels)
        
        # Save individual dataset
        combined.to_csv(f"data/preprocessed_{table}.csv", index=False)
        print(f"Preprocessed {table} data saved with {len(combined)} entries")
    
    # Concatenate all data
    if all_features and all_labels:
        features_all = np.vstack(all_features)
        labels_all = pd.concat(all_labels, ignore_index=True)
        
        # Save aggregated dataset
        combined_all = pd.concat([pd.DataFrame(features_all, columns=['feat_' + str(i) for i in range(features_all.shape[1])]), 
                                 labels_all], axis=1)
        combined_all.to_csv("data/citf_ml_dataset.csv", index=False)
        np.save("data/citf_features.npy", features_all)
        labels_all.to_csv("data/citf_labels.csv", index=False)
        print(f"Aggregated dataset saved with {len(combined_all)} total entries")

def main():
    preprocess_all_data()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
