from flask import Flask, jsonify, request, render_template
import requests
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.heasarc import Heasarc
from astroquery.gaia import Gaia
from astroquery.sdss import SDSS
import gwpy.timeseries
import healpy as hp
import os
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import threading
import time
import sqlite3
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

# Database configuration using DATABASE_URL from Railway
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///Database.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define Models for CITF-related data storage
class Exoplanet(db.Model):
    """Table for NASA Exoplanet Archive data."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    radius = db.Column(db.Float)  # Jupiter radii
    density = db.Column(db.Float)  # g/cm^3
    orbital_period = db.Column(db.Float)  # days
    citf_torsion = db.Column(db.Float)  # Calculated T_p
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class GWEvent(db.Model):
    """Table for LIGO LOSC GW strain data."""
    id = db.Column(db.Integer, primary_key=True)
    event_name = db.Column(db.String(50), nullable=False)
    gps_time = db.Column(db.Float)
    strain_sample = db.Column(db.Float)  # Sample strain value
    citf_shift = db.Column(db.Float)  # Predicted h_shift
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class CMBData(db.Model):
    """Table for Planck CMB data."""
    id = db.Column(db.Integer, primary_key=True)
    nside = db.Column(db.Integer)
    map_value = db.Column(db.Float)  # Sample CMB value
    citf_scaling = db.Column(db.Float)  # S factor influence
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class UHECR(db.Model):
    """Table for HEASARC UHECR data."""
    id = db.Column(db.Integer, primary_key=True)
    energy = db.Column(db.Float)  # eV
    ra = db.Column(db.Float)
    dec = db.Column(db.Float)
    citf_energy = db.Column(db.Float)  # Predicted CITF energy
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class StellarMotion(db.Model):
    """Table for SDSS/Gaia stellar data."""
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(20))  # 'SDSS' or 'Gaia'
    ra = db.Column(db.Float)
    dec = db.Column(db.Float)
    redshift = db.Column(db.Float)  # SDSS
    pmra = db.Column(db.Float)  # Gaia proper motion
    pmdec = db.Column(db.Float)
    citf_anomaly = db.Column(db.Float)  # Predicted CITF effect
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class PlanetaryData(db.Model):
    """Table for PDS planetary data."""
    id = db.Column(db.Integer, primary_key=True)
    planet = db.Column(db.String(20), nullable=False)
    radius = db.Column(db.Float)  # km
    density = db.Column(db.Float)  # g/cm^3
    rotation_period = db.Column(db.Float)  # days
    citf_torsion = db.Column(db.Float)  # Calculated T_p
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database tables
with app.app_context():
    if not status['training'].get('loss'):
        status['training']['loss'] = []
        status['training']['mae'] = []
        status['training']['torsion_error'] = []

# Global status for preprocessing and training
status = {
    'preprocessing': {'running': False, 'progress': 0, 'message': 'Idle'},
    'training': {'running': False, 'progress': 0, 'message': 'Idle', 'loss': [], 'mae': [], 'torsion_error': []},
    'testing': {'running': False, 'progress': 0, 'message': 'Idle', 'mse': [], 'mae': [], 'torsion_error': []}
}
model = None
X_train, X_test, y_train_dict, y_test_dict, output_dict = None, None, None, None, None

# Helper function for error handling
def api_error(message, status_code=500):
    return jsonify({"error": message}), status_code

# CITF Calculation Functions
def calculate_citf_torsion(omega_p, rho):
    """Calculate T_p = T_n + alpha * omega_p * rho."""
    T_n = 1e-6
    alpha = 2e-8
    return T_n + alpha * omega_p * rho

def calculate_citf_velocity_anomaly(T_p, mass=1, distance=1):
    """Calculate Delta v from CITF energy conduction."""
    E = T_p * mass * distance
    return (2 * E / mass) ** 0.5

def calculate_citf_shift(T_p, mass=40, distance=4000):
    """Calculate h_shift for GW chirality."""
    E = T_p * mass * distance
    k = 1e15  # Mirror stiffness
    delta_L = (2 * E / k) ** 0.5
    return delta_L / distance

def calculate_citf_energy(T_p, mass, distance):
    """Calculate energy conducted by CITF."""
    return T_p * mass * distance

def fetch_data(table_name):
    conn = sqlite3.connect(DATABASE_PATH)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
    
# Data Fetching and Storage with Features
@app.route('/exoplanet_data', methods=['GET'])
def get_exoplanet_data():
    try:
        query = "SELECT pl_name, pl_radj, pl_dens, pl_orbper FROM ps WHERE pl_dens IS NOT NULL LIMIT 10"
        data = NasaExoplanetArchive.query_criteria(table="ps", select=query).to_pandas()
        for _, row in data.iterrows():
            omega_p = 2 * 3.14159 / (row['pl_orbper'] * 86400) if row['pl_orbper'] else 0
            rho = row['pl_dens'] * 1000
            torsion = calculate_citf_torsion(omega_p, rho)
            velocity_anomaly = calculate_citf_velocity_anomaly(torsion, mass=1000, distance=5e5)
            planet = Exoplanet(name=row['pl_name'], radius=row['pl_radj'], density=row['pl_dens'], 
                              orbital_period=row['pl_orbper'], citf_torsion=torsion, 
                              citf_velocity_anomaly=velocity_anomaly)
            db.session.add(planet)
        db.session.commit()
        return jsonify({"status": "success", "data": data.to_dict(orient='records')})
    except Exception as e:
        return api_error(f"Failed to fetch exoplanet data: {str(e)}")

@app.route('/losc_gw_data', methods=['GET'])
def get_losc_gw_data():
    try:
        event = request.args.get('event', 'GW150914')
        url = f"https://gwosc.org/archive/data/{event}/H-H1_LOSC_4_V2-1126259446-32.hdf5"
        local_file = "gw_data.hdf5"
        response = requests.get(url)
        with open(local_file, 'wb') as f:
            f.write(response.content)
        strain = gwpy.timeseries.TimeSeries.read(local_file, format='hdf5.losc')
        sample_strain = strain.value[0]
        T_p = 1.008e-6  # Local Earth CITF
        shift = calculate_citf_shift(T_p * 5e-13, mass=40, distance=4000)  # S = 5e-13
        energy = calculate_citf_energy(T_p * 5e-13, mass=40, distance=4000)
        gw_event = GWEvent(event_name=event, gps_time=strain.times.value[0], 
                          strain_sample=sample_strain, citf_shift=shift, citf_energy=energy)
        db.session.add(gw_event)
        db.session.commit()
        data = {"time": strain.times.value[:100].tolist(), "strain": strain.value[:100].tolist()}
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to fetch LOSC data: {str(e)}")

@app.route('/planck_cmb_data', methods=['GET'])
def get_planck_cmb_data():
    try:
        map_file = "data/HFI_SkyMap_545_2048_R3.00_full.fits"
        if not os.path.exists(map_file):
            return api_error("Planck CMB FITS file not found - preload required")
        cmb_map = hp.read_map(map_file, verbose=False)
        nside = hp.get_nside(cmb_map)
        sample_value = cmb_map[0]
        scaling = 2e-7  # Cosmic S
        temp_anomaly = sample_value * scaling  # Placeholder Delta T
        cmb_data = CMBData(nside=nside, map_value=sample_value, citf_scaling=scaling, 
                          citf_temperature_anomaly=temp_anomaly)
        db.session.add(cmb_data)
        db.session.commit()
        data = {"nside": nside, "map_sample": cmb_map[:100].tolist()}
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to fetch Planck data: {str(e)}")

@app.route('/heasarc_uhecr_data', methods=['GET'])
def get_heasarc_uhecr_data():
    try:
        heasarc = Heasarc()
        table = "fermi8y"
        query = heasarc.query_mission_list(mission=table, fields="energy,ra,dec", 
                                         conditions="energy>1e9")
        data = query.to_pandas()
        for _, row in data.iterrows():
            T_p = 8.994e12  # Neutron star source
            energy = calculate_citf_energy(T_p * 1e-17, mass=1.6726e-27, distance=1e24)
            conduction_factor = 1e-17  # S for UHECR
            uhecr = UHECR(energy=row['energy'], ra=row['ra'], dec=row['dec'], 
                         citf_energy=energy, citf_conduction_factor=conduction_factor)
            db.session.add(uhecr)
        db.session.commit()
        return jsonify({"status": "success", "data": data.head(10).to_dict(orient='records')})
    except Exception as e:
        return api_error(f"Failed to fetch HEASARC data: {str(e)}")

@app.route('/stellar_motion_data', methods=['GET'])
def get_stellar_motion_data():
    try:
        sdss_query = "SELECT TOP 10 ra, dec, z FROM SpecObj WHERE class='STAR' AND z IS NOT NULL"
        sdss_data = SDSS.query_sql(sdss_query).to_pandas()
        for _, row in sdss_data.iterrows():
            omega_p = row['z'] * 3e8 / 3.0857e19  # Approximate from redshift
            rho = 1410  # Solar-like density
            torsion = calculate_citf_torsion(omega_p, rho)
            anomaly = calculate_citf_velocity_anomaly(torsion, mass=1, distance=1)
            motion = StellarMotion(source='SDSS', ra=row['ra'], dec=row['dec'], redshift=row['z'], 
                                  citf_torsion=torsion, citf_anomaly=anomaly)
            db.session.add(motion)
        gaia_query = "SELECT TOP 10 ra, dec, pmra, pmdec FROM gaiadr2.gaia_source WHERE pmra IS NOT NULL"
        gaia_data = Gaia.launch_job(gaia_query).get_results().to_pandas()
        for _, row in gaia_data.iterrows():
            omega_p = ((row['pmra']**2 + row['pmdec']**2)**0.5) * 4.74e-6 / 3.0857e19  # rad/s
            torsion = calculate_citf_torsion(omega_p, rho=1410)
            anomaly = calculate_citf_velocity_anomaly(torsion)
            motion = StellarMotion(source='Gaia', ra=row['ra'], dec=row['dec'], 
                                  pmra=row['pmra'], pmdec=row['pmdec'], citf_torsion=torsion, 
                                  citf_anomaly=anomaly)
            db.session.add(motion)
        db.session.commit()
        return jsonify({"status": "success", "sdss": sdss_data.to_dict(orient='records'), 
                        "gaia": gaia_data.to_dict(orient='records')})
    except Exception as e:
        return api_error(f"Failed to fetch stellar motion data: {str(e)}")

@app.route('/pds_planetary_data', methods=['GET'])
def get_pds_planetary_data():
    try:
        file_path = "data/jupiter_data.csv"
        if not os.path.exists(file_path):
            return api_error("PDS Jupiter CSV not found - preload required")
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            omega_p = 2 * 3.14159 / (row['rotation_period_days'] * 86400)
            rho = row['density_g_cm3'] * 1000
            torsion = calculate_citf_torsion(omega_p, rho)
            velocity_anomaly = calculate_citf_velocity_anomaly(torsion, mass=1000, distance=5e5)
            planet = PlanetaryData(planet='Jupiter', radius=row['radius_km'], 
                                  density=row['density_g_cm3'], rotation_period=row['rotation_period_days'], 
                                  citf_torsion=torsion, citf_velocity_anomaly=velocity_anomaly)
            db.session.add(planet)
        db.session.commit()
        return jsonify({"status": "success", "data": df.head(10).to_dict(orient='records')})
    except Exception as e:
        return api_error(f"Failed to fetch PDS data: {str(e)}")

@app.route('/kaggle_astrophysics_data', methods=['GET'])
def get_kaggle_astrophysics_data():
    try:
        dataset = "keplerspace/kepler-exoplanet-search-results"
        os.system(f"kaggle datasets download -d {dataset} -p data/ --unzip")
        file_path = "data/cumulative.csv"
        if not os.path.exists(file_path):
            return api_error("Kaggle dataset download failed - check API token")
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            omega_p = 2 * 3.14159 / (row['koi_period'] * 86400) if row['koi_period'] else 0
            rho = row.get('koi_dens', 0) * 1000
            torsion = calculate_citf_torsion(omega_p, rho)
            velocity_anomaly = calculate_citf_velocity_anomaly(torsion, mass=1000, distance=5e5)
            planet = Exoplanet(name=f"koi_{row['kepid']}", radius=row['koi_prad'], density=rho,
                              orbital_period=row['koi_period'], citf_torsion=torsion, 
                              citf_velocity_anomaly=velocity_anomaly)
            db.session.add(planet)
        db.session.commit()
        return jsonify({"status": "success", "data": df.head(10).to_dict(orient='records')})
    except Exception as e:
        return api_error(f"Failed to fetch Kaggle data: {str(e)}")

# Query Endpoints
@app.route('/get_exoplanet_data', methods=['GET'])
def get_exoplanet_data_query():
    try:
        limit = int(request.args.get('limit', 10))
        planets = Exoplanet.query.order_by(Exoplanet.timestamp.desc()).limit(limit).all()
        data = [{"id": p.id, "name": p.name, "radius": p.radius, "density": p.density, 
                 "orbital_period": p.orbital_period, "citf_torsion": p.citf_torsion, 
                 "citf_velocity_anomaly": p.citf_velocity_anomaly, "timestamp": str(p.timestamp)} 
                for p in planets]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to query exoplanet data: {str(e)}")

@app.route('/get_gw_data', methods=['GET'])
def get_gw_data_query():
    try:
        limit = int(request.args.get('limit', 10))
        gw_events = GWEvent.query.order_by(GWEvent.timestamp.desc()).limit(limit).all()
        data = [{"id": e.id, "event_name": e.event_name, "gps_time": e.gps_time, 
                 "strain_sample": e.strain_sample, "citf_shift": e.citf_shift, 
                 "citf_energy": e.citf_energy, "timestamp": str(e.timestamp)} 
                for e in gw_events]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to query GW data: {str(e)}")

@app.route('/get_cmb_data', methods=['GET'])
def get_cmb_data_query():
    try:
        limit = int(request.args.get('limit', 10))
        cmb_entries = CMBData.query.order_by(CMBData.timestamp.desc()).limit(limit).all()
        data = [{"id": c.id, "nside": c.nside, "map_value": c.map_value, 
                 "citf_scaling": c.citf_scaling, "citf_temperature_anomaly": c.citf_temperature_anomaly, 
                 "timestamp": str(c.timestamp)} for c in cmb_entries]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to query CMB data: {str(e)}")

@app.route('/get_uhecr_data', methods=['GET'])
def get_uhecr_data_query():
    try:
        limit = int(request.args.get('limit', 10))
        uhecrs = UHECR.query.order_by(UHECR.timestamp.desc()).limit(limit).all()
        data = [{"id": u.id, "energy": u.energy, "ra": u.ra, "dec": u.dec, 
                 "citf_energy": u.citf_energy, "citf_conduction_factor": u.citf_conduction_factor, 
                 "timestamp": str(u.timestamp)} for u in uhecrs]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to query UHECR data: {str(e)}")

@app.route('/get_stellar_motion_data', methods=['GET'])
def get_stellar_motion_data_query():
    try:
        limit = int(request.args.get('limit', 10))
        motions = StellarMotion.query.order_by(StellarMotion.timestamp.desc()).limit(limit).all()
        data = [{"id": m.id, "source": m.source, "ra": m.ra, "dec": m.dec, "redshift": m.redshift, 
                 "pmra": m.pmra, "pmdec": m.pmdec, "citf_anomaly": m.citf_anomaly, 
                 "citf_torsion": m.citf_torsion, "timestamp": str(m.timestamp)} for m in motions]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to query stellar motion data: {str(e)}")

@app.route('/get_planetary_data', methods=['GET'])
def get_planetary_data_query():
    try:
        limit = int(request.args.get('limit', 10))
        planets = PlanetaryData.query.order_by(PlanetaryData.timestamp.desc()).limit(limit).all()
        data = [{"id": p.id, "planet": p.planet, "radius": p.radius, "density": p.density, 
                 "rotation_period": p.rotation_period, "citf_torsion": p.citf_torsion, 
                 "citf_velocity_anomaly": p.citf_velocity_anomaly, "timestamp": str(p.timestamp)} 
                for p in planets]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return api_error(f"Failed to query planetary data: {str(e)}")

# HUD Web Page Route
@app.route('/hud', methods=['GET'])
def hud_page():
    """Render the HUD-style monitoring page."""
    return render_template('index.html')

# ML Preprocessing Logic (from preprocess_citf_data.py)
def connect_db():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_data(table_name):
    conn = connect_db()
    if conn is None:
        return None
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_all_data():
    global status
    status['preprocessing']['running'] = True
    status['preprocessing']['progress'] = 0
    status['preprocessing']['message'] = "Starting preprocessing..."

    tables = {
        'exoplanet': lambda df: (
            df.dropna(subset=['orbital_period', 'density']),
            ['omega_p', 'rho', 'mass_approx'],
            ['citf_torsion', 'citf_velocity_anomaly'],
            lambda df: df.assign(
                omega_p=2 * np.pi / (df['orbital_period'] * 86400),
                rho=df['density'] * 1000,  # Convert g/cm^3 to kg/m^3
                mass_approx=1000,  # Placeholder spacecraft mass (kg), consistent with Flyby
                citf_torsion=lambda x: calculate_citf_torsion(x['omega_p'], x['rho']),
                citf_velocity_anomaly=lambda x: calculate_citf_velocity_anomaly(x['citf_torsion'], mass=1000, distance=5e5)
            )
        ),
        'gw_event': lambda df: (
            df.dropna(subset=['strain_sample']),
            ['detector_mass', 'distance', 'strain_sample'],
            ['citf_shift', 'citf_energy'],
            lambda df: df.assign(
                detector_mass=40,  # LIGO mirror mass (kg)
                distance=4000,  # LIGO arm length (m)
                citf_shift=lambda x: calculate_citf_shift(1.008e-6 * 5e-13, mass=40, distance=4000),  # S = 5e-13 for GW
                citf_energy=lambda x: calculate_citf_energy(1.008e-6 * 5e-13, mass=40, distance=4000)
            )
        ),
        'cmb_data': lambda df: (
            df.dropna(subset=['map_value']),
            ['nside', 'map_value'],
            ['citf_scaling', 'citf_temperature_anomaly'],
            lambda df: df.assign(
                nside=df['nside'],  # HEALPix resolution
                map_value=df['map_value'],  # CMB temperature (K)
                citf_scaling=2e-7,  # Cosmic S factor
                citf_temperature_anomaly=lambda x: x['map_value'] * x['citf_scaling']  # Delta T
            )
        ),
        'uhecr': lambda df: (
            df.dropna(subset=['energy']),
            ['proton_mass', 'distance', 'energy', 'ra', 'dec'],
            ['citf_energy', 'citf_conduction_factor'],
            lambda df: df.assign(
                proton_mass=1.6726e-27,  # Proton mass (kg)
                distance=1e24,  # 100 Mpc (m)
                energy=df['energy'],  # Raw energy (eV)
                ra=df['ra'],  # Right Ascension
                dec=df['dec'],  # Declination
                citf_energy=lambda x: calculate_citf_energy(8.994e12 * 1e-17, mass=x['proton_mass'], distance=x['distance']),
                citf_conduction_factor=1e-17  # S for UHECR
            )
        ),
        'stellar_motion': lambda df: (
            df.dropna(subset=['ra', 'dec']),
            ['omega_p', 'rho', 'mass_approx', 'ra', 'dec'],
            ['citf_torsion', 'citf_anomaly'],
            lambda df: df.assign(
                omega_p=np.where(df['source'] == 'SDSS',
                                df['redshift'] * 3e8 / 3.0857e19,  # Approximate from redshift (rad/s)
                                ((df['pmra']**2 + df['pmdec']**2)**0.5) * 4.74e-6 / 3.0857e19),  # Gaia proper motion (rad/s)
                rho=1410,  # Solar-like density (kg/m^3)
                mass_approx=1,  # Placeholder stellar mass (kg)
                ra=df['ra'],
                dec=df['dec'],
                citf_torsion=lambda x: calculate_citf_torsion(x['omega_p'], x['rho']),
                citf_anomaly=lambda x: calculate_citf_velocity_anomaly(x['citf_torsion'], mass=1, distance=1)
            )
        ),
        'planetary_data': lambda df: (
            df.dropna(subset=['rotation_period', 'density']),
            ['omega_p', 'rho', 'mass_approx'],
            ['citf_torsion', 'citf_velocity_anomaly'],
            lambda df: df.assign(
                omega_p=2 * np.pi / (df['rotation_period'] * 86400),  # rad/s
                rho=df['density'] * 1000,  # kg/m^3
                mass_approx=1000,  # Placeholder spacecraft mass (kg)
                citf_torsion=lambda x: calculate_citf_torsion(x['omega_p'], x['rho']),
                citf_velocity_anomaly=lambda x: calculate_citf_velocity_anomaly(x['citf_torsion'], mass=1000, distance=5e5)
            )
        )
    }
    
    all_features = []
    all_labels = []
    datasets = []
    
    for table, (clean_func, feat_cols, label_cols, transform_func) in tables.items():
        df = fetch_data(table)
        if df is None or df.empty:
            status['preprocessing']['message'] = f"No data for {table}"
            continue
        
        df_clean = clean_func(df)
        df_transformed = transform_func(df_clean)
        
        features = df_transformed[feat_cols]
        labels = df_transformed[label_cols]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        all_features.append(features_scaled)
        all_labels.append(labels)
        datasets.extend([table] * len(features_scaled))
        
        status['preprocessing']['progress'] = min(status['preprocessing']['progress'] + 100 // len(tables), 100)
        status['preprocessing']['message'] = f"Processed {table}"
    
    if all_features and all_labels:
        features_all = np.vstack(all_features)
        labels_all = pd.concat(all_labels, ignore_index=True)
        datasets_all = np.array(datasets)
        
        np.save("data/citf_features.npy", features_all)
        labels_all.to_csv("data/citf_labels.csv", index=False)
        pd.DataFrame(datasets_all, columns=['dataset']).to_csv("data/citf_datasets.csv", index=False)
        
        status['preprocessing']['progress'] = 100
        status['preprocessing']['message'] = "Preprocessing completed"
    else:
        status['preprocessing']['message'] = "No valid data processed"
    
    status['preprocessing']['running'] = False
    return features_all, labels_all, datasets_all

# ML Model Logic (from train_citf_model.py)
def build_model(input_dim, output_dict):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = {dataset: tf.keras.layers.Dense(len(cols), activation='linear', name=dataset)(x) 
               for dataset, cols in output_dict.items()}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    loss_dict = {dataset: 'mse' for dataset in output_dict.keys()}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=loss_dict, metrics={'exoplanet': 'mae', 'gw_event': 'mae'})
    return model

def preprocess_ml_data(features, labels, dataset_tags):
    output_dict = {
        'exoplanet': ['citf_torsion', 'citf_velocity_anomaly'],
        'gw_event': ['citf_shift', 'citf_energy']
    }
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    for dataset in output_dict.keys():
        mask = dataset_tags == dataset
        X_subset = features[mask]
        y_subset = labels[output_dict[dataset]][mask].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        X_train[dataset] = X_tr
        X_test[dataset] = X_te
        y_train_dict[dataset] = y_tr
        y_test_dict[dataset] = y_te
    return X_train, X_test, y_train_dict, y_test_dict, output_dict

@app.route('/preprocess', methods=['POST'])
def preprocess():
    threading.Thread(target=preprocess_all_data).start()
    return jsonify({"status": "Preprocessing started"})

@app.route('/train', methods=['POST'])
def train():
    global model, status
    epochs = int(request.args.get('epochs', 50))
    batch_size = int(request.args.get('batch_size', 32))
    learning_rate = float(request.args.get('learning_rate', 0.001))
    dataset = request.args.get('dataset', 'all')
    
    def run_train():
        global status, model
        if X_train is None:
            status['training']['message'] = "Preprocessing required first"
            status['training']['running'] = False
            return
        
        status['training']['running'] = True
        status['training']['loss'] = []
        status['training']['mae'] = []
        status['training']['torsion_error'] = []
        
        input_dim = next(iter(X_train.values())).shape[1]
        selected_dict = output_dict if dataset == 'all' else {dataset: output_dict[dataset]}
        model = build_model(input_dim, selected_dict)
        model.optimizer.learning_rate = learning_rate
        
        train_data = {d: X_train[d] for d in selected_dict.keys()}
        train_labels = {d: y_train_dict[d] for d in selected_dict.keys()}
        val_data = {d: X_test[d] for d in selected_dict.keys()}
        val_labels = {d: y_test_dict[d] for d in selected_dict.keys()}
        
        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, 
                            validation_data=(val_data, val_labels), verbose=0)
        
        T_n = 1e-6
        for epoch in range(epochs):
            status['training']['progress'] = (epoch + 1) * 100 // epochs
            status['training']['message'] = f"Training {dataset} epoch {epoch + 1}/{epochs}"
            loss = history.history[f'{list(selected_dict.keys())[0]}_loss'][epoch]
            mae = history.history[f'{list(selected_dict.keys())[0]}_mae'][epoch]
            torsion_preds = model.predict(train_data, verbose=0)[list(selected_dict.keys())[0]][:, 0]
            torsion_true = train_labels[list(selected_dict.keys())[0]][:, 0]
            torsion_error = np.mean(np.abs(torsion_preds - T_n) / T_n)
            status['training']['loss'].append(loss)
            status['training']['mae'].append(mae)
            status['training']['torsion_error'].append(torsion_error)
            time.sleep(0.1)
        
        model.save("models/citf_model.h5")
        status['training']['running'] = False
        status['training']['message'] = "Training completed"
    
    threading.Thread(target=run_train).start()
    return jsonify({"status": "Training started"})

@app.route('/test', methods=['POST'])
def test():
    global model, status, latest_predictions
    dataset = request.args.get('dataset', 'all')
    
    def run_test():
        global status, latest_predictions
        if model is None or X_test is None:
            status['testing']['message'] = "Model training required first"
            status['testing']['running'] = False
            return
        
        status['testing']['running'] = True
        status['testing']['mse'] = []
        status['testing']['mae'] = []
        status['testing']['torsion_error'] = []
        
        selected_dict = output_dict if dataset == 'all' else {dataset: output_dict[dataset]}
        test_data = {d: X_test[d] for d in selected_dict.keys()}
        test_labels = {d: y_test_dict[d] for d in selected_dict.keys()}
        
        predictions = model.predict(test_data, verbose=0)
        latest_predictions = {d: p if len(selected_dict) > 1 else predictions for d, p in predictions.items()}
        
        T_n = 1e-6
        for i in range(1, 101):
            status['testing']['progress'] = i
            status['testing']['message'] = f"Testing {dataset} {i}%"
            time.sleep(0.05)
        
        for d in selected_dict.keys():
            y_true = test_labels[d]
            y_pred = latest_predictions[d]
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            torsion_true = y_true[:, 0]  # Assuming first column is torsion-like
            torsion_pred = y_pred[:, 0]
            torsion_error = np.mean(np.abs(torsion_pred - T_n) / T_n)
            status['testing']['mse'].append(mse)
            status['testing']['mae'].append(mae)
            status['testing']['torsion_error'].append(torsion_error)
        
        status['testing']['running'] = False
        status['testing']['message'] = "Testing completed"
    
    threading.Thread(target=run_test).start()
    return jsonify({"status": "Testing started"})

@app.route('/stream_data', methods=['GET'])
def stream_data():
    """Stream real-time CITF data from SQLite."""
    def generate():
        with app.app_context():  # Add application context
            while True:
                data = {
                    'exoplanet_count': Exoplanet.query.count(),
                    'gw_events': GWEvent.query.count(),
                    'cmb_entries': CMBData.query.count(),
                    'uhecr_count': UHECR.query.count(),
                    'stellar_count': StellarMotion.query.count(),
                    'planetary_count': PlanetaryData.query.count(),
                    'status': status,
                    'predictions': {d: p[:5].tolist() for d, p in latest_predictions.items()} if latest_predictions else {}
                }
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1)  # Update every second
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/visualize_ml', methods=['GET'])
def visualize_ml():
    """Generate temperature-color-coded ML visualizations."""
    section = request.args.get('section', 'training')
    
    plt.figure(figsize=(8, 6))
    if section == 'training' and status['training'].get('loss'):  # Use .get() to avoid KeyError
        plt.plot(status['training']['loss'], label='Loss', color='cyan')
        plt.plot(status['training']['mae'], label='MAE', color='magenta')
        plt.plot(status['training']['torsion_error'], label='Torsion Error rel. T_n', color='yellow')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, color='gray', linestyle='--', alpha=0.3)
    elif section == 'testing' and status['testing'].get('mse'):  # Use .get() here too
        datasets = list(output_dict.keys())[:len(status['testing']['mse'])]
        for i, (dataset, mse, mae, torsion_err) in enumerate(zip(datasets, status['testing']['mse'], 
                                                                  status['testing']['mae'], 
                                                                  status['testing']['torsion_error'])):
            plt.bar(i - 0.3, mse, 0.3, label=f'{dataset} MSE', color='cyan')
            plt.bar(i, mae, 0.3, label=f'{dataset} MAE', color='magenta')
            plt.bar(i + 0.3, torsion_err, 0.3, label=f'{dataset} Torsion Err', color='yellow')
        plt.xticks(range(len(datasets)), datasets, rotation=45)
        plt.title('Testing Metrics')
        plt.ylabel('Error Value')
        plt.legend()
    else:
        # Simplified default case—show placeholder if no data
        plt.text(0.5, 0.5, 'No Data Yet', ha='center', va='center', color='white', fontsize=14)
        plt.title(f'CITF {section.capitalize()} Progress')
        plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({"image": f"data:image/png;base64,{img_base64}"})

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "Flask CITF API with SQLite running", "version": "1.1"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
