import os
import pandas as pd
import numpy as np
from faker import Faker
from setuptools import setup, find_packages
import pytest

# Initialize Faker
fake = Faker()

# Data generation functions
def generate_hospitals(num_hospitals):
    hospitals = []
    for i in range(num_hospitals):
        hospitals.append({
            'hospital_id': f'H{i+1}',
            'name': fake.company() + ' Hospital',
            'location': fake.address()
        })
    return pd.DataFrame(hospitals)

def generate_clinics(num_clinics, num_hospitals):
    clinics = []
    for i in range(num_clinics):
        clinics.append({
            'clinic_id': f'C{i+1}',
            'name': fake.company() + ' Clinic',
            'location': fake.address(),
            'hospital_id': f'H{np.random.randint(1, num_hospitals + 1)}'
        })
    return pd.DataFrame(clinics)

def generate_providers(num_providers, num_hospitals, num_clinics):
    providers = []
    for i in range(num_providers):
        providers.append({
            'provider_id': f'P{i+1}',
            'name': fake.name(),
            'specialization': np.random.choice(['General Practitioner', 'Cardiologist', 'Dermatologist', 'Neurologist', 'Pediatrician']),
            'hospital_id': f'H{np.random.randint(1, num_hospitals + 1)}',
            'clinic_id': f'C{np.random.randint(1, num_clinics + 1)}'
        })
    return pd.DataFrame(providers)

def generate_patients(num_patients):
    patients = []
    for i in range(num_patients):
        patients.append({
            'patient_id': f'PAT{i+1}',
            'name': fake.name(),
            'date_of_birth': fake.date_of_birth(minimum_age=0, maximum_age=90),
            'address': fake.address(),
            'phone': fake.phone_number(),
            'email': fake.email()
        })
    return pd.DataFrame(patients)

def generate_appointments(num_appointments, num_providers, num_patients):
    appointments = []
    for i in range(num_appointments):
        appointments.append({
            'appointment_id': f'APPT{i+1}',
            'patient_id': f'PAT{np.random.randint(1, num_patients + 1)}',
            'provider_id': f'P{np.random.randint(1, num_providers + 1)}',
            'appointment_date': fake.date_time_this_year(),
            'reason': np.random.choice(['Checkup', 'Follow-up', 'Consultation', 'Emergency'])
        })
    return pd.DataFrame(appointments)

def generate_medical_assets(num_assets, num_hospitals, num_clinics):
    assets = []
    for i in range(num_assets):
        assets.append({
            'asset_id': f'A{i+1}',
            'type': np.random.choice(['MRI Machine', 'X-Ray Machine', 'Ultrasound Machine', 'CT Scanner', 'Defibrillator']),
            'manufacturer': fake.company(),
            'model': fake.word() + str(np.random.randint(100, 999)),
            'installation_date': fake.date_this_decade(),
            'status': np.random.choice(['Operational', 'Under Maintenance', 'Out of Service']),
            'hospital_id': f'H{np.random.randint(1, num_hospitals + 1)}',
            'clinic_id': f'C{np.random.randint(1, num_clinics + 1)}'
        })
    return pd.DataFrame(assets)

def generate_data():
    hospitals = generate_hospitals(10)
    clinics = generate_clinics(30, 10)
    providers = generate_providers(100, 10, 30)
    patients = generate_patients(200)
    appointments = generate_appointments(500, 100, 200)
    medical_assets = generate_medical_assets(50, 10, 30)

    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')

    hospitals.to_csv('data/hospitals.csv', index=False)
    clinics.to_csv('data/clinics.csv', index=False)
    providers.to_csv('data/providers.csv', index=False)
    patients.to_csv('data/patients.csv', index=False)
    appointments.to_csv('data/appointments.csv', index=False)
    medical_assets.to_csv('data/medical_assets.csv', index=False)
    print("Data generated successfully!")

# Generate data
generate_data()

# Test case script
def test_read_data_files():
    DATA_FILES = [
        'data/hospitals.csv',
        'data/clinics.csv',
        'data/providers.csv',
        'data/patients.csv',
        'data/appointments.csv',
        'data/medical_assets.csv'
    ]

    for file in DATA_FILES:
        try:
            df = pd.read_csv(file, low_memory=False)
            print(f"Successfully read {file}")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin-1', low_memory=False)
            print(f"Successfully read {file} with latin-1 encoding")
        
        print(df.head(4))  # Print first 4 rows
        
        # Check and convert data types if needed
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str)
                print(f"Column {col} in {file} converted to string")

# Run tests
test_read_data_files()

# Setup configuration
setup(
    name='my_project',
    version='0.1',
    packages=find_packages(),
    description='Generate healthcare data mimicking NHS structure for data analysis',
    author='Your Name',
    author_email='your.email@example.com',
    install_requires=[
        'pandas',
        'numpy',
        'faker',
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'generate_data=generate_data:generate_data',
        ],
    },
)
