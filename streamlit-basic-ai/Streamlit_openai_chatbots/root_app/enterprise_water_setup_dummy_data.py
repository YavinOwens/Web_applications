import os
import pandas as pd
import numpy as np
from faker import Faker
from setuptools import setup, find_packages
import streamlit as st
import plotly.express as px

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
            'specialization': np.random.choice(['General Practitioner', 'Cardiologist', 'Dermatologist', 'Neurologist', 'Pediatrician', 'Social Care Worker', 'Bank Staff']),
            'associated_with': np.random.choice(['Hospital', 'Provider Company', 'Social Care Agency']),
            'organization': fake.company(),
            'hospital_id': f'H{np.random.randint(1, num_hospitals + 1)}' if np.random.choice([True, False]) else None,
            'clinic_id': f'C{np.random.randint(1, num_clinics + 1)}' if np.random.choice([True, False]) else None
        })
    return pd.DataFrame(providers)

def generate_patients(num_patients, providers):
    patients = []
    provider_ids = providers['provider_id'].tolist()
    hospital_ids = providers['hospital_id'].tolist()
    for i in range(num_patients):
        assigned_provider = np.random.choice(provider_ids)
        assigned_hospital = providers.loc[providers['provider_id'] == assigned_provider, 'hospital_id'].values[0]
        patients.append({
            'patient_id': f'PAT{i+1}',
            'name': fake.name(),
            'date_of_birth': fake.date_of_birth(minimum_age=0, maximum_age=90),
            'address': fake.address(),
            'phone': fake.phone_number(),
            'email': fake.email(),
            'assigned_provider_id': assigned_provider,
            'assigned_hospital_id': assigned_hospital
        })
    return pd.DataFrame(patients)

def generate_appointments(num_appointments, providers, patients):
    appointments = []
    for i in range(num_appointments):
        appointment_date = fake.date_time_this_year()
        waiting = np.random.choice([True, False])
        appointments.append({
            'appointment_id': f'APPT{i+1}',
            'patient_id': f'PAT{np.random.randint(1, len(patients) + 1)}',
            'provider_id': f'P{np.random.randint(1, len(providers) + 1)}',
            'appointment_date': appointment_date,
            'reason': np.random.choice(['Checkup', 'Follow-up', 'Consultation', 'Emergency']),
            'waiting': waiting,
            'waiting_time': np.random.randint(0, 120) if waiting else 0
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

def generate_workforce(num_employees, num_hospitals):
    workforce = []
    for i in range(num_employees):
        workforce.append({
            'employee_id': f'E{i+1}',
            'name': fake.name(),
            'position': np.random.choice(['Nurse', 'Doctor', 'Technician', 'Administrative Staff']),
            'hospital_id': f'H{np.random.randint(1, num_hospitals + 1)}',
            'salary': fake.random_number(digits=5, fix_len=True),
            'hire_date': fake.date_this_decade()
        })
    return pd.DataFrame(workforce)

def generate_case_allocations(num_cases, providers, patients):
    case_allocations = []
    for i in range(num_cases):
        case_allocations.append({
            'case_id': f'C{i+1}',
            'patient_id': f'PAT{np.random.randint(1, len(patients) + 1)}',
            'provider_id': f'P{np.random.randint(1, len(providers) + 1)}',
            'case_description': fake.text(max_nb_chars=200),
            'case_status': np.random.choice(['Open', 'Closed', 'Pending'])
        })
    return pd.DataFrame(case_allocations)

def generate_data():
    num_hospitals = 10
    num_clinics = 30
    num_providers = 100
    num_patients = 200
    num_appointments = 500
    num_assets = 50
    num_employees = 100
    num_cases = 200
    
    hospitals = generate_hospitals(num_hospitals)
    clinics = generate_clinics(num_clinics, num_hospitals)
    providers = generate_providers(num_providers, num_hospitals, num_clinics)
    patients = generate_patients(num_patients, providers)
    appointments = generate_appointments(num_appointments, providers, patients)
    medical_assets = generate_medical_assets(num_assets, num_hospitals, num_clinics)
    workforce = generate_workforce(num_employees, num_hospitals)
    case_allocations = generate_case_allocations(num_cases, providers, patients)

    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')

    hospitals.to_csv('data/hospitals.csv', index=False)
    clinics.to_csv('data/clinics.csv', index=False)
    providers.to_csv('data/providers.csv', index=False)
    patients.to_csv('data/patients.csv', index=False)
    appointments.to_csv('data/appointments.csv', index=False)
    medical_assets.to_csv('data/medical_assets.csv', index=False)
    workforce.to_csv('data/workforce.csv', index=False)
    case_allocations.to_csv('data/case_allocations.csv', index=False)
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
        'data/medical_assets.csv',
        'data/workforce.csv',
        'data/case_allocations.csv'
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

# Streamlit app
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

st.title('Healthcare Dashboard')

# Load data functions with caching
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

hospitals = load_data('data/hospitals.csv')
clinics = load_data('data/clinics.csv')
providers = load_data('data/providers.csv')
patients = load_data('data/patients.csv')
appointments = load_data('data/appointments.csv')
medical_assets = load_data('data/medical_assets.csv')
workforce = load_data('data/workforce.csv')
case_allocations = load_data('data/case_allocations.csv')

# Ensure the appointment_date column is parsed as datetime
appointments['appointment_date'] = pd.to_datetime(appointments['appointment_date'])

# Define data filtering function
def filter_data(selected_hospital, search_id, search_id_type):
    filtered_clinics = clinics.copy()
    filtered_providers = providers.copy()
    filtered_patients = patients.copy()
    filtered_appointments = appointments.copy()
    filtered_medical_assets = medical_assets.copy()
    filtered_workforce = workforce.copy()
    filtered_case_allocations = case_allocations.copy()

    # Filter data based on selected hospital
    if selected_hospital != 'All':
        hospital_id = hospitals[hospitals['name'] == selected_hospital]['hospital_id'].values[0]
        filtered_clinics = filtered_clinics[filtered_clinics['hospital_id'] == hospital_id]
        filtered_providers = filtered_providers[filtered_providers['hospital_id'] == hospital_id]
        filtered_appointments = filtered_appointments[filtered_appointments['provider_id'].isin(filtered_providers['provider_id'])]
        filtered_patients = filtered_patients[filtered_patients['assigned_hospital_id'] == hospital_id]
        filtered_medical_assets = filtered_medical_assets[filtered_medical_assets['hospital_id'] == hospital_id]
        filtered_workforce = filtered_workforce[filtered_workforce['hospital_id'] == hospital_id]
        filtered_case_allocations = filtered_case_allocations[filtered_case_allocations['provider_id'].isin(filtered_providers['provider_id'])]

    # Filter data based on search ID
    if search_id:
        if search_id_type == 'Patient ID':
            filtered_patients = filtered_patients[filtered_patients['patient_id'] == search_id]
            filtered_appointments = filtered_appointments[filtered_appointments['patient_id'] == search_id]
            filtered_providers = filtered_providers[filtered_providers['provider_id'].isin(filtered_patients['assigned_provider_id'].unique())]
            if not filtered_providers.empty:
                filtered_medical_assets = filtered_medical_assets[filtered_medical_assets['hospital_id'].isin(filtered_providers['hospital_id'].unique())]
        elif search_id_type == 'Provider ID':
            filtered_providers = filtered_providers[filtered_providers['provider_id'] == search_id]
            filtered_appointments = filtered_appointments[filtered_appointments['provider_id'] == search_id]
            filtered_patients = filtered_patients[filtered_patients['assigned_provider_id'] == search_id]
            if not filtered_providers.empty:
                filtered_medical_assets = filtered_medical_assets[filtered_medical_assets['hospital_id'].isin(filtered_providers['hospital_id'].unique())]
        elif search_id_type == 'Appointment ID':
            filtered_appointments = filtered_appointments[filtered_appointments['appointment_id'] == search_id]
            filtered_providers = filtered_providers[filtered_providers['provider_id'].isin(filtered_appointments['provider_id'].unique())]
            filtered_patients = filtered_patients[filtered_patients['patient_id'].isin(filtered_appointments['patient_id'].unique())]
            if not filtered_providers.empty:
                filtered_medical_assets = filtered_medical_assets[filtered_medical_assets['hospital_id'].isin(filtered_providers['hospital_id'].unique())]

    return {
        "clinics": filtered_clinics,
        "providers": filtered_providers,
        "patients": filtered_patients,
        "appointments": filtered_appointments,
        "medical_assets": filtered_medical_assets,
        "workforce": filtered_workforce,
        "case_allocations": filtered_case_allocations
    }

# Define the pages
def overview_page():
    st.header('Overview')

    # Sidebar
    with st.sidebar:
        st.header('Filter')
        hospital_options = ['All'] + hospitals['name'].tolist()
        selected_hospital = st.selectbox('Choose Hospital', hospital_options)

        search_id_type = st.selectbox('Search by ID', ['Patient ID', 'Provider ID', 'Appointment ID'])
        search_id = st.text_input('Enter ID')

    # Filter data
    filtered_data = filter_data(selected_hospital, search_id, search_id_type)
    
    clinics = filtered_data["clinics"]
    providers = filtered_data["providers"]
    patients = filtered_data["patients"]
    appointments = filtered_data["appointments"]
    medical_assets = filtered_data["medical_assets"]
    workforce = filtered_data["workforce"]
    case_allocations = filtered_data["case_allocations"]

    # Main panel
    with st.container():
        st.header('Metrics')
        total_hospitals = len(hospitals)
        avg_appointments_per_day = appointments.groupby(pd.to_datetime(appointments['appointment_date']).dt.date).size().mean()
        total_appointments = len(appointments)
        total_patients = len(patients)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Hospitals", total_hospitals)
        col2.metric("Average Appointments/Day", f"{avg_appointments_per_day:.1f}")
        col3.metric("Total Appointments", total_appointments)
        col4.metric("Total Patients", total_patients)

    with st.container():
        st.header('Visualizations')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Number of Appointments by Hour')
            appointments['hour'] = pd.to_datetime(appointments['appointment_date']).dt.hour
            appt_by_hour = appointments.groupby('hour').size().reset_index(name='count')
            fig = px.bar(appt_by_hour, x='hour', y='count', title='Number of Appointments by Hour')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('Appointments per Day')
            appointments['date'] = pd.to_datetime(appointments['appointment_date']).dt.date
            appt_by_date = appointments.groupby('date').size().reset_index(name='count')
            fig = px.bar(appt_by_date, x='date', y='count', title='Appointments per Day')
            st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.subheader('Provider Specializations')
        specializations = providers['specialization'].value_counts().reset_index()
        specializations.columns = ['specialization', 'count']
        fig = px.bar(specializations, x='specialization', y='count', title='Provider Specializations')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander('Patients Assigned to Each Provider'):
        patients_per_provider = patients['assigned_provider_id'].value_counts().reset_index()
        patients_per_provider.columns = ['provider_id', 'patient_count']
        provider_patient_info = pd.merge(providers[['provider_id', 'name', 'specialization']], patients_per_provider, left_on='provider_id', right_on='provider_id', how='left').fillna(0)
        st.dataframe(provider_patient_info)

        st.subheader('Manage Patients')
        action = st.selectbox('Select Action', ['Add New Patient', 'Update Existing Patient'])
        
        if action == 'Add New Patient':
            with st.form('Add Patient Form'):
                new_patient_id = st.text_input('Patient ID')
                new_patient_name = st.text_input('Name')
                new_patient_dob = st.date_input('Date of Birth')
                new_patient_address = st.text_input('Address')
                new_patient_phone = st.text_input('Phone')
                new_patient_email = st.text_input('Email')
                new_assigned_provider = st.selectbox('Assigned Provider', providers['provider_id'])
                new_assigned_hospital = st.selectbox('Assigned Hospital', hospitals['hospital_id'])
                submit_button = st.form_submit_button(label='Add Patient')

                if submit_button:
                    new_patient = {
                        'patient_id': new_patient_id,
                        'name': new_patient_name,
                        'date_of_birth': pd.to_datetime(new_patient_dob),
                        'address': new_patient_address,
                        'phone': new_patient_phone,
                        'email': new_patient_email,
                        'assigned_provider_id': new_assigned_provider,
                        'assigned_hospital_id': new_assigned_hospital
                    }
                    patients = patients.append(new_patient, ignore_index=True)
                    patients.to_csv('data/patients.csv', index=False)
                    st.success('New patient added successfully!')

        elif action == 'Update Existing Patient':
            with st.form('Update Patient Form'):
                patient_to_update = st.selectbox('Select Patient to Update', patients['patient_id'])
                update_field = st.selectbox('Field to Update', ['name', 'date_of_birth', 'address', 'phone', 'email', 'assigned_provider_id', 'assigned_hospital_id'])
                new_value = st.text_input('New Value')
                submit_button = st.form_submit_button(label='Update Patient')

                if submit_button:
                    if update_field == 'date_of_birth':
                        new_value = pd.to_datetime(new_value)
                    patients.loc[patients['patient_id'] == patient_to_update, update_field] = new_value
                    patients.to_csv('data/patients.csv', index=False)
                    st.success('Patient updated successfully!')

    with st.expander('Current Waiting Appointments'):
        waiting_appts = appointments[appointments['waiting'] == True]
        waiting_appts_display = waiting_appts[['patient_id', 'provider_id', 'appointment_date', 'reason', 'waiting_time']]
        st.dataframe(waiting_appts_display)

        median_waiting_time = waiting_appts['waiting_time'].median()
        for index, row in waiting_appts.iterrows():
            delta = row['waiting_time'] - median_waiting_time
            arrow = "🔺" if delta > 0 else "🔻"
            st.metric(label=f"Appointment {row['appointment_id']}", value=row['waiting_time'], delta=f"{arrow} {abs(delta)}")

        st.subheader('Manage Appointments')
        action = st.selectbox('Select Action', ['Add New Appointment', 'Update Existing Appointment'])
        
        if action == 'Add New Appointment':
            with st.form('Add Appointment Form'):
                new_appt_id = st.text_input('Appointment ID')
                new_patient_id = st.selectbox('Patient ID', patients['patient_id'])
                new_provider_id = st.selectbox('Provider ID', providers['provider_id'])
                new_appt_date = st.date_input('Appointment Date')
                new_reason = st.selectbox('Reason', ['Checkup', 'Follow-up', 'Consultation', 'Emergency'])
                new_waiting = st.selectbox('Waiting', [True, False])
                new_waiting_time = st.number_input('Waiting Time', min_value=0)
                submit_button = st.form_submit_button(label='Add Appointment')

                if submit_button:
                    new_appt = {
                        'appointment_id': new_appt_id,
                        'patient_id': new_patient_id,
                        'provider_id': new_provider_id,
                        'appointment_date': pd.to_datetime(new_appt_date),
                        'reason': new_reason,
                        'waiting': new_waiting,
                        'waiting_time': new_waiting_time
                    }
                    appointments = appointments.append(new_appt, ignore_index=True)
                    appointments.to_csv('data/appointments.csv', index=False)
                    st.success('New appointment added successfully!')

        elif action == 'Update Existing Appointment':
            with st.form('Update Appointment Form'):
                appt_to_update = st.selectbox('Select Appointment to Update', appointments['appointment_id'])
                update_field = st.selectbox('Field to Update', ['patient_id', 'provider_id', 'appointment_date', 'reason', 'waiting', 'waiting_time'])
                new_value = st.text_input('New Value')
                submit_button = st.form_submit_button(label='Update Appointment')

                if submit_button:
                    if update_field == 'appointment_date':
                        new_value = pd.to_datetime(new_value)
                    if update_field == 'waiting':
                        new_value = new_value == 'True'
                    appointments.loc[appointments['appointment_id'] == appt_to_update, update_field] = new_value
                    appointments.to_csv('data/appointments.csv', index=False)
                    st.success('Appointment updated successfully!')

def workforce_management_page():
    st.header('Workforce Management')

    # Sidebar
    with st.sidebar:
        st.header('Filter')
        hospital_options = ['All'] + hospitals['name'].tolist()
        selected_hospital = st.selectbox('Choose Hospital', hospital_options)

        search_id_type = st.selectbox('Search by ID', ['Patient ID', 'Provider ID', 'Appointment ID'])
        search_id = st.text_input('Enter ID')

    # Filter data
    filtered_data = filter_data(selected_hospital, search_id, search_id_type)
    
    workforce = filtered_data["workforce"]
    case_allocations = filtered_data["case_allocations"]

    # Metrics
    with st.container():
        st.header('Metrics')
        total_employees = len(workforce)
        avg_salary = workforce['salary'].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total Employees", total_employees)
        col2.metric("Average Salary", f"${avg_salary:.2f}")

    # Visualizations
    with st.container():
        st.header('Visualizations')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Workforce Distribution by Position')
            workforce_distribution = workforce['position'].value_counts().reset_index()
            workforce_distribution.columns = ['position', 'count']
            fig = px.bar(workforce_distribution, x='position', y='count', title='Workforce Distribution by Position')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('Employee Information')
            st.dataframe(workforce)

    st.expander('Case Allocations')
    st.dataframe(case_allocations)

def patient_management_page():
    st.header('Patient Management')

    # Sidebar
    with st.sidebar:
        st.header('Filter')
        hospital_options = ['All'] + hospitals['name'].tolist()
        selected_hospital = st.selectbox('Choose Hospital', hospital_options)

        search_id_type = st.selectbox('Search by ID', ['Patient ID', 'Provider ID', 'Appointment ID'])
        search_id = st.text_input('Enter ID')

    # Filter data
    filtered_data = filter_data(selected_hospital, search_id, search_id_type)
    
    patients = filtered_data["patients"]
    appointments = filtered_data["appointments"]

    # Metrics
    with st.container():
        st.header('Metrics')
        total_patients = len(patients)
        avg_waiting_time = appointments['waiting_time'].mean()
        median_waiting_time = appointments['waiting_time'].median()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", total_patients)
        col2.metric("Average Waiting Time", f"{avg_waiting_time:.2f} minutes")
        col3.metric("Median Waiting Time", f"{median_waiting_time:.2f} minutes")

    # Visualizations
    with st.container():
        st.header('Visualizations')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Number of Appointments by Reason')
            appt_reason_distribution = appointments['reason'].value_counts().reset_index()
            appt_reason_distribution.columns = ['reason', 'count']
            fig = px.bar(appt_reason_distribution, x='reason', y='count', title='Number of Appointments by Reason')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('Patient Information')
            st.dataframe(patients)

    st.subheader('Appointment Information')
    st.dataframe(appointments)

def asset_management_page():
    st.header('Asset Management')

    # Sidebar
    with st.sidebar:
        st.header('Filter')
        hospital_options = ['All'] + hospitals['name'].tolist()
        selected_hospital = st.selectbox('Choose Hospital', hospital_options)

        search_id_type = st.selectbox('Search by ID', ['Patient ID', 'Provider ID', 'Appointment ID'])
        search_id = st.text_input('Enter ID')

    # Filter data
    filtered_data = filter_data(selected_hospital, search_id, search_id_type)
    
    medical_assets = filtered_data["medical_assets"]

    # Metrics
    with st.container():
        st.header('Metrics')
        total_assets = len(medical_assets)
        operational_assets = len(medical_assets[medical_assets['status'] == 'Operational'])

        col1, col2 = st.columns(2)
        col1.metric("Total Assets", total_assets)
        col2.metric("Operational Assets", operational_assets)

    # Display metrics for each asset status
    status_counts = medical_assets['status'].value_counts()
    cols = st.columns(len(status_counts))
    for i, (status, count) in enumerate(status_counts.items()):
        cols[i].metric(status, count)

    # Visualizations
    with st.container():
        st.header('Visualizations')
        st.subheader('Asset Distribution by Type')
        asset_distribution = medical_assets['type'].value_counts().reset_index()
        asset_distribution.columns = ['type', 'count']
        fig = px.bar(asset_distribution, x='type', y='count', title='Asset Distribution by Type')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Medical Assets Information')
    st.dataframe(medical_assets)

    # Asset management actions
    st.subheader('Manage Assets')
    action = st.selectbox('Select Action', ['Add New Asset', 'Update Existing Asset', 'Delete Asset'])

    if action == 'Add New Asset':
        with st.form('Add Asset Form'):
            new_asset_id = st.text_input('Asset ID')
            new_asset_type = st.selectbox('Asset Type', ['MRI Machine', 'X-Ray Machine', 'Ultrasound Machine', 'CT Scanner', 'Defibrillator'])
            new_asset_manufacturer = st.text_input('Manufacturer')
            new_asset_model = st.text_input('Model')
            new_asset_installation_date = st.date_input('Installation Date')
            new_asset_status = st.selectbox('Status', ['Operational', 'Under Maintenance', 'Out of Service'])
            new_asset_hospital_id = st.selectbox('Hospital ID', hospitals['hospital_id'])
            new_asset_clinic_id = st.selectbox('Clinic ID', clinics['clinic_id'])
            submit_button = st.form_submit_button(label='Add Asset')

            if submit_button:
                new_asset = {
                    'asset_id': new_asset_id,
                    'type': new_asset_type,
                    'manufacturer': new_asset_manufacturer,
                    'model': new_asset_model,
                    'installation_date': pd.to_datetime(new_asset_installation_date),
                    'status': new_asset_status,
                    'hospital_id': new_asset_hospital_id,
                    'clinic_id': new_asset_clinic_id
                }
                medical_assets = medical_assets.append(new_asset, ignore_index=True)
                medical_assets.to_csv('data/medical_assets.csv', index=False)
                st.success('New asset added successfully!')

    elif action == 'Update Existing Asset':
        with st.form('Update Asset Form'):
            asset_to_update = st.selectbox('Select Asset to Update', medical_assets['asset_id'])
            update_field = st.selectbox('Field to Update', ['type', 'manufacturer', 'model', 'installation_date', 'status', 'hospital_id', 'clinic_id'])
            new_value = st.text_input('New Value')
            submit_button = st.form_submit_button(label='Update Asset')

            if submit_button:
                if update_field == 'installation_date':
                    new_value = pd.to_datetime(new_value)
                medical_assets.loc[medical_assets['asset_id'] == asset_to_update, update_field] = new_value
                medical_assets.to_csv('data/medical_assets.csv', index=False)
                st.success('Asset updated successfully!')

    elif action == 'Delete Asset':
        with st.form('Delete Asset Form'):
            asset_to_delete = st.selectbox('Select Asset to Delete', medical_assets['asset_id'])
            submit_button = st.form_submit_button(label='Delete Asset')

            if submit_button:
                medical_assets = medical_assets[medical_assets['asset_id'] != asset_to_delete]
                medical_assets.to_csv('data/medical_assets.csv', index=False)
                st.success('Asset deleted successfully!')

# Page navigation
page = st.sidebar.selectbox("Select Page", ["Overview", "Workforce Management", "Patient Management", "Asset Management"])

if page == "Overview":
    overview_page()
elif page == "Workforce Management":
    workforce_management_page()
elif page == "Patient Management":
    patient_management_page()
elif page == "Asset Management":
    asset_management_page()

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
        'streamlit',
        'plotly',
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'generate_data=generate_data:generate_data',
        ],
    },
)
