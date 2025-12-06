import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import uuid # Import uuid for unique filenames

# ML Imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors # For finding similar therapy plans
import joblib # To save and load the trained model and encoders
import numpy as np # For numerical operations

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Database Configuration
instance_path = os.path.join(os.getcwd(), 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'slp_clinic.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- ML Model Setup ---
# Paths for saving/loading ML components
ML_MODELS_DIR = 'ml_models'
os.makedirs(ML_MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ML_MODELS_DIR, 'therapy_plan_nn_model.pkl')
ENCODER_PATH = os.path.join(ML_MODELS_DIR, 'diagnosis_onehot_encoder.pkl')
SCALER_PATH = os.path.join(ML_MODELS_DIR, 'age_scaler.pkl')
DATA_PATH = 'agewise_therapy_plan_dataset.csv' # Assuming the CSV is in the same directory

# Global variables to hold the trained model, encoders, and original data
# These will be loaded once when the app starts
therapy_plan_model = None
diagnosis_encoder = None
age_scaler = None
agewise_df = None

def train_and_save_ml_model():
    """
    Loads the dataset, trains the NearestNeighbors model and encoders,
    and saves them to disk. This function should be called once.
    """
    global therapy_plan_model, diagnosis_encoder, age_scaler, agewise_df

    try:
        # Load the dataset
        try:
            agewise_df = pd.read_csv(DATA_PATH)
            print("ML: Dataset loaded successfully.")
        except FileNotFoundError:
            print(f"ML: Error: '{DATA_PATH}' not found. Please ensure the dataset is in the correct directory.")
            agewise_df = None # Ensure agewise_df is None on file not found
            raise # Re-raise to be caught by the outer except block

        # Handle missing values: Drop rows where 'Diagnosis', 'Age', 'Goals', or 'Activities' are missing
        initial_rows = agewise_df.shape[0]
        agewise_df.dropna(subset=['Diagnosis', 'Age', 'Goals', 'Activities'], inplace=True)
        print(f"ML: Dataset after dropping NaNs: {agewise_df.shape[0]} rows (removed {initial_rows - agewise_df.shape[0]} rows).")

        # Check if DataFrame is empty after dropping NaNs
        if agewise_df.empty:
            raise ValueError("Dataset is empty after removing rows with missing 'Diagnosis', 'Age', 'Goals', or 'Activities'. Cannot train model.")

        # Feature Engineering
        # One-hot encode 'Diagnosis'
        diagnosis_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        try:
            # Check if 'Diagnosis' column exists before accessing
            if 'Diagnosis' not in agewise_df.columns:
                raise KeyError("Column 'Diagnosis' not found in the dataset.")
            encoded_diagnoses = diagnosis_encoder.fit_transform(agewise_df[['Diagnosis']])
        except (ValueError, KeyError) as ve:
            raise ValueError(f"Error during OneHotEncoding of 'Diagnosis': {ve}. Check 'Diagnosis' column data or existence.")

        encoded_diagnosis_df = pd.DataFrame(encoded_diagnoses, columns=diagnosis_encoder.get_feature_names_out(['Diagnosis']))

        # Scale 'Age'
        age_scaler = StandardScaler()
        try:
            # Check if 'Age' column exists before accessing
            if 'Age' not in agewise_df.columns:
                raise KeyError("Column 'Age' not found in the dataset.")
            scaled_age = age_scaler.fit_transform(agewise_df[['Age']])
        except (ValueError, KeyError) as ve:
            raise ValueError(f"Error during StandardScaler fit of 'Age': {ve}. Check 'Age' column data or existence.")
        scaled_age_df = pd.DataFrame(scaled_age, columns=['Age_scaled'])

        # Combine features
        # Reset index of agewise_df, encoded_diagnosis_df, and scaled_age_df to ensure proper concatenation
        agewise_df.reset_index(drop=True, inplace=True)
        encoded_diagnosis_df.reset_index(drop=True, inplace=True)
        scaled_age_df.reset_index(drop=True, inplace=True)

        features = pd.concat([encoded_diagnosis_df, scaled_age_df], axis=1)

        # Train NearestNeighbors model
        # We use NearestNeighbors to find the closest data point based on input features
        # The index of the closest data point will then be used to retrieve the Goals and Activities
        therapy_plan_model = NearestNeighbors(n_neighbors=1, algorithm='brute')
        try:
            therapy_plan_model.fit(features)
        except ValueError as ve:
            raise ValueError(f"Error fitting NearestNeighbors model. This might be due to empty or invalid features DataFrame: {ve}")

        print("ML: NearestNeighbors model trained.")

        # Save the trained model and encoders
        joblib.dump(therapy_plan_model, MODEL_PATH)
        joblib.dump(diagnosis_encoder, ENCODER_PATH)
        joblib.dump(age_scaler, SCALER_PATH)
        print("ML: Model and encoders saved.")

    except FileNotFoundError:
        # This will be caught if pd.read_csv re-raises FileNotFoundError
        print(f"ML: Model training skipped due to missing dataset file: {DATA_PATH}")
        agewise_df = None
    except Exception as e:
        # This will now catch the more specific ValueErrors as well
        print(f"ML: An unexpected error occurred during model training: {e}")
        agewise_df = None # Set to None if error occurs


def load_ml_model():
    """
    Loads the pre-trained ML model and encoders from disk.
    """
    global therapy_plan_model, diagnosis_encoder, age_scaler, agewise_df
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SCALER_PATH):
            therapy_plan_model = joblib.load(MODEL_PATH)
            diagnosis_encoder = joblib.load(ENCODER_PATH)
            age_scaler = joblib.load(SCALER_PATH)
            agewise_df = pd.read_csv(DATA_PATH) # Reload original data for lookup
            # Ensure consistency by applying dropna again
            initial_rows = agewise_df.shape[0]
            agewise_df.dropna(subset=['Diagnosis', 'Age', 'Goals', 'Activities'], inplace=True)
            agewise_df.reset_index(drop=True, inplace=True)
            if agewise_df.empty:
                raise ValueError("Loaded dataset is empty after consistency checks. Cannot use model.")
            print("ML: Model and encoders loaded successfully.")
        else:
            print("ML: No pre-trained model found. Training a new one...")
            train_and_save_ml_model()
    except Exception as e:
        print(f"ML: Error loading model components: {e}")
        therapy_plan_model = None
        diagnosis_encoder = None
        age_scaler = None
        agewise_df = None # Clear data if loading fails

# --- Flask Routes ---

# Call load_ml_model when the app starts
with app.app_context():
    load_ml_model()


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video uploads."""
    if 'video' not in request.files:
        return jsonify({"message": "No video part in the request"}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({"message": "No selected video"}), 400
    if video and allowed_file(video.filename):
        # Generate a unique filename using UUID
        ext = video.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        video.save(filepath)
        return jsonify({
            "message": "Video uploaded successfully",
            "filename": unique_filename,
            "url": f"/uploads/{unique_filename}"
        }), 200
    return jsonify({"message": "Invalid file type"}), 400

# --- User model: use consistent role naming ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='therapist') # Use 'therapist' as default
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False) # Changed from date_of_birth for ML
    diagnosis = db.Column(db.String(255))
    assigned_therapist_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Clinician managing the patient
    status = db.Column(db.String(50), default='Active') # Added patient status
    date_joined = db.Column(db.DateTime, default=datetime.utcnow) # Date patient joined

class TherapyPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    therapist_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Therapist who created the plan
    goals = db.Column(db.Text, nullable=False)
    activities = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default='Pending Review') # e.g., 'Pending Review', 'Approved', 'Revisions Needed'
    supervisor_feedback = db.Column(db.Text) # Supervisor's feedback
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

class SessionNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    therapist_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String, nullable=False)
    note = db.Column(db.Text, nullable=False)
    plan_id = db.Column(db.Integer, nullable=True)
    video_url = db.Column(db.String, nullable=True)



@app.route('/session_notes', methods=['GET'])
def get_session_notes():
    patient_id = request.args.get('patient_id')
    notes = SessionNote.query.filter_by(patient_id=patient_id).all()
    return jsonify([{
        'id': n.id,
        'patient_id': n.patient_id,
        'therapist_id': n.therapist_id,
        'date': n.date,
        'note': n.note,
        'plan_id': n.plan_id,
        'video_url': n.video_url
    } for n in notes])

class ProgressReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    therapist_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String(10), nullable=False) #YYYY-MM-DD
    summary = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default='Pending Review') # e.g., 'Pending Review', 'Approved', 'Revisions Needed'
    supervisor_evaluation = db.Column(db.Text) # Supervisor's evaluation
    clinical_rating = db.Column(db.String(50)) # e.g., 'Excellent', 'Good', 'Needs Improvement'

# Route for creating initial data (from original app.py)
@app.route('/create_initial_data', methods=['POST'])
def create_initial_data_route():
    with app.app_context():
        create_initial_data()
    return jsonify({"message": "Initial data created/checked."}), 200

# Helper function to create initial data (from original app.py)
def create_initial_data():
    db.create_all() # Create tables if they don't exist

    # Create a default admin if not exists
    if not User.query.filter_by(username='admin1').first():
        admin = User(name='Admin User', username='admin1', role='admin')
        admin.set_password('adminpass')
        db.session.add(admin)
        db.session.commit()
        print("Default admin 'admin1' created.")

    # Create a default therapist if not exists
    if not User.query.filter_by(username='clinician1').first():
        therapist = User(name='Clini Cian', username='clinician1', role='therapist')
        therapist.set_password('password')
        db.session.add(therapist)
        db.session.commit()
        print("Default therapist 'clinician1' created.")

    # Create a default supervisor if not exists
    if not User.query.filter_by(username='supervisor1').first():
        supervisor = User(name='Super Visor', username='supervisor1', role='supervisor')
        supervisor.set_password('password')
        db.session.add(supervisor)
        db.session.commit()
        print("Default supervisor 'supervisor1' created.")

    # Add a sample patient if not exists
    therapist = User.query.filter_by(username='clinician1').first()
    if therapist and not Patient.query.filter_by(name='John Doe').first():
        john = Patient(name='John Doe', age=10, diagnosis='Childhood Apraxia of Speech', assigned_therapist_id=therapist.id)
        db.session.add(john)
        db.session.commit()
        print("Sample patient 'John Doe' created.")

    # Add a sample therapy plan for John Doe if not exists
    john_patient = Patient.query.filter_by(name='John Doe').first()
    if john_patient and not TherapyPlan.query.filter_by(patient_id=john_patient.id).first():
        john_plan = TherapyPlan(
            patient_id=john_patient.id,
            therapist_id=therapist.id,
            goals="Improve speech intelligibility by 50% in 6 months. Increase functional communication.",
            activities="DTTC exercises, repetitive drills, prosody practice, AAC introduction.",
            status="Pending Review"
        )
        db.session.add(john_plan)
        db.session.commit()
        print("Sample therapy plan for John Doe created.")

    # Add a sample session note for John Doe if not exists
    john_plan = TherapyPlan.query.filter_by(patient_id=john_patient.id).first() if john_patient else None
    if john_patient and john_plan and not SessionNote.query.filter_by(patient_id=john_patient.id).first():
        john_note = SessionNote(
            patient_id=john_patient.id,
            therapist_id=therapist.id,
            date='2024-06-10',
            note="S: Parent reports John is more verbal at home. O: John produced 7/10 target words. A: Good progress. P: Continue current plan.",
            plan_id=john_plan.id,
            video_url=None
        )
        db.session.add(john_note)
        db.session.commit()
        print("Sample session note for John Doe created.")

    # Add a sample progress report for John Doe if not exists
    if john_patient and not ProgressReport.query.filter_by(patient_id=john_patient.id).first():
        john_report = ProgressReport(
            patient_id=john_patient.id,
            therapist_id=therapist.id,
            date='2024-06-15',
            summary="John has shown consistent improvement in speech intelligibility, utilizing learned strategies effectively.",
            status='Pending Review'
        )
        db.session.add(john_report)
        db.session.commit()
        print("Sample progress report for John Doe created.")

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role_attempt = data.get('role') # Role attempted by the user during login

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        # Check if the role matches the user's actual role in the DB
        if user.role == role_attempt:
            return jsonify({"message": "Login successful", "userId": user.id, "name": user.name, "role": user.role}), 200
        else:
            return jsonify({"message": f"Login failed: You are a {user.role}, but tried to log in as {role_attempt}."}), 401
    else:
        return jsonify({"message": "Invalid username or password"}), 401

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')

    if not name or not username or not password or not role:
        return jsonify({"message": "All fields are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists"}), 409

    new_user = User(name=name, username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": f"User {username} ({role}) registered successfully"}), 201

@app.route('/users', methods=['GET'])
def get_users():
    # Only admins and supervisors can view all users
    # For simplicity, current implementation allows all to view users with specific roles,
    # but could be restricted further by requiring authentication token.
    role_filter = request.args.get('role')
    query = User.query

    if role_filter:
        query = query.filter_by(role=role_filter)

    users = query.all()
    users_data = {}
    for u in users:
        users_data[u.id] = {
            'id': u.id,
            'name': u.name,
            'username': u.username,
            'role': u.role,
            'date_joined': u.date_joined.isoformat()
        }
    return jsonify(users_data), 200

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"message": "User not found"}), 404
    
    if user.role == 'admin':
        return jsonify({"message": "Cannot delete admin user via this route"}), 403

    try:
        if user.role == 'therapist':
            Patient.query.filter_by(assigned_therapist_id=user_id).update({'assigned_therapist_id': None})
        TherapyPlan.query.filter_by(therapist_id=user_id).delete()
        SessionNote.query.filter_by(therapist_id=user_id).delete()
        ProgressReport.query.filter_by(therapist_id=user_id).delete()
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": f"User {user.username} and associated data deleted successfully."}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"Error deleting user: {str(e)}"}), 500


@app.route('/patients', methods=['POST'])
def add_patient():
    data = request.get_json()
    name = data.get('name')
    age = data.get('age')
    diagnosis = data.get('diagnosis')
    assigned_therapist_id = data.get('assigned_therapist_id')

    if not name or age is None or not diagnosis or assigned_therapist_id is None:
        return jsonify({"message": "Missing patient details"}), 400

    new_patient = Patient(name=name, age=age, diagnosis=diagnosis, assigned_therapist_id=assigned_therapist_id)
    db.session.add(new_patient)
    db.session.commit()
    return jsonify({"message": "Patient added successfully", "patient_id": new_patient.id}), 201

@app.route('/patients', methods=['GET'])
def get_patients():
    user_role = request.args.get('role')
    user_id = request.args.get('userId', type=int)

    query = Patient.query

    if user_role == 'therapist' and user_id:
        query = query.filter_by(assigned_therapist_id=user_id)

    patients = query.all()
    patients_data = []
    for p in patients:
        patients_data.append({
            'id': p.id,
            'name': p.name,
            'age': p.age,
            'diagnosis': p.diagnosis,
            'assigned_therapist_id': p.assigned_therapist_id,
            'status': p.status,
            'date_joined': p.date_joined.isoformat()
        })
    return jsonify(patients_data), 200

@app.route('/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if patient:
        return jsonify({
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'diagnosis': patient.diagnosis,
            'assigned_therapist_id': patient.assigned_therapist_id,
            'status': patient.status,
            'date_joined': patient.date_joined.isoformat()
        }), 200
    return jsonify({"message": "Patient not found"}), 404

@app.route('/patients/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient:
        return jsonify({"message": "Patient not found"}), 404

    data = request.get_json()
    patient.name = data.get('name', patient.name)
    patient.age = data.get('age', patient.age)
    patient.diagnosis = data.get('diagnosis', patient.diagnosis)
    patient.assigned_therapist_id = data.get('assigned_therapist_id', patient.assigned_therapist_id)
    patient.status = data.get('status', patient.status)
    db.session.commit()
    return jsonify({"message": "Patient updated successfully"}), 200

@app.route('/patients/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient:
        return jsonify({"message": "Patient not found"}), 404
    
    try:
        TherapyPlan.query.filter_by(patient_id=patient_id).delete()
        SessionNote.query.filter_by(patient_id=patient_id).delete()
        ProgressReport.query.filter_by(patient_id=patient_id).delete()
        db.session.delete(patient)
        db.session.commit()
        return jsonify({"message": "Patient and all associated data deleted successfully."}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"Error deleting patient: {str(e)}"}), 500


@app.route('/therapy_plans', methods=['POST'])
def add_therapy_plan():
    data = request.get_json()
    patient_id = data.get('patient_id')
    therapist_id = data.get('therapist_id')
    goals = data.get('goals')
    activities = data.get('activities')

    if not patient_id or not therapist_id or not goals or not activities:
        return jsonify({"message": "Missing therapy plan details"}), 400

    new_plan = TherapyPlan(patient_id=patient_id, therapist_id=therapist_id, goals=goals, activities=activities)
    db.session.add(new_plan)
    db.session.commit()
    return jsonify({"message": "Therapy plan added successfully", "plan_id": new_plan.id}), 201

@app.route('/therapy_plans', methods=['GET'])
def get_therapy_plans():
    patient_id = request.args.get('patient_id', type=int)
    query = TherapyPlan.query
    if patient_id:
        query = query.filter_by(patient_id=patient_id)
    
    plans = query.all()
    plans_data = []
    for p in plans:
        plans_data.append({
            'id': p.id,
            'patient_id': p.patient_id,
            'therapist_id': p.therapist_id,
            'goals': p.goals,
            'activities': p.activities,
            'status': p.status,
            'supervisor_feedback': p.supervisor_feedback,
            'date_created': p.date_created.isoformat()
        })
    return jsonify(plans_data), 200

@app.route('/therapy_plans/<int:plan_id>', methods=['GET'])
def get_therapy_plan(plan_id):
    plan = db.session.get(TherapyPlan, plan_id)
    if plan:
        return jsonify({
            'id': plan.id,
            'patient_id': plan.patient_id,
            'therapist_id': plan.therapist_id,
            'goals': plan.goals,
            'activities': plan.activities,
            'status': plan.status,
            'supervisor_feedback': plan.supervisor_feedback,
            'date_created': plan.date_created.isoformat()
        }), 200
    return jsonify({"message": "Therapy plan not found"}), 404

@app.route('/therapy_plans/<int:plan_id>', methods=['PUT'])
def update_therapy_plan(plan_id):
    plan = db.session.get(TherapyPlan, plan_id)
    if not plan:
        return jsonify({"message": "Therapy plan not found"}), 404
    data = request.get_json()
    plan.goals = data.get('goals', plan.goals)
    plan.activities = data.get('activities', plan.activities)
    db.session.commit()
    return jsonify({"message": "Therapy plan updated successfully"}), 200

@app.route('/reviews/plans/<int:plan_id>', methods=['PUT'])
def review_therapy_plan(plan_id):
    plan = db.session.get(TherapyPlan, plan_id)
    if not plan:
        return jsonify({"message": "Therapy plan not found"}), 404
    data = request.get_json()
    plan.status = data.get('status', plan.status)
    plan.supervisor_feedback = data.get('feedback', plan.supervisor_feedback)
    db.session.commit()
    return jsonify({"message": "Therapy plan reviewed successfully"}), 200


@app.route('/session_notes', methods=['POST'])
def add_session_note():
    data = request.json
    note = SessionNote(
        patient_id=data['patient_id'],
        therapist_id=data['therapist_id'],
        date=data['date'],
        note=data['note'],
        plan_id=data.get('plan_id'),
        video_url=data.get('video_url')
    )
    db.session.add(note)
    db.session.commit()
    return jsonify({'id': note.id}), 201



@app.route('/progress_reports', methods=['POST'])
def add_progress_report():
    data = request.get_json()
    patient_id = data.get('patient_id')
    therapist_id = data.get('therapist_id')
    date = data.get('date')
    summary = data.get('summary')

    if not patient_id or not therapist_id or not date or not summary:
        return jsonify({"message": "Missing progress report details"}), 400

    new_report = ProgressReport(patient_id=patient_id, therapist_id=therapist_id, date=date, summary=summary)
    db.session.add(new_report)
    db.session.commit()
    return jsonify({"message": "Progress report added successfully", "report_id": new_report.id}), 201

@app.route('/progress_reports', methods=['GET'])
def get_progress_reports():
    patient_id = request.args.get('patient_id', type=int)
    query = ProgressReport.query
    if patient_id:
        query = query.filter_by(patient_id=patient_id)
    
    reports = query.all()
    reports_data = []
    for r in reports:
        reports_data.append({
            'id': r.id,
            'patient_id': r.patient_id,
            'therapist_id': r.therapist_id,
            'date': r.date,
            'summary': r.summary,
            'status': r.status,
            'supervisor_evaluation': r.supervisor_evaluation,
            'clinical_rating': r.clinical_rating
        })
    return jsonify(reports_data), 200

@app.route('/progress_reports/<int:report_id>', methods=['GET'])
def get_progress_report(report_id):
    report = db.session.get(ProgressReport, report_id)
    if report:
        return jsonify({
            'id': report.id,
            'patient_id': report.patient_id,
            'therapist_id': report.therapist_id,
            'date': report.date,
            'summary': report.summary,
            'status': report.status,
            'supervisor_evaluation': report.supervisor_evaluation,
            'clinical_rating': report.clinical_rating
        }), 200
    return jsonify({"message": "Report not found"}), 404

@app.route('/progress_reports/<int:report_id>', methods=['PUT'])
def update_progress_report(report_id):
    report = db.session.get(ProgressReport, report_id)
    if not report:
        return jsonify({"message": "Report not found"}), 404
    data = request.get_json()
    report.date = data.get('date', report.date)
    report.summary = data.get('summary', report.summary)
    db.session.commit()
    return jsonify({"message": "Progress report updated successfully"}), 200

@app.route('/reviews/reports/<int:report_id>', methods=['PUT'])
def review_progress_report_route(report_id):
    report = db.session.get(ProgressReport, report_id)
    if not report:
        return jsonify({"message": "Report not found"}), 404

    data = request.get_json()
    report.status = data.get('status', report.status)
    report.supervisor_evaluation = data.get('evaluation', report.supervisor_evaluation)
    report.clinical_rating = data.get('clinical_rating', report.clinical_rating)
    db.session.commit()
    return jsonify({"message": "Progress report reviewed successfully"}), 200

@app.route('/therapy_plans/generate', methods=['POST'])
def generate_therapy_plan():
    """
    Generates a therapy plan based on diagnosis and age using the trained ML model.
    """
    data = request.get_json()
    diagnosis = data.get('diagnosis')
    age = data.get('age')

    if not diagnosis or age is None:
        return jsonify({"message": "Diagnosis and Age are required for plan generation"}), 400

    if therapy_plan_model is None or diagnosis_encoder is None or age_scaler is None or agewise_df is None:
        # Check if the model failed to load/train initially
        if os.path.exists(DATA_PATH) and pd.read_csv(DATA_PATH).empty:
            return jsonify({"message": "ML model cannot be loaded/trained because the dataset is empty or invalid."}), 500
        elif not os.path.exists(DATA_PATH):
             return jsonify({"message": "ML model cannot be loaded/trained because the dataset file is missing."}), 500
        else:
            return jsonify({"message": "ML model not loaded or trained. Please check backend logs for training errors."}), 500


    try:
        # Prepare input features for the model
        input_diagnosis_df = pd.DataFrame([{'Diagnosis': diagnosis}])
        encoded_input_diagnosis = diagnosis_encoder.transform(input_diagnosis_df[['Diagnosis']])
        # Ensure the encoded diagnosis has the same columns as during training
        encoded_input_diagnosis_df = pd.DataFrame(encoded_input_diagnosis, columns=diagnosis_encoder.get_feature_names_out(['Diagnosis']))

        scaled_input_age = age_scaler.transform(np.array([[age]]))
        scaled_input_age_df = pd.DataFrame(scaled_input_age, columns=['Age_scaled'])

        # Create a DataFrame for prediction
        # Ensure column order matches training data
        # Fill missing columns with zeros if the input diagnosis wasn't seen during training,
        # but OneHotEncoder with handle_unknown='ignore' should handle this by outputting zeros.
        input_features = pd.concat([encoded_input_diagnosis_df, scaled_input_age_df], axis=1)

        # Find the nearest neighbor
        distances, indices = therapy_plan_model.kneighbors(input_features)

        # Retrieve the closest plan from the original dataset
        closest_plan_index = indices[0][0]
        closest_plan = agewise_df.iloc[closest_plan_index]

        return jsonify({
            "goals": closest_plan['Goals'],
            "activities": closest_plan['Activities'],
            "source_diagnosis": closest_plan['Diagnosis'], # For transparency
            "source_age": int(closest_plan['Age']) # For transparency
        }), 200

    except ValueError as ve:
        # This often happens if a diagnosis is not seen by the encoder and handle_unknown='error'
        # or if input dimensions don't match.
        print(f"ML: Value Error during prediction: {ve}")
        return jsonify({"message": f"Could not generate plan. Input error: {ve}. Ensure diagnosis is valid and matches trained data categories."}), 400
    except Exception as e:
        print(f"ML: An unexpected error occurred during plan generation: {e}")
        return jsonify({"message": f"An internal error occurred during plan generation: {e}. Check backend logs for details."}), 500


# --- Main Run ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_initial_data()
        load_ml_model()
    # app.run(debug=True, port=5000, host='0.0.0.0') # Uncomment for network access
    app.run(debug=True, port=5000) # Set debug=False for production
