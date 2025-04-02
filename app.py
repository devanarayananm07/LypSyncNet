import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
import sys
import time
import dlib
import math
import subprocess
from collections import deque
import threading

# Define constants directly from demo/constants.py
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')
sys.path.append(DEMO_DIR)
from constants import TOTAL_FRAMES, LIP_WIDTH, LIP_HEIGHT, PAST_BUFFER_SIZE, NOT_TALKING_THRESHOLD

# Define LABEL_DICT manually to match what's in demo/predict_live.py
LABEL_DICT = {
    0: 'a', 1: 'bye', 2: 'can', 3: 'cat', 4: 'demo', 5: 'dog', 
    6: 'hello', 7: 'here', 8: 'is', 9: 'lips', 10: 'my', 
    11: 'read', 12: 'you'
}

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lip_reading.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
# Enable template reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'model_weights.h5')
FACE_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'face_weights.dat')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Copy model files if they don't exist
if not os.path.exists(MODEL_WEIGHTS_PATH):
    import shutil
    src_model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'model_weights.h5')
    if os.path.exists(src_model):
        shutil.copy2(src_model, MODEL_WEIGHTS_PATH)
    else:
        print(f"Warning: Could not find source model weights at {src_model}")

if not os.path.exists(FACE_WEIGHTS_PATH):
    import shutil
    src_face = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'face_weights.dat')
    if os.path.exists(src_face):
        shutil.copy2(src_face, FACE_WEIGHTS_PATH)
    else:
        print(f"Warning: Could not find face weights at {src_face}")

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the model
try:
    # Define the input shape
    input_shape = (TOTAL_FRAMES, LIP_HEIGHT, LIP_WIDTH, 3)
    
    # Define the model architecture to match saved weights
    # Modified to match the architecture in predict_live.py with 5 layers
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(LABEL_DICT), activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load the model weights
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model loaded successfully!")
    print("Model summary:")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Global variables for tracking
spoken_already = []
frame_count = 0
is_talking = False

# Global variables for lip reading demo
demo_detector = None
demo_predictor = None
demo_model = None
demo_cap = None
demo_curr_word_frames = []
demo_not_talking_counter = 0
demo_past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)
demo_predicted_word = None
demo_prediction_display_counter = 0
demo_spoken_already = []
demo_is_running = False
demo_count = 0

# Add a global variable to store the confidence
demo_confidence = 0.0

# Add a global variable to track if a new prediction is available
demo_new_prediction = False

# Frame processing rate control variable
demo_frame_skip = 1  # Process every nth frame
demo_is_processing = False  # Flag to prevent multiple processing threads

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Dataset model
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('datasets', lazy=True))

# Training History model
class TrainingHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('training_history', lazy=True))

# Data Contribution model
class DataContribution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('contributions', lazy=True))

# Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create admin user function
def create_admin():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', email='admin@example.com', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created!")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's contributions
    contributions = DataContribution.query.filter_by(user_id=current_user.id).order_by(DataContribution.created_at.desc()).limit(5).all()
    
    # Get recent predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(10).all()
    
    # Count contributions by word
    contribution_counts = db.session.query(
        DataContribution.word, 
        db.func.count(DataContribution.id).label('count')
    ).filter_by(user_id=current_user.id).group_by(DataContribution.word).all()
    
    # Count predictions by word
    prediction_counts = db.session.query(
        Prediction.word,
        db.func.count(Prediction.id).label('count'),
        db.func.avg(Prediction.confidence).label('avg_confidence')
    ).filter_by(user_id=current_user.id).group_by(Prediction.word).all()
    
    # Total contributions and predictions
    total_contributions = DataContribution.query.filter_by(user_id=current_user.id).count()
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    
    return render_template(
        'dashboard.html', 
        contributions=contributions,
        predictions=predictions,
        contribution_counts=contribution_counts,
        prediction_counts=prediction_counts,
        total_contributions=total_contributions,
        total_predictions=total_predictions
    )

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('You do not have permission to access this page')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    datasets = Dataset.query.all()
    training_history = TrainingHistory.query.all()
    contributions = DataContribution.query.all()
    
    return render_template('admin.html', 
                          users=users, 
                          datasets=datasets, 
                          training_history=training_history,
                          contributions=contributions)

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('You do not have permission to access this page')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/datasets')
@login_required
def admin_datasets():
    if not current_user.is_admin:
        flash('You do not have permission to access this page')
        return redirect(url_for('dashboard'))
    
    datasets = Dataset.query.all()
    return render_template('admin_datasets.html', datasets=datasets)

@app.route('/admin/contributions')
@login_required
def admin_contributions():
    if not current_user.is_admin:
        flash('You do not have permission to access this page')
        return redirect(url_for('dashboard'))
    
    # Get all contributions with user information
    contributions = db.session.query(
        DataContribution, User.username
    ).join(User).order_by(DataContribution.created_at.desc()).all()
    
    # Get contribution statistics
    word_stats = db.session.query(
        DataContribution.word, 
        db.func.count(DataContribution.id).label('count')
    ).group_by(DataContribution.word).order_by(db.func.count(DataContribution.id).desc()).all()
    
    # Get user contribution statistics
    user_stats = db.session.query(
        User.username,
        db.func.count(DataContribution.id).label('count')
    ).join(User).group_by(User.username).order_by(db.func.count(DataContribution.id).desc()).all()
    
    return render_template(
        'admin_contributions.html', 
        contributions=contributions,
        word_stats=word_stats,
        user_stats=user_stats,
        total_contributions=len(contributions)
    )

@app.route('/admin/training')
@login_required
def admin_training():
    if not current_user.is_admin:
        flash('You do not have permission to access this page')
        return redirect(url_for('dashboard'))
    
    training_history = TrainingHistory.query.all()
    return render_template('admin_training.html', training_history=training_history)

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global spoken_already, frame_count, is_talking
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FACE_WEIGHTS_PATH)
        
        cap = cv2.VideoCapture(0)
        curr_word_frames = []
        not_talking_counter = 0
        past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)
        predicted_word = None
        prediction_display_counter = 0
        
        print("Starting video feed...")
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera")
                break
            else:
                try:
                    frame_count += 1
                    
                    # Ensure frame is in BGR format (OpenCV default)
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
                        print(f"Invalid frame format. Shape: {frame.shape}")
                        continue
                    
                    # Convert BGR to RGB for dlib
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Create grayscale image for face detection
                    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                    
                    # Ensure the image is contiguous
                    gray = np.ascontiguousarray(gray)
                    rgb_frame = np.ascontiguousarray(rgb_frame)
                    
                    # Detect faces
                    faces = detector(gray)
                    print(f"Frame {frame_count}: Detected {len(faces)} faces")
                    
                    for face in faces:
                        # Get face landmarks
                        landmarks = predictor(gray, face)
                        
                        # Calculate mouth opening
                        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
                        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
                        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], 
                                               mouth_bottom[1] - mouth_top[1])
                        
                        print(f"Frame {frame_count}: Lip distance = {lip_distance}")
                        
                        if lip_distance > 45:  # person is talking
                            if not is_talking:
                                print("Started talking")
                                is_talking = True
                            
                            cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 255, 0), 2)
                            
                            # Extract and process lip region
                            lip_frame = extract_lip_region(rgb_frame, landmarks)
                            
                            # Ensure lip frame is in the correct format
                            if lip_frame is not None and len(lip_frame.shape) == 3 and lip_frame.shape[2] == 3:
                                curr_word_frames.append(lip_frame)
                                print(f"Frame {frame_count}: Added frame to current word buffer. Total frames: {len(curr_word_frames)}")
                            else:
                                print(f"Invalid lip frame format. Shape: {lip_frame.shape if lip_frame is not None else None}")
                            
                            not_talking_counter = 0
                            prediction_display_counter = 0
                        else:
                            if is_talking:
                                print("Stopped talking")
                                is_talking = False
                            
                            cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 0, 255), 2)
                            not_talking_counter += 1
                            
                            # Check if we should make a prediction
                            if (not_talking_counter >= NOT_TALKING_THRESHOLD and 
                                len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES):
                                print("Making prediction...")
                                
                                # Prepare frames for prediction
                                frames_to_predict = list(past_word_frames) + curr_word_frames
                                frames_to_predict = frames_to_predict[:TOTAL_FRAMES]
                                curr_data = np.array([frames_to_predict])
                                
                                # Make prediction
                                prediction = model.predict(curr_data, verbose=1)
                                predicted_class_index = np.argmax(prediction)
                                confidence = prediction[0][predicted_class_index]
                                
                                # Find next best prediction if word already spoken
                                while LABEL_DICT[predicted_class_index] in spoken_already:
                                    prediction[0][predicted_class_index] = 0
                                    predicted_class_index = np.argmax(prediction)
                                    confidence = prediction[0][predicted_class_index]
                                
                                predicted_word = LABEL_DICT[predicted_class_index]
                                print(f"Predicted word: {predicted_word} (confidence: {confidence:.2f})")
                                
                                # Store prediction in database if user is logged in
                                if current_user.is_authenticated:
                                    prediction_record = Prediction(
                                        word=predicted_word,
                                        confidence=float(confidence),
                                        user_id=current_user.id
                                    )
                                    with app.app_context():
                                        db.session.add(prediction_record)
                                        db.session.commit()
                                
                                spoken_already.append(predicted_word)
                                prediction_display_counter = 20
                                
                                curr_word_frames = []
                                not_talking_counter = 0
                            
                            if len(curr_word_frames) > 0:
                                past_word_frames.append(curr_word_frames[-1])
                        
                        # Draw landmarks
                        for n in range(48, 61):
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                    
                    # Draw prediction and confidence on frame
                    if predicted_word and prediction_display_counter > 0:
                        # Draw word
                        cv2.putText(frame, predicted_word, (50, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                        # Draw confidence
                        confidence_text = f"Confidence: {confidence:.2f}"
                        cv2.putText(frame, confidence_text, (50, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        prediction_display_counter -= 1
                    
                    # Convert frame to JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    frame_bytes = jpeg.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    # If error occurs, try to display the original frame
                    try:
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except Exception as e:
                        print(f"Error encoding frame: {str(e)}")
                        continue
        
        print("Releasing camera...")
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def extract_lip_region(rgb_frame, landmarks):
    """Helper function to extract and process lip region"""
    # Get lip region coordinates
    lip_left = landmarks.part(48).x
    lip_right = landmarks.part(54).x
    lip_top = landmarks.part(50).y
    lip_bottom = landmarks.part(58).y
    
    # Add padding to match required dimensions
    width_diff = LIP_WIDTH - (lip_right - lip_left)
    height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
    pad_left = width_diff // 2
    pad_right = width_diff - pad_left
    pad_top = height_diff // 2
    pad_bottom = height_diff - pad_top
    
    # Ensure padding doesn't exceed image boundaries
    pad_left = min(pad_left, lip_left)
    pad_right = min(pad_right, rgb_frame.shape[1] - lip_right)
    pad_top = min(pad_top, lip_top)
    pad_bottom = min(pad_bottom, rgb_frame.shape[0] - lip_bottom)
    
    # Extract lip region
    lip_frame = rgb_frame[lip_top - pad_top:lip_bottom + pad_bottom, 
                         lip_left - pad_left:lip_right + pad_right]
    
    # Resize to exact dimensions needed by the model
    lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))
    
    return lip_frame

@app.route('/reset_predictions')
def reset_predictions():
    global spoken_already
    spoken_already = []
    return jsonify({'status': 'success'})

@app.route('/collect')
# @login_required  # Temporarily disabled for testing
def collect():
    # Test with new template to see if Flask picks up the changes
    return render_template('collect.html')

@app.route('/collect_original')
# @login_required  # Temporarily disabled for testing
def collect_original():
    # Keep original endpoint for comparison
    return render_template('collect.html')

@app.route('/api/collect', methods=['POST'])
# @login_required  # Temporarily disabled for testing
def collect_data():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    word = request.form.get('word')
    
    if video_file.filename == '' or not word:
        return jsonify({'error': 'Missing video or word label'}), 400
    
    # Create directory for the word if it doesn't exist
    word_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset', word)
    os.makedirs(word_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = int(time.time())
    user_id = current_user.id if current_user.is_authenticated else 1  # Use admin user (ID 1) for testing
    filename = f"{word}_{user_id}_{timestamp}.webm"
    video_path = os.path.join(word_dir, filename)
    
    # Save the uploaded video
    video_file.save(video_path)
    
    # Save contribution to database
    contribution = DataContribution(
        word=word,
        filename=filename,
        user_id=user_id
    )
    db.session.add(contribution)
    db.session.commit()
    
    # Log the contribution
    print(f"User {current_user.username} contributed a sample for '{word}'")
    
    return jsonify({
        'success': True,
        'message': f'Successfully saved sample for "{word}"',
        'filename': filename
    })

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Save the uploaded video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)
    
    # Process the video and make prediction
    try:
        # This is a placeholder for the actual prediction logic
        # In a real implementation, you would process the video frames
        # and use the model to make a prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Placeholder prediction
        prediction = {'word': 'hello', 'confidence': 0.95}
        
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/run_demo', methods=['POST'])
def run_demo():
    global demo_detector, demo_predictor, demo_model, demo_cap, demo_is_running
    
    try:
        if demo_is_running:
            return jsonify({'status': 'success', 'message': 'Demo is already running'})
        
        # Initialize the detector and predictor if not already done
        if demo_detector is None:
            demo_detector = dlib.get_frontal_face_detector()
        
        if demo_predictor is None:
            predictor_path = os.path.join(MODEL_DIR, 'face_weights.dat')
            demo_predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize the model if not already done
        if demo_model is None:
            # Define the input shape
            input_shape = (TOTAL_FRAMES, LIP_HEIGHT, LIP_WIDTH, 3)
            
            # Define the model architecture to match the one in predict_live.py
            # This is the exact architecture from predict_live.py
            demo_model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling3D((2, 2, 2)),
                tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
                tf.keras.layers.MaxPooling3D((2, 2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(LABEL_DICT), activation='softmax')
            ])
            
            # Load the model weights
            model_weights_path = os.path.join(MODEL_DIR, 'model_weights2.h5')  # Try the alternative weights file
            try:
                demo_model.load_weights(model_weights_path)
                print("Successfully loaded model weights from", model_weights_path)
            except Exception as e:
                print(f"Error loading model weights from {model_weights_path}: {e}")
                # Try the original weights file as fallback
                model_weights_path = os.path.join(MODEL_DIR, 'model_weights.h5')
                try:
                    demo_model.load_weights(model_weights_path)
                    print("Successfully loaded model weights from", model_weights_path)
                except Exception as e:
                    print(f"Error loading model weights from {model_weights_path}: {e}")
                    # Try the proof of concept weights as a last resort
                    model_weights_path = os.path.join(MODEL_DIR, 'proof_of_concept_model_weights.h5')
                    demo_model.load_weights(model_weights_path)
                    print("Successfully loaded model weights from", model_weights_path)
        
        # Initialize the camera if not already done
        if demo_cap is None:
            demo_cap = cv2.VideoCapture(0)
            if not demo_cap.isOpened():
                return jsonify({'status': 'error', 'error': 'Failed to open camera'})
        
        # Set the demo as running
        demo_is_running = True
        
        return jsonify({'status': 'success', 'message': 'Demo started successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/stop_demo', methods=['POST'])
def stop_demo():
    global demo_cap, demo_is_running
    
    try:
        if demo_cap is not None:
            demo_cap.release()
            demo_cap = None
        
        demo_is_running = False
        
        return jsonify({'status': 'success', 'message': 'Demo stopped successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/reset_demo', methods=['POST'])
def reset_demo():
    global demo_spoken_already, demo_curr_word_frames, demo_not_talking_counter
    
    try:
        demo_spoken_already = []
        demo_curr_word_frames = []
        demo_not_talking_counter = 0
        
        return jsonify({'status': 'success', 'message': 'Demo reset successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/toggle_multimodal', methods=['POST'])
def toggle_multimodal():
    # Remove this functionality or keep as a placeholder
    return jsonify({
        'status': 'success',
        'multimodal': False,
        'message': "Multimodal functionality has been removed"
    })

@app.route('/demo_feed')
def demo_feed():
    def generate():
        global demo_detector, demo_predictor, demo_model, demo_cap, demo_curr_word_frames
        global demo_not_talking_counter, demo_past_word_frames, demo_predicted_word
        global demo_prediction_display_counter, demo_spoken_already, demo_is_running, demo_count
        global demo_confidence, demo_new_prediction, demo_frame_skip, demo_is_processing
        
        current_frame = 0
        
        while True:
            if not demo_is_running or demo_cap is None:
                # Return a placeholder frame
                placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
                cv2.putText(placeholder, "Demo not running", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(placeholder, "Click 'Start Demo' to begin", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue
            
            success, frame = demo_cap.read()
            if not success:
                print("Failed to read frame from camera")
                demo_is_running = False
                continue
            
            # Skip frames to reduce processing load
            current_frame += 1
            if current_frame % demo_frame_skip != 0:
                # Just show the frame without processing
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue
            
            try:
                # Downsample the frame to speed up face detection
                scale_factor = 0.5
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                
                # Convert image into grayscale
                gray = cv2.cvtColor(src=small_frame, code=cv2.COLOR_BGR2GRAY)
                
                # Use detector to find landmarks
                faces = demo_detector(gray)
                
                if len(faces) == 0:
                    cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue
                
                for face in faces:
                    # Scale face back to original image size
                    orig_face = dlib.rectangle(
                        int(face.left() / scale_factor),
                        int(face.top() / scale_factor),
                        int(face.right() / scale_factor),
                        int(face.bottom() / scale_factor)
                    )
                    
                    # Get landmarks on original size image
                    landmarks = demo_predictor(image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), box=orig_face)
                    
                    # Calculate mouth opening - efficient calculation
                    mouth_top_y = landmarks.part(51).y
                    mouth_bottom_y = landmarks.part(57).y
                    lip_distance = mouth_bottom_y - mouth_top_y
                    
                    # Extract lip region - simplified
                    lip_left = landmarks.part(48).x
                    lip_right = landmarks.part(54).x
                    lip_top = landmarks.part(50).y
                    lip_bottom = landmarks.part(58).y
                    
                    # Add safety margin to prevent errors
                    margin_x = max(5, int((lip_right - lip_left) * 0.1))
                    margin_y = max(5, int((lip_bottom - lip_top) * 0.1))
                    
                    # Ensure within image bounds
                    safe_left = max(0, lip_left - margin_x)
                    safe_right = min(frame.shape[1], lip_right + margin_x)
                    safe_top = max(0, lip_top - margin_y)
                    safe_bottom = min(frame.shape[0], lip_bottom + margin_y)
                    
                    try:
                        # Extract region and resize directly to target size
                        lip_frame = frame[safe_top:safe_bottom, safe_left:safe_right]
                        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))
                        
                        # Simplified image enhancement - just apply contrast enhancement
                        lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
                        cl = clahe.apply(l)
                        enhanced = cv2.merge((cl, a, b))
                        lip_frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                        
                    except Exception as e:
                        print(f"Error processing lip frame: {e}")
                        continue
                    
                    # Draw only essential landmarks for speed
                    for n in range(48, 61, 3):  # Draw fewer points
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        cv2.circle(img=frame, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                    
                    if lip_distance > 35:  # Person is talking (reduced threshold for sensitivity)
                        cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        demo_curr_word_frames.append(lip_frame)
                        demo_not_talking_counter = 0
                    else:
                        cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        demo_not_talking_counter += 1
                        
                        # Store frames for future prediction
                        if len(demo_curr_word_frames) > 0:
                            demo_past_word_frames.append(demo_curr_word_frames[-1])
                        
                        # Check if we should make a prediction - do it less frequently
                        if (demo_not_talking_counter >= NOT_TALKING_THRESHOLD and 
                            len(demo_curr_word_frames) + len(demo_past_word_frames) >= TOTAL_FRAMES and
                            not demo_is_processing):
                            
                            # Start prediction in a thread
                            demo_is_processing = True
                            threading.Thread(
                                target=make_prediction_threaded,
                                args=(list(demo_past_word_frames) + demo_curr_word_frames,),
                                daemon=True
                            ).start()
                            
                            # Reset for next word
                            demo_curr_word_frames = []
                            demo_not_talking_counter = 0
                
                # Draw prediction on frame
                if demo_predicted_word and demo_prediction_display_counter > 0:
                    cv2.putText(frame, demo_predicted_word, (50, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                    
                    # Draw confidence
                    confidence_text = f"Confidence: {demo_confidence:.2f}"
                    cv2.putText(frame, confidence_text, (50, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    demo_prediction_display_counter -= 1
                
                # Convert frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                try:
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {str(e)}")
                    continue
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global demo_predicted_word, demo_prediction_display_counter, demo_confidence, demo_new_prediction
    
    if demo_predicted_word and (demo_prediction_display_counter > 0 or demo_new_prediction):
        # Reset the new prediction flag after sending it
        is_new = demo_new_prediction
        demo_new_prediction = False
        
        return jsonify({
            'word': demo_predicted_word,
            'confidence': float(demo_confidence),
            'display_counter': demo_prediction_display_counter,
            'is_new': is_new
        })
    else:
        return jsonify({
            'word': None,
            'confidence': 0.0,
            'display_counter': 0,
            'is_new': False
        })

def make_prediction_threaded(frames):
    """Run prediction in a separate thread to avoid blocking video processing"""
    global demo_predicted_word, demo_confidence, demo_new_prediction, demo_spoken_already, demo_prediction_display_counter, demo_is_processing
    
    try:
        # Prepare frames for prediction
        frames_to_predict = frames[:TOTAL_FRAMES]
        
        # Ensure we have enough frames
        while len(frames_to_predict) < TOTAL_FRAMES:
            frames_to_predict.append(frames_to_predict[-1])
        
        # Convert to numpy arrays if needed
        for i in range(len(frames_to_predict)):
            if isinstance(frames_to_predict[i], list):
                frames_to_predict[i] = np.array(frames_to_predict[i])
        
        # Prepare batch for model
        curr_data = np.array(frames_to_predict)
        
        # Fix shape if needed
        if curr_data.shape != (TOTAL_FRAMES, LIP_HEIGHT, LIP_WIDTH, 3):
            curr_data = curr_data.reshape((TOTAL_FRAMES, LIP_HEIGHT, LIP_WIDTH, 3))
        
        if len(curr_data.shape) == 4:
            curr_data = np.expand_dims(curr_data, axis=0)
        
        # Make visual prediction
        visual_prediction = demo_model.predict(curr_data, verbose=0)
        
        # Use the visual prediction directly
        final_prediction = visual_prediction[0]
        
        # Get the predicted class 
        predicted_class_index = np.argmax(final_prediction)
        confidence = final_prediction[predicted_class_index]
        
        # Skip words already spoken
        while LABEL_DICT[predicted_class_index] in demo_spoken_already:
            # Zero out this class and find next best
            final_prediction[predicted_class_index] = 0
            if np.max(final_prediction) == 0:
                break  # All words are used
            predicted_class_index = np.argmax(final_prediction)
            confidence = final_prediction[predicted_class_index]
        
        # Store the prediction results in global variables
        demo_predicted_word = LABEL_DICT[predicted_class_index]
        demo_confidence = float(confidence)
        demo_new_prediction = True
        demo_spoken_already.append(demo_predicted_word)
        demo_prediction_display_counter = 30  # Show prediction for longer
        
        print(f"Predicted word: {demo_predicted_word} (confidence: {confidence:.2f})")
    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo_is_processing = False

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin()
    app.run(debug=True) 