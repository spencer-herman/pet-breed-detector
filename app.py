from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import os
from PIL import Image, ImageDraw, ImageFont
import io
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from flask_migrate import Migrate  # Added for migrations

# Initialize the Flask app
app = Flask(__name__)

# Load the .env file
load_dotenv()

# Secret key and the upload folder for saving user images
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Database URL for the Heroku postgres database or local SQLite database
database_url = os.getenv('DATABASE_URL', 'sqlite:///site.db')
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# If there is no SECRET_KEY, raise an error
if not app.config['SECRET_KEY']:
    raise ValueError("Please set the SECRET_KEY environment variable")

# Get the API key from the environment variable
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("Please set the ROBOFLOW_API_KEY environment variable")

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create the forms with Flask-WTF
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Route for the home page (if user is not logged in it will redirect to login page)
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Login failed. Check email and password.', 'danger')
    return render_template('login.html', form=form)

# Route for user logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Route for uploading and processing images
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)

        file = request.files['image']
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_image_path)

        result = CLIENT.infer(temp_image_path, model_id='dog_breed_detector/1')
        image = Image.open(temp_image_path)
        draw = ImageDraw.Draw(image)
        
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.load_default()
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 48)

        predictions = result['predictions']
        if not predictions:
            os.remove(temp_image_path)
            flash('No detections found.', 'danger')
            return redirect(request.url)

        for pred in predictions:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            label = pred['class']
            confidence = pred['confidence']
            box = [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)]
            draw.rectangle(box, outline="red", width=3)
            text = f"{label}: {confidence:.2f}"
            text_position = (x - width / 2, y - height / 2 - 60)
            if text_position[1] < 0:
                text_position = (x - width / 2, y + height / 2 + 10)
            draw.text(text_position, text, fill="red", font=font)

        result_img_io = io.BytesIO()
        image.save(result_img_io, 'JPEG')
        result_img_io.seek(0)
        os.remove(temp_image_path)
        return send_file(result_img_io, mimetype='image/jpeg')

    return render_template('upload_form.html')

if __name__ == '__main__':
    app.run(debug=True)