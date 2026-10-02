# app.py
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
import logging
import sys
import threading
import os

# Fix for Windows console encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{BASE_DIR}/job_recommendation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Initialize database
from database import db, User, Job, Recommendation
db.init_app(app)

# Import other modules
from model import JobRecommender
from load_data import load_jobs_to_database

# Global variables for initialization status
recommender = None
init_status = {
    'initialized': False,
    'database_initialized': False,
    'jobs_loaded': False,
    'model_training': False,
    'error': None,
    'progress': 0
}

def init_database():
    """Initialize database tables"""
    try:
        with app.app_context():
            db.create_all()
            init_status['database_initialized'] = True
            init_status['progress'] = 25
            logger.info("Database tables created/verified")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        init_status['error'] = str(e)
        return False

def load_jobs():
    """Load jobs into database"""
    try:
        with app.app_context():
            # Check if jobs exist
            job_count = Job.query.count()
            if job_count > 0:
                logger.info(f"Found {job_count} existing jobs")
                init_status['jobs_loaded'] = True
                init_status['progress'] = 50
                return True
            
            # Load jobs
            logger.info("Loading jobs into database...")
            success = load_jobs_to_database(db.session)
            
            if success:
                init_status['jobs_loaded'] = True
                init_status['progress'] = 50
                logger.info("Jobs loaded successfully")
                return True
            else:
                init_status['error'] = "Failed to load jobs"
                return False
    except Exception as e:
        logger.error(f"Failed to load jobs: {e}")
        init_status['error'] = str(e)
        return False

def train_model():
    """Train recommendation model with persistence"""
    try:
        with app.app_context():
            global recommender
            
            # Initialize recommender
            recommender = JobRecommender()
            
            # Try to load existing model first
            if recommender.load_model():
                logger.info("Model loaded from disk successfully")
                init_status['initialized'] = True
                init_status['progress'] = 100
                init_status['model_training'] = False
                return True
            
            # If no model exists, train new one
            init_status['model_training'] = True
            init_status['progress'] = 75
            
            # Get all jobs
            jobs = Job.query.all()
            
            if not jobs:
                logger.error("No jobs found for training")
                init_status['error'] = "No jobs available"
                init_status['model_training'] = False
                return False
            
            # Prepare jobs data
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    'id': job.id,
                    'title': job.title,
                    'skills_required': job.skills_required,
                    'min_experience': job.min_experience,
                    'category': job.category,
                    'description': job.description
                })
            
            logger.info(f"Preparing to train model with {len(jobs_data)} jobs...")
            
            # Train model
            success = recommender.fit(jobs_data)
            
            if success:
                # Save the trained model
                recommender.save_model()
                
                init_status['progress'] = 100
                init_status['initialized'] = True
                init_status['model_training'] = False
                logger.info(f"Model trained and saved with {len(jobs_data)} jobs")
                return True
            else:
                init_status['error'] = "Model training failed"
                init_status['model_training'] = False
                return False
                
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        init_status['error'] = str(e)
        init_status['model_training'] = False
        return False

def initialize_app():
    """Initialize the entire application in background"""
    logger.info("Starting application initialization...")
    
    # Step 1: Initialize database
    if not init_database():
        logger.error("Database initialization failed")
        return
    
    # Step 2: Load jobs
    if not load_jobs():
        logger.error("Job loading failed")
        return
    
    # Step 3: Train or load model
    if not train_model():
        logger.error("Model training/loading failed")
        return
    
    logger.info("Application initialization complete!")

# Start initialization in background thread
init_thread = threading.Thread(target=initialize_app, daemon=True)
init_thread.start()

@app.route('/')
def index():
    """Home page - redirect to loading if not ready"""
    if not init_status['initialized']:
        return render_template('loading.html')
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    status = {
        'status': 'healthy' if init_status['initialized'] else 'initializing',
        'initialized': init_status['initialized'],
        'database_initialized': init_status['database_initialized'],
        'jobs_loaded': init_status['jobs_loaded'],
        'model_training': init_status['model_training'],
        'progress': init_status['progress'],
        'model_initialized': recommender is not None and recommender.is_fitted if recommender else False
    }
    
    # Add total jobs if available
    if recommender and recommender.jobs_df is not None:
        status['total_jobs'] = len(recommender.jobs_df)
    
    if init_status['error']:
        status['error'] = init_status['error']
        status['status'] = 'error'
    
    return jsonify(status)

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get job recommendations based on user skills"""
    # Check if system is ready
    if not init_status['initialized']:
        return jsonify({'error': 'System is still initializing. Please wait a moment and try again.'}), 503
    
    if not recommender or not recommender.is_fitted:
        return jsonify({'error': 'Model not ready. Please wait.'}), 503
    
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_skills = data.get('skills', '').strip()
        min_experience = data.get('min_experience')
        user_id = data.get('user_id')
        
        # Validate user skills
        if not user_skills:
            return jsonify({'error': 'Please enter your skills'}), 400
        
        # Validate experience
        if min_experience is not None:
            try:
                min_experience = float(min_experience)
                if min_experience < 0:
                    return jsonify({'error': 'Experience cannot be negative'}), 400
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid experience value'}), 400
        
        logger.info(f"Processing recommendation request - Skills: {user_skills[:50]}...")
        
        # Get recommendations - increased top_n to 10 for more results
        recommendations = recommender.get_recommendations(
            user_skills=user_skills,
            top_n=10,  # Changed from 5 to 10
            min_experience=min_experience
        )
        
        # Save to database if user_id provided
        if user_id and recommendations:
            try:
                with app.app_context():
                    user = User.query.get(user_id)
                    if user:
                        for rec in recommendations:
                            # Check if recommendation already exists
                            existing = Recommendation.query.filter_by(
                                user_id=user_id,
                                job_id=rec['job_id']
                            ).first()
                            
                            if not existing:
                                recommendation = Recommendation(
                                    user_id=user_id,
                                    job_id=rec['job_id'],
                                    similarity_score=rec['similarity_score'] / 100
                                )
                                db.session.add(recommendation)
                        
                        db.session.commit()
                        logger.info(f"Saved {len(recommendations)} recommendations for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to save recommendations: {e}")
                db.session.rollback()
        
        # Return response with helpful message if no recommendations
        if not recommendations:
            return jsonify({
                'user_skills': user_skills,
                'recommendations': [],
                'message': 'No matching jobs found. Try different skills or reduce experience requirement.'
            })
        
        return jsonify({
            'user_skills': user_skills,
            'recommendations': recommendations,
            'total_found': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/skill_gap', methods=['POST'])
def analyze_skill_gap():
    """Analyze skill gap for user - shows missing skills for top jobs"""
    if not init_status['initialized']:
        return jsonify({'error': 'System initializing'}), 503
    
    if not recommender or not recommender.is_fitted:
        return jsonify({'error': 'Model not ready'}), 503
    
    try:
        data = request.json
        user_skills = data.get('skills', '').strip()
        
        if not user_skills:
            return jsonify({'error': 'Please enter your skills'}), 400
        
        # Get skill gap analysis from recommender
        analysis = recommender.analyze_skill_gap(user_skills, top_n=3)
        
        # Add the required skills to the response for display
        if analysis and 'skill_gap_analysis' in analysis:
            for job_analysis in analysis['skill_gap_analysis']:
                # Get the full job details to include required skills
                recommendations = recommender.get_recommendations(user_skills, top_n=1, use_cache=False)
                for rec in recommendations:
                    if rec['title'] == job_analysis['job_title']:
                        job_analysis['required_skills'] = rec['skills_required']
                        break
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in skill gap analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user"""
    if not init_status['initialized']:
        return jsonify({'error': 'System initializing. Please wait.'}), 503
    
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['name', 'email', 'skills', 'experience']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing or empty field: {field}'}), 400
        
        # Validate experience
        try:
            experience = float(data['experience'])
            if experience < 0:
                return jsonify({'error': 'Experience cannot be negative'}), 400
            if experience > 50:
                return jsonify({'error': 'Experience cannot exceed 50 years'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid experience value'}), 400
        
        # Validate email format (basic)
        if '@' not in data['email'] or '.' not in data['email']:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists
        with app.app_context():
            existing_user = User.query.filter_by(email=data['email']).first()
            if existing_user:
                return jsonify({'error': 'User with this email already exists'}), 400
            
            # Create new user
            user = User(
                name=data['name'][:100],  # Limit length
                email=data['email'][:100],
                skills=data['skills'][:1000],  # Limit length
                experience=experience
            )
            
            db.session.add(user)
            db.session.commit()
            
            logger.info(f"Created new user: {user.name} ({user.email})")
            
            return jsonify({
                'message': 'User created successfully',
                'user': user.to_dict()
            }), 201
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID"""
    if not init_status['initialized']:
        return jsonify({'error': 'System initializing'}), 503
    
    try:
        with app.app_context():
            user = User.query.get(user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            return jsonify(user.to_dict())
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/users/<int:user_id>/recommendations', methods=['GET'])
def get_user_recommendations(user_id):
    """Get saved recommendations for a user"""
    if not init_status['initialized']:
        return jsonify({'error': 'System initializing'}), 503
    
    try:
        with app.app_context():
            recommendations = Recommendation.query.filter_by(user_id=user_id)\
                .order_by(Recommendation.similarity_score.desc())\
                .limit(20)\
                .all()
            
            return jsonify([rec.to_dict() for rec in recommendations])
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all jobs with optional filtering"""
    if not init_status['initialized']:
        return jsonify({'error': 'System initializing'}), 503
    
    try:
        category = request.args.get('category')
        min_experience = request.args.get('min_experience')
        limit = request.args.get('limit', default=50, type=int)
        
        with app.app_context():
            query = Job.query
            
            if category:
                query = query.filter_by(category=category)
            
            if min_experience:
                try:
                    min_exp = float(min_experience)
                    query = query.filter(Job.min_experience <= min_exp)
                except ValueError:
                    pass
            
            jobs = query.limit(limit).all()
            return jsonify([job.to_dict() for job in jobs])
        
    except Exception as e:
        logger.error(f"Error fetching jobs: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all unique job categories"""
    if not init_status['initialized']:
        return jsonify({'error': 'System initializing'}), 503
    
    try:
        with app.app_context():
            categories = db.session.query(Job.category).distinct().all()
            categories = [c[0] for c in categories if c[0]]
            return jsonify({'categories': categories})
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("="*50)
    logger.info("AI Job Recommendation System Starting...")
    logger.info("="*50)
    logger.info(f"Project directory: {BASE_DIR}")
    logger.info(f"Database path: {app.config['SQLALCHEMY_DATABASE_URI']}")
    logger.info("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)