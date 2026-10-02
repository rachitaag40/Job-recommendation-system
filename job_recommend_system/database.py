# database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging

# Initialize SQLAlchemy without app first (will be initialized in app.py)
db = SQLAlchemy()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    skills = db.Column(db.Text, nullable=False)
    experience = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    recommendations = db.relationship('Recommendation', backref='user', lazy=True, cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        db.Index('idx_user_email', 'email'),
        db.Index('idx_user_experience', 'experience'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'skills': self.skills,
            'experience': self.experience,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<User {self.name}>'

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False, index=True)
    category = db.Column(db.String(100), index=True)
    skills_required = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)
    min_experience = db.Column(db.Float, default=0, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    recommendations = db.relationship('Recommendation', backref='job', lazy=True, cascade='all, delete-orphan')
    
    # Indexes for faster queries
    __table_args__ = (
        db.Index('idx_job_category', 'category'),
        db.Index('idx_job_experience', 'min_experience'),
        db.Index('idx_job_title', 'title'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'category': self.category,
            'skills_required': self.skills_required,
            'description': self.description,
            'min_experience': self.min_experience
        }
    
    def __repr__(self):
        return f'<Job {self.title}>'

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False, index=True)
    similarity_score = db.Column(db.Float, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Composite indexes for common query patterns
    __table_args__ = (
        db.Index('idx_user_job', 'user_id', 'job_id', unique=True),
        db.Index('idx_user_score', 'user_id', 'similarity_score'),
        db.Index('idx_job_score', 'job_id', 'similarity_score'),
        db.Index('idx_created', 'created_at'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'job_id': self.job_id,
            'similarity_score': self.similarity_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'job': self.job.to_dict() if self.job else None
        }
    
    def __repr__(self):
        return f'<Recommendation User:{self.user_id} Job:{self.job_id}>'