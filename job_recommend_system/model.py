# model.py
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import joblib
import re
from functools import lru_cache
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobRecommender:
    """Job recommendation system using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.job_vectors = None
        self.jobs_df = None
        self.is_fitted = False
        self.model_path = Path(__file__).parent / 'models' / 'job_recommender.pkl'
        self.cache = {}
    
    def preprocess_skills(self, skills_text: str) -> str:
        """Enhanced preprocessing with better skill matching"""
        if pd.isna(skills_text) or not skills_text:
            return ""
        
        skills_text = str(skills_text)
        
        # Common skill variations mapping
        skill_variations = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence', 
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning',
            'k8s': 'kubernetes',
            'ci/cd': 'ci cd',
            'api': 'api development',
            'ui/ux': 'ui ux design',
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'react': 'reactjs',
            'node': 'nodejs',
            'sql': 'sql database',
            'nosql': 'nosql database',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'azure': 'microsoft azure'
        }
        
        skills_lower = skills_text.lower()
        
        # Replace variations
        for abbr, full in skill_variations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            skills_lower = re.sub(pattern, full, skills_lower)
        
        # Split by commas and clean
        skills_list = [s.strip() for s in skills_lower.split(',')]
        
        # Remove empty and very short skills
        skills_list = [s for s in skills_list if len(s) > 1]
        
        # Expand skill synonyms
        expanded_skills = []
        synonym_map = {
            'python': ['python', 'python programming', 'python development'],
            'javascript': ['javascript', 'js', 'javascript development'],
            'react': ['react', 'reactjs', 'react.js'],
            'angular': ['angular', 'angularjs', 'angular.js'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'node': ['node', 'nodejs', 'node.js'],
            'docker': ['docker', 'containerization', 'docker containers'],
            'kubernetes': ['kubernetes', 'k8s', 'container orchestration'],
            'aws': ['aws', 'amazon web services', 'ec2', 's3'],
            'tensorflow': ['tensorflow', 'tf', 'tensor flow'],
            'pytorch': ['pytorch', 'torch', 'py torch'],
            'machine learning': ['machine learning', 'ml', 'machine learning algorithms'],
            'data science': ['data science', 'data analysis', 'data analytics'],
            'sql': ['sql', 'structured query language', 'database queries'],
            'mongodb': ['mongodb', 'mongo', 'nosql'],
            'postgresql': ['postgresql', 'postgres', 'relational database']
        }
        
        for skill in skills_list:
            # Check if skill has synonyms
            found = False
            for key, synonyms in synonym_map.items():
                if skill in synonyms or skill in key or key in skill:
                    expanded_skills.extend(synonyms)
                    found = True
                    break
            
            if not found:
                # Add skill as is and add plural form if applicable
                expanded_skills.append(skill)
                if not skill.endswith('ing') and not skill.endswith('er'):
                    expanded_skills.append(skill + 'ing')
                    expanded_skills.append(skill + 'er')
        
        # Remove duplicates while preserving order
        unique_skills = []
        for skill in expanded_skills:
            if skill not in unique_skills:
                unique_skills.append(skill)
        
        # Join back with spaces
        processed_text = ' '.join(unique_skills)
        processed_text = ' '.join(processed_text.split())
        
        return processed_text
    
    def validate_jobs_data(self, jobs_data: List[Dict]) -> bool:
        """Validate jobs data before fitting"""
        if not jobs_data:
            logger.error("No jobs data provided")
            return False
        
        required_fields = ['skills_required', 'title']
        for job in jobs_data:
            for field in required_fields:
                if field not in job:
                    logger.error(f"Missing required field: {field} in job: {job.get('title', 'Unknown')}")
                    return False
        
        logger.info(f"✓ Validated {len(jobs_data)} jobs")
        return True
    
    def fit(self, jobs_data: List[Dict]) -> bool:
        """Fit the model on job data with improved parameters"""
        try:
            if not self.validate_jobs_data(jobs_data):
                return False
            
            self.jobs_df = pd.DataFrame(jobs_data)
            
            # Preprocess job skills
            self.jobs_df['processed_skills'] = self.jobs_df['skills_required'].apply(self.preprocess_skills)
            
            # Add title and category to skills for better context
            self.jobs_df['combined_text'] = self.jobs_df.apply(
                lambda row: f"{row['processed_skills']} {row['title']} {row.get('category', '')}", 
                axis=1
            )
            
            # Check for empty skills
            empty_skills = self.jobs_df[self.jobs_df['combined_text'] == '']
            if not empty_skills.empty:
                logger.warning(f"Found {len(empty_skills)} jobs with empty text")
                self.jobs_df = self.jobs_df[self.jobs_df['combined_text'] != '']
            
            if len(self.jobs_df) == 0:
                logger.error("No valid jobs after preprocessing")
                return False
            
            # Improved TF-IDF parameters
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=False,
                token_pattern=r'(?u)\b\w[\w-]+\b',
                stop_words='english',
                ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
                max_features=5000,
                min_df=1,  # Include all skills
                max_df=0.95,  # Only ignore skills in 95% of jobs
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True
            )
            
            # Fit and transform
            self.job_vectors = self.tfidf_vectorizer.fit_transform(self.jobs_df['combined_text'])
            self.is_fitted = True
            
            logger.info(f"✓ Model fitted with {len(self.jobs_df)} jobs")
            logger.info(f"✓ TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            self.is_fitted = False
            return False
    
    def get_recommendations(self, user_skills: str, top_n: int = 5, 
                           min_experience: Optional[float] = None,
                           use_cache: bool = True) -> List[Dict]:
        """Get job recommendations based on skills - experience is secondary"""
        
        cache_key = self._get_cache_key(user_skills, top_n, min_experience)
        if use_cache and cache_key in self.cache:
            logger.info("Returning cached recommendations")
            return self.cache[cache_key]
        
        if not self.is_fitted:
            logger.error("Model not fitted. Call fit() first.")
            return []
        
        try:
            if not user_skills or not user_skills.strip():
                logger.warning("Empty user skills provided")
                return []
            
            # Preprocess user skills
            processed_user_skills = self.preprocess_skills(user_skills)
            
            if not processed_user_skills:
                logger.warning("No valid skills after preprocessing")
                return []
            
            # Transform user skills
            user_vector = self.tfidf_vectorizer.transform([processed_user_skills])
            
            # Calculate cosine similarity - PURE SKILLS MATCH
            similarity_scores = cosine_similarity(user_vector, self.job_vectors).flatten()
            
            # Create results DataFrame
            results = self.jobs_df.copy()
            results['similarity_score'] = similarity_scores
            
            # Experience filter - ONLY AS A SECONDARY FACTOR
            if min_experience is not None and 'min_experience' in results.columns:
                # Don't filter out jobs, just add a penalty for experience mismatch
                # Jobs requiring more experience get a SMALL penalty (max 10% reduction)
                results['experience_penalty'] = results['min_experience'].apply(
                    lambda x: max(0, (x - min_experience) * 0.02) if min_experience < x else 0
                )
                # Apply penalty (max 10% reduction)
                results['similarity_score'] = results['similarity_score'] * (1 - results['experience_penalty'])
            
            # Sort by similarity score (skills match first, experience penalty applied)
            results = results.sort_values('similarity_score', ascending=False)
            
            # Get top N results - show ALL matches, no minimum threshold
            results = results.head(top_n)
            
            # Prepare output
            recommendations = []
            for _, row in results.iterrows():
                # Calculate final score (skills match minus any experience penalty)
                final_score = row['similarity_score']
                
                recommendations.append({
                    'job_id': row.get('id', row.name),
                    'title': row['title'],
                    'category': row.get('category', 'General'),
                    'skills_required': row['skills_required'],
                    'description': row.get('description', 'No description available'),
                    'similarity_score': round(final_score * 100, 2),
                    'min_experience': row.get('min_experience', 0),
                    'match_quality': self._get_match_quality(final_score)
                })
            
            logger.info(f"✓ Found {len(recommendations)} recommendations based on skills")
            
            # Cache results
            if use_cache and recommendations:
                self.cache[cache_key] = recommendations
                if len(self.cache) > 100:
                    keys_to_remove = list(self.cache.keys())[:20]
                    for key in keys_to_remove:
                        del self.cache[key]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _get_match_quality(self, score: float) -> str:
        """Determine match quality based on score"""
        if score >= 0.7:
            return "Excellent Match"
        elif score >= 0.5:
            return "Good Match"
        elif score >= 0.3:
            return "Fair Match"
        else:
            return "Partial Match"
    
    def _get_cache_key(self, user_skills: str, top_n: int, min_experience: Optional[float]) -> str:
        """Generate cache key for recommendations"""
        key_str = f"{user_skills}_{top_n}_{min_experience}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def analyze_skill_gap(self, user_skills: str, top_n: int = 3) -> Dict:
        """Analyze missing skills for top job matches"""
        recommendations = self.get_recommendations(user_skills, top_n=top_n, use_cache=False)
        
        if not recommendations:
            return {'error': 'No recommendations found'}
        
        user_skill_set = set([s.strip().lower() for s in user_skills.split(',')])
        
        analysis = []
        for rec in recommendations:
            job_skills = [s.strip().lower() for s in rec['skills_required'].split(',')]
            missing_skills = [s for s in job_skills if s not in user_skill_set]
            
            analysis.append({
                'job_title': rec['title'],
                'match_percentage': rec['similarity_score'],
                'missing_skills': missing_skills[:5],
                'total_missing': len(missing_skills),
                'required_skills': rec['skills_required']  # Added for display
            })
        
        return {'skill_gap_analysis': analysis}
    
    def save_model(self, path: str = None):
        """Save trained model to disk"""
        if not self.is_fitted:
            logger.warning("Model not fitted. Cannot save.")
            return False
        
        try:
            if path is None:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                path = self.model_path
            
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'job_vectors': self.job_vectors,
                'jobs_df': self.jobs_df,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(model_data, path)
            logger.info(f"✓ Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str = None):
        """Load trained model from disk"""
        try:
            if path is None:
                path = self.model_path
            
            if not Path(path).exists():
                logger.warning(f"Model file not found: {path}")
                return False
            
            model_data = joblib.load(path)
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.job_vectors = model_data['job_vectors']
            self.jobs_df = model_data['jobs_df']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"✓ Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False