# load_data.py
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
from database import db, Job
from jobs import JobCreator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handle loading jobs into database"""
    
    def __init__(self, db_session):
        self.db_session = db_session
        self.base_dir = Path(__file__).parent
    
    def load_from_csv(self, csv_path: str = None) -> List[Dict]:
        """Load jobs from CSV file"""
        if csv_path is None:
            csv_path = self.base_dir / 'data' / 'jobs.csv'
        
        try:
            if not Path(csv_path).exists():
                logger.warning(f"CSV file not found: {csv_path}")
                return None
            
            jobs_df = pd.read_csv(csv_path)
            jobs = jobs_df.to_dict('records')
            logger.info(f"Loaded {len(jobs)} jobs from CSV")
            return jobs
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return None
    
    def create_new_jobs(self) -> List[Dict]:
        """Create new jobs from dataset"""
        creator = JobCreator()
        if creator.load_data():
            jobs = creator.create_jobs()
            creator.save_to_csv()
            return jobs
        return []
    
    def clear_existing_jobs(self):
        """Clear existing jobs from database"""
        try:
            count = self.db_session.query(Job).delete()
            logger.info(f"Cleared {count} existing jobs")
            return True
        except Exception as e:
            logger.error(f"Failed to clear jobs: {e}")
            self.db_session.rollback()
            return False
    
    def insert_jobs(self, jobs: List[Dict]) -> bool:
        """Insert jobs into database"""
        if not jobs:
            logger.warning("No jobs to insert")
            return False
        
        try:
            for job_data in jobs:
                # Check if job already exists (by title and category)
                existing = self.db_session.query(Job).filter_by(
                    title=job_data['title'],
                    category=job_data.get('category', 'General')
                ).first()
                
                if existing:
                    logger.debug(f"Job already exists: {job_data['title']}")
                    continue
                
                job = Job(
                    title=job_data['title'],
                    category=job_data.get('category', 'General'),
                    skills_required=job_data['skills_required'],
                    description=job_data.get('description', ''),
                    min_experience=float(job_data.get('min_experience', 0))
                )
                self.db_session.add(job)
            
            self.db_session.commit()
            logger.info(f"✓ Inserted {len(jobs)} jobs into database")
            return True
        except Exception as e:
            logger.error(f"Failed to insert jobs: {e}")
            self.db_session.rollback()
            return False
    
    def load_jobs_to_database(self, force_reload: bool = False) -> bool:
        """Main method to load jobs into database"""
        try:
            # Try to load from CSV first
            jobs = self.load_from_csv()
            
            # If CSV doesn't exist or force reload, create new jobs
            if jobs is None or force_reload:
                logger.info("Creating new jobs from dataset...")
                jobs = self.create_new_jobs()
                if not jobs:
                    logger.error("Failed to create jobs")
                    return False
            
            # Clear existing jobs if force reload
            if force_reload:
                self.clear_existing_jobs()
            
            # Insert jobs
            return self.insert_jobs(jobs)
            
        except Exception as e:
            logger.error(f"Error loading jobs: {e}")
            self.db_session.rollback()
            return False

def load_jobs_to_database(db_session, force_reload: bool = False) -> bool:
    """Convenience function to load jobs"""
    loader = DataLoader(db_session)
    return loader.load_jobs_to_database(force_reload)