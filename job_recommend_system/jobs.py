# jobs.py
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobCreator:
    """Class to create job postings from freelancer dataset"""
    
    def __init__(self, csv_path: str = None):
        """Initialize with CSV file path"""
        if csv_path is None:
            self.base_dir = Path(__file__).parent
            self.csv_path = self.base_dir / 'data' / 'freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv'
        else:
            self.csv_path = Path(csv_path)
        
        self.df = None
        self.jobs = []
    
    def load_data(self) -> bool:
        """Load CSV data with error handling"""
        try:
            if not self.csv_path.exists():
                logger.error(f"CSV file not found: {self.csv_path}")
                return False
            
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"✓ Loaded {len(self.df)} freelancer records")
            return True
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False
    
    def extract_skills_by_level(self, category: str, experience_level: str = None) -> List[str]:
        """Extract skills based on experience level"""
        if experience_level:
            category_skills = self.df[(self.df['category'] == category) & 
                                      (self.df['experience_level'] == experience_level)]['primary_skills'].tolist()
        else:
            category_skills = self.df[self.df['category'] == category]['primary_skills'].tolist()
        
        all_skills = []
        for skills in category_skills:
            if pd.notna(skills):
                skill_list = [s.strip() for s in str(skills).split(',')]
                all_skills.extend(skill_list)
        
        # Count skill frequencies
        skill_freq = {}
        for skill in all_skills:
            skill_freq[skill] = skill_freq.get(skill, 0) + 1
        
        # Sort by frequency
        sorted_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top skills
        return [skill for skill, freq in sorted_skills]
    
    def calculate_avg_experience_by_level(self, category: str, level: str) -> float:
        """Calculate average experience for specific level"""
        exp_data = self.df[(self.df['category'] == category) & 
                          (self.df['experience_level'] == level)]['years_experience']
        if len(exp_data) > 0:
            return exp_data.mean()
        return 2.0
    
    def create_jobs(self) -> List[Dict]:
        """Create diverse job postings"""
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return []
        
        # Create jobs by experience level and skill combinations
        job_id = 1
        
        # Junior level jobs (0-2 years)
        junior_jobs = [
            {
                'title': 'Junior Python Developer',
                'category': 'Backend Development',
                'skills_required': 'Python, Django, Flask, SQL, Git, REST API',
                'description': 'Entry-level position for Python developers. Learn and grow with our team.',
                'min_experience': 0.5
            },
            {
                'title': 'Frontend Developer Trainee',
                'category': 'Web Development',
                'skills_required': 'HTML, CSS, JavaScript, React, Git',
                'description': 'Start your career in web development. Work with modern frameworks.',
                'min_experience': 0
            },
            {
                'title': 'Junior Data Analyst',
                'category': 'Data Engineering',
                'skills_required': 'Python, SQL, Pandas, Excel, Statistics',
                'description': 'Entry-level data analysis role. Learn data processing and visualization.',
                'min_experience': 0
            },
            {
                'title': 'UI Design Intern',
                'category': 'UI/UX Design',
                'skills_required': 'Figma, Adobe XD, Wireframing, Prototyping',
                'description': 'Learn UI/UX design principles. Work with senior designers.',
                'min_experience': 0
            },
            {
                'title': 'Junior DevOps Engineer',
                'category': 'DevOps',
                'skills_required': 'Linux, Bash, Docker, AWS Basics, Git',
                'description': 'Start your DevOps journey. Learn cloud and automation.',
                'min_experience': 0
            }
        ]
        
        for job in junior_jobs:
            job['id'] = job_id
            self.jobs.append(job)
            job_id += 1
        
        # Mid-level jobs (2-5 years)
        mid_jobs = [
            {
                'title': 'Full Stack Developer',
                'category': 'Full Stack',
                'skills_required': 'React, Node.js, MongoDB, Express, JavaScript, REST API',
                'description': 'Build end-to-end applications. Work on both frontend and backend.',
                'min_experience': 3.0
            },
            {
                'title': 'DevOps Engineer',
                'category': 'DevOps',
                'skills_required': 'Docker, Kubernetes, AWS, Jenkins, CI/CD, Linux',
                'description': 'Manage cloud infrastructure and deployment pipelines.',
                'min_experience': 3.0
            },
            {
                'title': 'Machine Learning Engineer',
                'category': 'AI/ML Engineering',
                'skills_required': 'Python, TensorFlow, Scikit-learn, Pandas, NumPy, SQL',
                'description': 'Build and deploy machine learning models.',
                'min_experience': 3.0
            },
            {
                'title': 'Mobile App Developer',
                'category': 'Mobile Development',
                'skills_required': 'React Native, JavaScript, Redux, iOS, Android, Firebase',
                'description': 'Develop cross-platform mobile applications.',
                'min_experience': 2.5
            },
            {
                'title': 'Security Analyst',
                'category': 'Cybersecurity',
                'skills_required': 'Network Security, Penetration Testing, Firewalls, SIEM, Python',
                'description': 'Monitor and protect organizational assets.',
                'min_experience': 3.0
            },
            {
                'title': 'Data Engineer',
                'category': 'Data Engineering',
                'skills_required': 'Python, Spark, SQL, Airflow, AWS, ETL',
                'description': 'Build and maintain data pipelines.',
                'min_experience': 3.0
            },
            {
                'title': 'Backend Developer',
                'category': 'Backend Development',
                'skills_required': 'Python, Django, PostgreSQL, REST API, Redis, Celery',
                'description': 'Build scalable backend services and APIs.',
                'min_experience': 2.5
            },
            {
                'title': 'Frontend Developer',
                'category': 'Web Development',
                'skills_required': 'React, Redux, TypeScript, HTML/CSS, Webpack, Jest',
                'description': 'Build responsive and interactive web applications.',
                'min_experience': 2.0
            }
        ]
        
        for job in mid_jobs:
            job['id'] = job_id
            self.jobs.append(job)
            job_id += 1
        
        # Senior level jobs (5+ years)
        senior_jobs = [
            {
                'title': 'Senior Backend Architect',
                'category': 'Backend Development',
                'skills_required': 'Java, Spring Boot, Microservices, Kafka, PostgreSQL, System Design',
                'description': 'Lead backend architecture decisions. Mentor junior developers.',
                'min_experience': 6.0
            },
            {
                'title': 'Lead Data Scientist',
                'category': 'AI/ML Engineering',
                'skills_required': 'Python, PyTorch, Deep Learning, NLP, Computer Vision, MLOps',
                'description': 'Lead AI/ML initiatives. Drive innovation in machine learning.',
                'min_experience': 6.0
            },
            {
                'title': 'Cloud Solutions Architect',
                'category': 'DevOps',
                'skills_required': 'AWS, Azure, GCP, Terraform, Kubernetes, Cloud Security',
                'description': 'Design cloud solutions. Optimize costs and performance.',
                'min_experience': 7.0
            },
            {
                'title': 'Senior UX Researcher',
                'category': 'UI/UX Design',
                'skills_required': 'User Research, Usability Testing, Data Analysis, Figma, Statistics',
                'description': 'Lead user research initiatives. Drive product decisions.',
                'min_experience': 5.0
            },
            {
                'title': 'Blockchain Architect',
                'category': 'Blockchain',
                'skills_required': 'Solidity, Ethereum, Web3, Smart Contracts, DeFi, Cryptography',
                'description': 'Design blockchain solutions. Lead development teams.',
                'min_experience': 5.0
            },
            {
                'title': 'Senior Frontend Lead',
                'category': 'Web Development',
                'skills_required': 'React, TypeScript, Next.js, GraphQL, Performance Optimization',
                'description': 'Lead frontend development. Set coding standards.',
                'min_experience': 6.0
            },
            {
                'title': 'Engineering Manager',
                'category': 'Full Stack',
                'skills_required': 'Leadership, Agile, System Design, Python, React, Team Management',
                'description': 'Lead engineering teams. Drive technical strategy.',
                'min_experience': 8.0
            }
        ]
        
        for job in senior_jobs:
            job['id'] = job_id
            self.jobs.append(job)
            job_id += 1
        
        # Add specialized roles
        specialized_jobs = [
            {
                'title': 'Computer Vision Engineer',
                'category': 'AI/ML Engineering',
                'skills_required': 'Python, OpenCV, TensorFlow, PyTorch, Image Processing',
                'description': 'Build computer vision applications. Work with images and video.',
                'min_experience': 3.0
            },
            {
                'title': 'NLP Engineer',
                'category': 'AI/ML Engineering',
                'skills_required': 'Python, Transformers, BERT, NLTK, spaCy, Deep Learning',
                'description': 'Work on natural language processing solutions.',
                'min_experience': 3.0
            },
            {
                'title': 'iOS Developer',
                'category': 'Mobile Development',
                'skills_required': 'Swift, iOS, UIKit, SwiftUI, Core Data, Xcode',
                'description': 'Build iOS applications for Apple devices.',
                'min_experience': 3.0
            },
            {
                'title': 'Android Developer',
                'category': 'Mobile Development',
                'skills_required': 'Kotlin, Android, Jetpack Compose, Room, Firebase',
                'description': 'Build Android applications for Google Play Store.',
                'min_experience': 3.0
            },
            {
                'title': 'Flutter Developer',
                'category': 'Mobile Development',
                'skills_required': 'Flutter, Dart, iOS, Android, Firebase, UI Design',
                'description': 'Build cross-platform mobile apps with Flutter.',
                'min_experience': 2.0
            },
            {
                'title': 'Vue.js Developer',
                'category': 'Web Development',
                'skills_required': 'Vue.js, JavaScript, Vuex, Vuetify, REST API',
                'description': 'Build reactive web applications with Vue.js.',
                'min_experience': 2.0
            },
            {
                'title': 'Angular Developer',
                'category': 'Web Development',
                'skills_required': 'Angular, TypeScript, RxJS, NgRx, HTML/CSS',
                'description': 'Build enterprise web applications with Angular.',
                'min_experience': 3.0
            },
            {
                'title': 'DevSecOps Engineer',
                'category': 'Cybersecurity',
                'skills_required': 'DevOps, Security, CI/CD, Docker, Kubernetes, AWS',
                'description': 'Integrate security into DevOps pipelines.',
                'min_experience': 4.0
            },
            {
                'title': 'Database Administrator',
                'category': 'Data Engineering',
                'skills_required': 'SQL, PostgreSQL, MySQL, MongoDB, Database Optimization, Backup',
                'description': 'Manage and optimize database systems.',
                'min_experience': 3.0
            },
            {
                'title': 'QA Automation Engineer',
                'category': 'Web Development',
                'skills_required': 'Selenium, Python, Jenkins, Test Automation, CI/CD, API Testing',
                'description': 'Build and maintain test automation frameworks.',
                'min_experience': 2.0
            }
        ]
        
        for job in specialized_jobs:
            job['id'] = job_id
            self.jobs.append(job)
            job_id += 1
        
        logger.info(f"✓ Created {len(self.jobs)} diverse job postings")
        return self.jobs
    
    def save_to_csv(self, output_path: str = None):
        """Save jobs to CSV file"""
        if not self.jobs:
            logger.warning("No jobs to save")
            return False
        
        try:
            if output_path is None:
                # Create data directory if it doesn't exist
                data_dir = self.base_dir / 'data'
                data_dir.mkdir(exist_ok=True)
                output_path = data_dir / 'jobs.csv'
            
            jobs_df = pd.DataFrame(self.jobs)
            jobs_df.to_csv(output_path, index=False)
            logger.info(f"✓ Jobs saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")
            return False
    
    def get_jobs(self) -> List[Dict]:
        """Return created jobs"""
        return self.jobs

def create_jobs_from_dataset():
    """Convenience function to create jobs"""
    creator = JobCreator()
    if creator.load_data():
        jobs = creator.create_jobs()
        creator.save_to_csv()
        return jobs
    return []