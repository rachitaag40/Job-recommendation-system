
# 🎯 SkillMatch AI - Intelligent Job Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![SQLite](https://img.shields.io/badge/SQLite-3.x-lightgrey.svg)
![Machine Learning](https://img.shields.io/badge/ML-TF--IDF-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

> An AI-powered job recommendation system that matches candidates with perfect job opportunities using Natural Language Processing and Machine Learning.



## ✨ Features

- 🤖 **AI-Powered Matching**: Uses TF-IDF vectorization and cosine similarity for intelligent job matching
- 📊 **Skill Gap Analysis**: Identifies missing skills for target job roles
- 💾 **Database Integration**: Stores users, jobs, and recommendation history
- 🎨 **Modern UI**: Clean, professional dashboard with real-time updates
- 📈 **Recommendation History**: Tracks all past recommendations per user
- ⚡ **Real-time Processing**: Instant job matching based on your skills

  

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)



## 📖 How It Works

### 1. **Enter Your Skills**
Type your technical skills separated by commas:

Python, Machine Learning, SQL, React, AWS


### 2. **Set Experience Level**
Input your years of experience (optional - only used for slight ranking adjustment)

### 3. **Get Recommendations**
The system:
- Preprocesses skills using NLP techniques
- Converts skills to TF-IDF vectors
- Calculates similarity with job requirements
- Ranks jobs by match percentage
- Returns top 5 best matches

### 4. **Analyze Skill Gap**
See what skills you're missing for your dream job and get actionable insights!




## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check system status |
| `/api/recommend` | POST | Get job recommendations |
| `/api/skill_gap` | POST | Analyze missing skills |
| `/api/users` | POST | Create user profile |
| `/api/users/<id>` | GET | Get user details |
| `/api/users/<id>/recommendations` | GET | Get user's saved recommendations |
| `/api/jobs` | GET | List all jobs |
| `/api/categories` | GET | List job categories |




## 🧠 Machine Learning Approach

### TF-IDF Vectorization
- Converts text skills into numerical vectors
- Considers skill importance across all jobs
- Handles multi-word skills (e.g., "Machine Learning")

### Cosine Similarity
- Measures similarity between user skills and job requirements
- Returns match percentage (0-100%)
- Experience only affects score minimally (max 10% penalty)

### Skills Preprocessing
- Expands abbreviations (ML → Machine Learning)
- Handles synonyms (React → ReactJS)
- Removes duplicates and normalizes text



## 📊 Database Schema

### Users Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary Key |
| name | String | User's full name |
| email | String | Unique email |
| skills | Text | Comma-separated skills |
| experience | Float | Years of experience |

### Jobs Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary Key |
| title | String | Job title |
| category | String | Job category |
| skills_required | Text | Required skills |
| description | Text | Job description |
| min_experience | Float | Minimum experience needed |

### Recommendations Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary Key |
| user_id | Integer | Foreign key to Users |
| job_id | Integer | Foreign key to Jobs |
| similarity_score | Float | Match score (0-1) |

## 🎨 UI Features

- **Dark/Light Professional Theme**: Clean, modern design
- **Real-time Date Display**: Shows current date
- **Interactive Stats Cards**: Visual metrics at a glance
- **Skill Tags**: Color-coded skill badges
- **Match Quality Indicators**: 
  - 🏆 Excellent (70%+)
  - 👍 Good (50-69%)
  - 📚 Fair (30-49%)
  - 🔍 Partial (<30%)


## 📈 Performance

- **Model Training**: < 2 seconds (30 jobs)
- **Recommendation Time**: < 100ms
- **Vocabulary Size**: ~500 skill terms
- **Jobs Database**: 30+ diverse roles

## 🚀 Future Enhancements

- [ ] Resume parsing (PDF/DOCX)
- [ ] Email notifications for new jobs
- [ ] User dashboard with saved searches
- [ ] Job application tracking
- [ ] Company profiles
- [ ] Salary insights
- [ ] Interview preparation resources


## 👨‍💻 Author

**Your Name**
- GitHub: [sonusg](https://github.com/sonusg)
- LinkedIn: [Sonu SG](www.linkedin.com/in/sonu-sg-14274425b)


⭐ Star this repo if you found it helpful!
</div>

