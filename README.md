# Leveraging NLP for HR Recruitment

This project leverages modern Natural Language Processing (NLP) techniques to automate and enhance the candidate filtering process for a Human Resources recruitment agency. Traditionally, filtering ideal candidates is a labor-intensive task. Here, we attempt to build a system that recommends the best-suited candidates for specific HR roles using various NLP methods and a learn-to-rank model.

---

## 📚 Table of Contents

- [Project Overview](#-project-overview)
- [Motivation](#motivation)
- [Data Description](#data-description)
- [Methodology](#-methodology)
- [Technologies & Libraries Used](#-technologies--libraries-used)
- [Results](#-results)
- [Future Work](#-future-work)
- [License](#-license)

---

## 📘 Project Overview

The project is structured into three major sections:

1. **Exploratory Data Analysis (EDA) and Data Cleaning**  
   Load and inspect candidate data, handle missing values, and prepare the dataset for NLP processing.

2. **NLP Vectorization Techniques for Candidate Matching**  
   Test and evaluate the effectiveness of different vectorization methods to match candidates with job-related keywords like `"Aspiring HR Professional"`:
   - TF-IDF
   - Word2Vec
   - GloVe
   - FastText
   - BERT (Transformer-based encoder)

3. **Pairwise Learn-to-Rank Model**  
   Build a neural network model using pairwise learning-to-rank to automatically learn candidate rankings based on keyword fit, using earlier similarity scores as input features.

---

## Motivation

Recruitment agencies often face the daunting challenge of filtering through hundreds or thousands of resumes to find a handful of qualified candidates for specific roles. This process is not only time-consuming but also prone to human bias and inconsistency.

The goal of this project is to explore whether modern NLP methods can intelligently match candidates to job descriptions based on semantic relevance, thereby automating the first stages of candidate screening. A successful system would help HR teams reduce manual effort, ensure consistency in evaluation, and highlight strong candidates who may otherwise be overlooked.

---

## Data Description

The dataset used in this project contains 104 candidate records. Each record includes various features that represent a candidate’s LinkedIn profile or resume content.

Key columns in the dataset:

- `job_title`: The candidate's current job title
- `location`: The geographic location of the candidate
- `connections`: A numerical indicator of the candidate's professional connections (between 0 and 500+)
- `fit`: A column left empty by the provider, presumably for use in modeling
---

## 🔍 Methodology

### 1. Data Preparation
- Load candidate dataset (`104` records)
- Remove/handle empty columns (e.g., `'fit'`)
- Normalize the `connections` feature
- Merge the `job_title` and `location` features

### 2. NLP Matching Pipeline
- Embed candidate CVs and job keywords using 5 different vectorization methods
- Calculate cosine similarity between job keyword and CVs
- Rank candidates based on similarity scores

### 3. Learn-to-Rank Model
- Generate synthetic pairwise labels from similarity-based rankings
- Use a small neural network in PyTorch to train a pairwise ranker
- Evaluate using loss metrics and validate relative ranking preservation

---

## 🛠️ Technologies & Libraries Used

- Python 3.10+
- Pandas & NumPy
- Scikit-learn
- Gensim (Word2Vec, FastText)
- Transformers (BERT encoder)
- PyTorch (custom Dataset, DataLoader, and model training)
- Matplotlib & Seaborn (optional for EDA)

---

## 📊 Results

- All vectorization methods produced viable similarity rankings. However, **TF-IDF**, **Word2Vec**, and **BERT** encoders marginally outperformed the other vectorization methods.
- The pairwise ranker successfully learned to generalize rankings with reasonably low loss, although improvements are possible with more data and fine-tuning.

---

## 🧠 Future Work

- Use a larger dataset of resumes and job descriptions
- Introduce human-labeled relevance scores to improve supervision
- Fine-tune transformer-based embeddings for domain-specific context
- Deploy as an API or integrate into a web-based recruitment dashboard

---

## 📄 License

This project is open-source and available under the MIT License.
