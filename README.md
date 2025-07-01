Authentic Personal Blog Discovery System

Features:
Authentic Page Searching: Users can search for web pages using keywords, with results ranked using PageRank algorithm and TF-IDF scoring
Personal Blog Focus: Specialized in filtering and displaying only personal blogs and insightful articles

Architecture: 
The system follows a modern microservices architecture with:
Frontend: React-based user interface
Backend: Flask API for search functionality
Database: MongoDB Atlas for scalable cloud storage
ML Pipeline: DistilBERT/BERT-base models with LoRA adapters
Search Engine: TF-IDF with PageRank for relevance scoring

TF-IDF Implementation:

Why TF-IDF? Term Frequency-Inverse Document Frequency is ideal for this project because it:
Identifies important terms while reducing noise from common words
Provides fast similarity calculations for real-time search
Works well with the hybrid scoring system combining PageRank
Scales efficiently with MongoDB Atlas infrastructure

Workflow Process:

1. Data Collection: Scrapy framework scrapes blog content from various websites
2. Data Processing: Content is cleaned and prepared for ML training
3. Model Training: DistilBERT and BERT-base models are fine-tuned with LoRA adapters
4. Classification: Trained models classify blogs as personal vs non-personal
5. Indexing: TF-IDF vectorization creates searchable indices
6. Storage: Processed data stored in MongoDB Atlas
7. Search: Hybrid search combining TF-IDF, text search, and PageRank scores
8. Frontend: React application displays filtered personal blog results
