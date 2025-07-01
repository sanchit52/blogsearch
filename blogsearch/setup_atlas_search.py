#!/usr/bin/env python3
"""
Blog Search Engine Setup Script for MongoDB Atlas
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required Python packages"""
    requirements = [
        'pymongo[srv]',  # Include srv for Atlas connection
        'scikit-learn',
        'nltk',
        'flask',
        'flask-cors',
        'python-dotenv',
        'numpy'
    ]
    
    print("📦 Installing Python dependencies...")
    for package in requirements:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                          check=True, capture_output=True)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")

def setup_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'search_engine',
        'database'
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print("⚠️  .env file not found!")
        print("Please create a .env file with the following variables:")
        print("""
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.xxxxx.mongodb.net/blogsearch?retryWrites=true&w=majority
DATABASE_NAME=blogsearch
FLASK_ENV=development
FLASK_DEBUG=True
        """)
        return False
    else:
        print("✅ .env file found")
        
        # Check if connection string exists
        with open(env_file, 'r') as f:
            content = f.read()
            if 'MONGODB_CONNECTION_STRING' in content:
                print("✅ MongoDB connection string found in .env")
                return True
            else:
                print("❌ MONGODB_CONNECTION_STRING not found in .env file")
                return False

def setup_database():
    """Initialize the database with blog data"""
    try:
        print("🗄️  Setting up MongoDB Atlas database...")
        
        # Import and run the Atlas setup
        from database.atlas_setup import BlogSearchAtlasDB
        
        db = BlogSearchAtlasDB()
        
        if db.test_connection():
            db.create_indexes()
            
            # Check if data files exist
            personal_file = "data/processed/final/personal_blogs_fixed.jsonl"
            non_personal_file = "data/processed/final/non_personal_blogs_fixed.jsonl"
            
            if os.path.exists(personal_file) or os.path.exists(non_personal_file):
                count = db.load_blog_data(personal_file, non_personal_file)
                print(f"✅ Database setup complete! {count} blogs indexed.")
                
                # Show final stats
                db.get_collection_stats()
                return True
            else:
                print("⚠️  Blog data files not found.")
                print(f"   Looking for: {personal_file}")
                print(f"   Looking for: {non_personal_file}")
                return False
        else:
            print("❌ Failed to connect to MongoDB Atlas")
            return False
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Blog Search Engine with MongoDB Atlas...\n")
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Check environment configuration
    if not check_env_file():
        print("\n❌ Setup cannot continue without proper .env configuration")
        sys.exit(1)
    
    # Step 4: Setup database
    if setup_database():
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Start the Flask API: python search_api/app.py")