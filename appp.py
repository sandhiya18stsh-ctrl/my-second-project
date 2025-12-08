from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
from typing import List, Optional
import io
import csv
import uuid
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import download
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from pydantic import BaseModel
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download NLTK data
try:
    download('punkt')
    download('stopwords')
except:
    pass

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    filename = Column(String(255))
    total_comments = Column(Integer, default=0)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), index=True)
    text = Column(Text)
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    summary = Column(Text)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class CommentBase(BaseModel):
    text: str
    sentiment: str
    sentiment_score: float
    summary: str

class AnalysisSessionBase(BaseModel):
    filename: str
    total_comments: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

class SentimentSummary(BaseModel):
    total_comments: int
    positive_count: int
    negative_count: int
    neutral_count: int

class WordCloudData(BaseModel):
    text: str
    weight: int

class AnalysisResults(BaseModel):
    session_id: str
    summary: SentimentSummary
    wordcloud_data: List[WordCloudData]

# Utility classes
class SentimentAnalyzer:
    def __init__(self, use_transformers=True):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.use_transformers = use_transformers
        
        # Initialize transformers model for more accurate sentiment analysis
        if use_transformers:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
                self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
                self.transformers_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            except Exception as e:
                print(f"Failed to load transformers model: {e}")
                self.use_transformers = False
        
    def analyze_sentiment(self, text: str):
        """Analyze sentiment using VADER or transformers with improved neutral detection"""
        if not text or str(text).strip() == '':
            return {"sentiment": "neutral", "score": 0.0}
            
        # Use transformers for more accurate sentiment analysis if available
        if self.use_transformers:
            try:
                result = self.transformers_pipeline(text[:512])[0]  # Limit text length
                label = result['label']
                score = result['score']
                
                if label == 'positive':
                    return {"sentiment": "positive", "score": score}
                elif label == 'negative':
                    return {"sentiment": "negative", "score": score}
                else:
                    return {"sentiment": "neutral", "score": score}
            except Exception as e:
                print(f"Transformers sentiment analysis failed: {e}")
                # Fall back to VADER
                pass
                
        # Fall back to VADER if transformers not available or failed
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        # Improved neutral detection with tighter thresholds
        if compound_score >= 0.1:
            return {"sentiment": "positive", "score": compound_score}
        elif compound_score <= -0.1:
            return {"sentiment": "negative", "score": compound_score}
        else:
            # Check if text is actually neutral or just short/ambiguous
            if len(text.split()) < 5:
                return {"sentiment": "neutral", "score": compound_score}
            
            # Additional check for mixed sentiment
            if abs(scores['pos'] - scores['neg']) < 0.1:
                return {"sentiment": "neutral", "score": compound_score}
            elif scores['pos'] > scores['neg']:
                return {"sentiment": "positive", "score": compound_score}
            else:
                return {"sentiment": "negative", "score": compound_score}
    
    def generate_summary(self, text: str, max_sentences: int = 2):
        """Generate improved extractive summary using hybrid approach"""
        if not text or len(text.split()) < 10:
            return text
            
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
            
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except:
            # Fallback if TF-IDF fails
            return ". ".join(sentences[:max_sentences]) + "."
        
        # Calculate sentence scores using TF-IDF
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Additional scoring based on position (first sentences often important)
        for i in range(len(sentence_scores)):
            # Give bonus to first few sentences
            if i < 2:
                sentence_scores[i] *= 1.5
            # Penalize very short sentences
            if len(sentences[i].split()) < 5:
                sentence_scores[i] *= 0.7
        
        # Get top sentences
        top_sentence_indices = sentence_scores.argsort()[-max_sentences:][::-1]
        summary_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
        
        # Ensure the summary makes sense
        summary = ' '.join(summary_sentences)
        if len(summary.split()) < 10:  # If summary is too short, use different approach
            # Fallback: use first and last sentences
            if len(sentences) >= 2:
                summary = sentences[0] + " " + sentences[-1]
            else:
                summary = text[:200] + "..." if len(text) > 200 else text
        
        return summary
    
    def process_for_word_cloud(self, texts: List[str]):
        """Process texts to generate word frequency data with improved filtering"""
        all_words = []
        
        for text in texts:
            if text:
                # Clean and tokenize text with better filtering
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                words = [word for word in words if word not in self.stop_words]
                # Remove common but uninformative words
                common_words = {'would', 'could', 'should', 'also', 'however', 'therefore', 'thus'}
                words = [word for word in words if word not in common_words]
                all_words.extend(words)
        
        # Count word frequencies
        word_freq = Counter(all_words)
        
        # Convert to list of dictionaries
        return [{"text": word, "weight": count} for word, count in word_freq.most_common(50)]

class CSVProcessor:
    def process_csv(self, file_content: bytes):
        """Process uploaded CSV file and extract comments with improved error handling"""
        try:
            # Try to read with pandas first
            df = pd.read_csv(io.BytesIO(file_content))
            comments = []
            
            # Look for potential comment columns with improved detection
            comment_columns = []
            column_priority = ['comment', 'feedback', 'suggestion', 'text', 'review', 'response', 'opinion']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in column_priority):
                    comment_columns.append((col, column_priority.index(next(k for k in column_priority if k in col_lower))))
            
            # Sort by priority
            comment_columns.sort(key=lambda x: x[1])
            comment_columns = [col[0] for col in comment_columns]
            
            if not comment_columns:
                # If no obvious comment columns, use the first string column
                for col in df.columns:
                    if df[col].dtype == 'object':  # String column
                        comment_columns.append(col)
                        break
            
            if not comment_columns:
                raise ValueError("No suitable comment column found in CSV")
            
            # Extract comments with improved handling
            for _, row in df.iterrows():
                comment_added = False
                for col in comment_columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        comment_text = str(row[col]).strip()
                        if len(comment_text) > 10:  # Minimum length requirement
                            comments.append({
                                'text': comment_text,
                                'source': f"Column: {col}"
                            })
                            comment_added = True
                            break  # Only use one comment per row
                
                # If no comment found in priority columns, try all columns
                if not comment_added:
                    for col in df.columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            comment_text = str(row[col]).strip()
                            if len(comment_text) > 10:
                                comments.append({
                                    'text': comment_text,
                                    'source': f"Column: {col}"
                                })
                                break
            
            return comments
            
        except Exception as e:
            # Fallback to CSV reader if pandas fails
            try:
                decoded_content = file_content.decode('utf-8')
                io_string = io.StringIO(decoded_content)
                reader = csv.reader(io_string)
                
                comments = []
                headers = next(reader)
                
                for row in reader:
                    for i, value in enumerate(row):
                        if value and str(value).strip():
                            comment_text = str(value).strip()
                            if len(comment_text) > 10:  # Minimum length requirement
                                comments.append({
                                    'text': comment_text,
                                    'source': f"Column: {headers[i] if i < len(headers) else f'Column {i}'}"
                                })
                                break  # Only use one value per row
                
                return comments
            except Exception as e2:
                raise Exception(f"Failed to process CSV: {str(e2)}")

def generate_wordcloud_image(wordcloud_data: List[dict]):
    """Generate word cloud image and return as base64 string with improved styling"""
    if not wordcloud_data:
        return ""
    
    # Create word frequency dictionary
    word_freq = {item['text']: item['weight'] for item in wordcloud_data}
    
    # Generate word cloud with improved styling
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(word_freq)
    
    # Convert to base64
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_str}"

# FastAPI app
app = FastAPI(title="E-Consultation Sentiment Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
sentiment_analyzer = SentimentAnalyzer(use_transformers=True)
csv_processor = CSVProcessor()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze_text")
async def analyze_text(text: str = Form(...), db: Session = Depends(get_db)):
    try:
        session_id = str(uuid.uuid4())
        
        # Create analysis session
        db_session = AnalysisSession(
            session_id=session_id,
            filename="manual_input",
            total_comments=1
        )
        
        # Analyze the text
        sentiment_result = sentiment_analyzer.analyze_sentiment(text)
        summary = sentiment_analyzer.generate_summary(text)
        
        # Update sentiment counts
        if sentiment_result["sentiment"] == "positive":
            db_session.positive_count = 1
        elif sentiment_result["sentiment"] == "negative":
            db_session.negative_count = 1
        else:
            db_session.neutral_count = 1
        
        db.add(db_session)
        
        # Store comment
        db_comment = Comment(
            session_id=session_id,
            text=text,
            sentiment=sentiment_result["sentiment"],
            sentiment_score=sentiment_result["score"],
            summary=summary
        )
        db.add(db_comment)
        db.commit()
        
        # Generate word cloud data
        wordcloud_data = sentiment_analyzer.process_for_word_cloud([text])
        wordcloud_image = generate_wordcloud_image(wordcloud_data)
        
        return {
            "session_id": session_id,
            "sentiment": sentiment_result["sentiment"],
            "sentiment_score": sentiment_result["score"],
            "summary": summary,
            "wordcloud_image": wordcloud_image,
            "message": "Text analyzed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        content = await file.read()
        comments_data = csv_processor.process_csv(content)
        
        if not comments_data:
            raise HTTPException(status_code=400, detail="No comments found in the file")
        
        session_id = str(uuid.uuid4())
        db_session = AnalysisSession(
            session_id=session_id,
            filename=file.filename,
            total_comments=len(comments_data)
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        analyzed_comments = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        all_texts = []
        
        for comment_data in comments_data:
            text = comment_data['text']
            all_texts.append(text)
            
            sentiment_result = sentiment_analyzer.analyze_sentiment(text)
            summary = sentiment_analyzer.generate_summary(text)
            
            db_comment = Comment(
                session_id=session_id,
                text=text,
                sentiment=sentiment_result["sentiment"],
                sentiment_score=sentiment_result["score"],
                summary=summary
            )
            db.add(db_comment)
            
            sentiment_counts[sentiment_result["sentiment"]] += 1
            
            analyzed_comments.append({
                "text": text,
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["score"],
                "summary": summary
            })
        
        db_session.positive_count = sentiment_counts["positive"]
        db_session.negative_count = sentiment_counts["negative"]
        db_session.neutral_count = sentiment_counts["neutral"]
        db.commit()
        
        wordcloud_data = sentiment_analyzer.process_for_word_cloud(all_texts)
        wordcloud_image = generate_wordcloud_image(wordcloud_data)
        
        summary = {
            "total_comments": len(comments_data),
            "positive_count": sentiment_counts["positive"],
            "negative_count": sentiment_counts["negative"],
            "neutral_count": sentiment_counts["neutral"]
        }
        
        return {
            "session_id": session_id,
            "summary": summary,
            "wordcloud_image": wordcloud_image,
            "message": f"Processed {len(comments_data)} comments successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session_data(session_id: str, db: Session = Depends(get_db)):
    session = db.query(AnalysisSession).filter(AnalysisSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    comments = db.query(Comment).filter(Comment.session_id == session_id).all()
    all_texts = [comment.text for comment in comments]
    wordcloud_data = sentiment_analyzer.process_for_word_cloud(all_texts)
    wordcloud_image = generate_wordcloud_image(wordcloud_data)
    
    summary = {
        "total_comments": session.total_comments,
        "positive_count": session.positive_count,
        "negative_count": session.negative_count,
        "neutral_count": session.neutral_count
    }
    
    return {
        "session_id": session_id,
        "summary": summary,
        "wordcloud_image": wordcloud_image,
        "comments": [{
            "text": c.text,
            "sentiment": c.sentiment,
            "sentiment_score": c.sentiment_score,
            "summary": c.summary
        } for c in comments]
    }

@app.get("/api/stats")
async def get_overall_stats(db: Session = Depends(get_db)):
    """Get overall statistics across all analysis sessions"""
    sessions = db.query(AnalysisSession).all()
    
    total_comments = sum(session.total_comments for session in sessions)
    total_positive = sum(session.positive_count for session in sessions)
    total_negative = sum(session.negative_count for session in sessions)
    total_neutral = sum(session.neutral_count for session in sessions)
    
    return {
        "total_sessions": len(sessions),
        "total_comments": total_comments,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "total_neutral": total_neutral
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)