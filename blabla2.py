from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request,Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import pandas as pd
from typing import List, Optional
import io
import csv
import uuid
import re
import nltk
from nltk.tokenize import sent_tokenize
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
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os

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
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_sentiment(self, text: str):
        if not text or str(text).strip() == '':
            return {"sentiment": "neutral", "score": 0.0}
            
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            return {"sentiment": "positive", "score": compound_score}
        elif compound_score <= -0.05:
            return {"sentiment": "negative", "score": compound_score}
        else:
            return {"sentiment": "neutral", "score": compound_score}
    
    def generate_summary(self, text: str, max_sentences: int = 2):
        if not text or len(text.split()) < 10:
            return text
            
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
            
        # Calculate word frequencies
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word not in self.stop_words and word not in '.,!?;:()[]{}"\'']
        word_freq = Counter(words)
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_freq:
                    if i in sentence_scores:
                        sentence_scores[i] += word_freq[word]
                    else:
                        sentence_scores[i] = word_freq[word]
        
        # Get top sentences
        if sentence_scores:
            top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
            summary_sentences = [sentences[i] for i in sorted(top_sentences)]
            return ' '.join(summary_sentences)
        else:
            return sentences[0]
    
    def process_for_word_cloud(self, texts: List[str]):
        all_words = []
        
        for text in texts:
            if text:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                words = [word for word in words if word not in self.stop_words]
                all_words.extend(words)
        
        word_freq = Counter(all_words)
        return [{"text": word, "weight": count} for word, count in word_freq.most_common(50)]

class CSVProcessor:
    def process_csv(self, file_content: bytes):
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            comments = []
            
            comment_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['comment', 'feedback', 'suggestion', 'text', 'review']):
                    comment_columns.append(col)
            
            if not comment_columns:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        comment_columns.append(col)
                        break
            
            if not comment_columns:
                raise ValueError("No suitable comment column found in CSV")
            
            for _, row in df.iterrows():
                for col in comment_columns:
                    if pd.notna(row[col]):
                        comments.append({
                            'text': str(row[col]),
                            'source': f"Column: {col}"
                        })
            
            return comments
            
        except Exception as e:
            try:
                decoded_content = file_content.decode('utf-8')
                io_string = io.StringIO(decoded_content)
                reader = csv.reader(io_string)
                
                comments = []
                headers = next(reader)
                
                for row in reader:
                    for i, value in enumerate(row):
                        if value and str(value).strip():
                            comments.append({
                                'text': str(value),
                                'source': f"Column: {headers[i] if i < len(headers) else f'Column {i}'}"
                            })
                
                return comments
            except Exception as e2:
                raise Exception(f"Failed to process CSV: {str(e2)}")

def generate_wordcloud_image(wordcloud_data: List[dict]):
    if not wordcloud_data:
        return ""
    
    word_freq = {item['text']: item['weight'] for item in wordcloud_data}
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    
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
sentiment_analyzer = SentimentAnalyzer()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
