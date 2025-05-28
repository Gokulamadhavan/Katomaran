from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class FaceRecord(Base):
    __tablename__ = "face_records"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    embedding = Column(LargeBinary)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

DATABASE_URL = "sqlite:///./db.sqlite"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def add_face_record(name: str, embedding: bytes, timestamp: datetime.datetime):
    db = SessionLocal()
    record = FaceRecord(name=name, embedding=embedding, timestamp=timestamp)
    db.add(record)
    db.commit()
    db.close()

def get_all_face_encodings():
    db = SessionLocal()
    records = db.query(FaceRecord).all()
    db.close()
    return [{"name": r.name, "embedding": r.embedding} for r in records]
