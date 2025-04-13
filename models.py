from sqlalchemy import Column, Integer, String, Float, Date, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Define the base class for SQLAlchemy models
Base = declarative_base()

# Hotel Booking Model
class HotelBooking(Base):
    __tablename__ = "hotel_bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    hotel = Column(String, nullable=False)
    arrival_date = Column(Date, nullable=False)
    country = Column(String, nullable=True)
    adr = Column(Float, nullable=True)
    is_canceled = Column(Integer, default=0)

# Query History Model
class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# SQLite Database URL
DATABASE_URL = "sqlite:///./hotel_bookings.db"

# SQLAlchemy engine & sessionmaker
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to initialize the database
def init_db():
    Base.metadata.create_all(bind=engine)
