# Hotel Booking Analytics with RAG System

## Overview

This project, **Hotel Booking Analytics with a Retrieval-Augmented Generation (RAG) System**, is designed to process hotel booking data, generate actionable insights, and enable natural language question-answering for users. This system integrates data analytics, machine learning models, and AI-backed question answering to help hotel staff and analysts make informed, data-driven decisions regarding revenue optimization, guest experience, and operational planning.

---

## Features

- **Data Ingestion & Cleaning**: Loads raw hotel booking data, handles missing values, and derives key features.
- **Feature Engineering**: Generates meaningful business metrics like total revenue, average daily rate (ADR), and total guests.
- **Analytics & Visualizations**: Computes monthly trends, cancellation rates, and other important statistics; visualizations provided for better understanding.
- **RAG System**: Converts processed data into a knowledge base using embeddings and allows for natural language queries.
- **Interactive UI**: Users can interact with the system via a Streamlit interface to ask questions and explore insights.

---

## Technologies Used

- **FastAPI**: RESTful API framework to expose endpoints for querying and interacting with the system.
- **Streamlit**: For building an interactive web interface for querying and displaying insights.
- **Pandas & NumPy**: Data processing and feature engineering.
- **LangChain**: Embedding generation and natural language processing (NLP).
- **FAISS**: A vector database used to store and retrieve embeddings for fast similarity search.
- **Mistral-7B** (via Hugging Face): A language model used for answering queries with context.
- **Matplotlib & Seaborn**: Data visualization tools for creating interactive charts and graphs.
- **SQLAlchemy & PostgreSQL** (Optional): For storing and managing hotel booking data.

---

## Installation

To run this project locally, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/SharmaKanishkaa/Buyogo-LLM-Powered-Booking-Analytics-QA-System.git
cd Buyogo-LLM-Powered-Booking-Analytics-QA-System
```

### 2. Install Dependencies

Create a virtual environment and install the required libraries:

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
# OR
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
```

### 3. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

This will open a web interface where you can interact with the system and ask questions based on hotel booking data.

---

## API Endpoints

Initially run: uvicorn main:app --reload --port 8000

The FastAPI backend exposes the following endpoints:

### 1. **/analytics**

- **Method**: `GET`
- **Description**: Retrieves analytics data (e.g., trends, cancellations, ADR) in a structured format.
- **Example Request**: 
  ```bash
  http://localhost:8000/analytics
  ```

### 2. **/ask**

- **Method**: `POST`
- **Description**: Accepts a natural language query and returns a generated response with relevant data insights.
- **Example Request**:
  ```bash
  http://127.0.0.1:8000/visualizations/monthly_adr
  curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"query": "Which months had the highest ADR?"}'
  ```
- **Example Response**:
  ```json
  {
    "response": "August and July, averaging $130 and $125 respectively."
  }
  ```

### 3. **/visualizations**

- **Method**: `GET`
- **Description**: Returns a list of available visualizations (e.g., ADR trends, cancellation rates).
- **Example Request**:
  ```bash
  curl http://localhost:8000/visualizations
  ```

---

## Sample Queries

- "Which months had the highest ADR?"
- "What are the peak check-in days?"
- "What markets have the highest cancellation rates?"
- "What is the average booking revenue for the past year?"

---

## Deployment

### Local Deployment

To run the application locally, follow the installation instructions above and use:

```bash
streamlit run app.py
```

## Limitations & Future Work

### Limitations
- FAISS may not scale well with very large datasets.
- The language model (Mistral-7B) may generate incorrect or vague answers with unclear prompts.
- The system currently supports only basic queries, but future work may include enhancing the natural language query handling.

### Future Enhancements
- Integrate live hotel Property Management Systems (PMS) for real-time data access.
- Add user authentication and role-based access.
- Expand the system to handle larger datasets using alternative vector stores (e.g., Pinecone).
- Implement more advanced query handling with fine-tuned models.

---

## Conclusion

The **Hotel Booking Analytics with RAG System** is a powerful tool that combines traditional data analytics with AI-driven insights. This solution helps hotel management teams optimize their operations, improve revenue forecasting, and enhance guest experiences through intelligent, data-backed decisions.
