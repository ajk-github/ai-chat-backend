# AI AR Chat Backend

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=firebase_config.json
```


### 5. Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `POST /api/upload` - Upload and process files (Excel, CSV, Parquet)
- `POST /api/chat` - Send chat messages and query data
- `DELETE /api/chat/delete/{chat_id}?user_id={user_id}` - Delete a chat session
- `GET /health` - Health check endpoint



For production (No Auto Reload):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```