from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import traceback

from ai_modules.image_vision import answer_about_image
from ai_modules.pdf_rag import process_pdf, ask_question, load_existing_vectorstore

app = FastAPI(
    title="Mashroom RAG API",
    description="A local-only RAG system for PDF and image-based interior design Q&A.",
    version="1.0.0"
)

# Directories
VSTORE_DIR = "vectorstore_index"
UPLOAD_DIR = "sample_data"
IMAGE_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "images")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

# PDF vector store
stored_db = None

@app.get("/")
async def root():
    return {"message": "Mashroom local RAG API is running."}

# PDF Upload
@app.post("/upload_pdf/", summary="Upload and embed a PDF")
async def upload_pdf(file: UploadFile = File(...)):
    global stored_db
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if os.path.exists(os.path.join(VSTORE_DIR, "index.faiss")):
            stored_db = load_existing_vectorstore(VSTORE_DIR)
            message = "Existing vectorstore loaded."
        else:
            stored_db = process_pdf(file_location, VSTORE_DIR)
            message = "New vectorstore created and PDF embedded successfully."

        return {"message": message}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to process PDF: {str(e)}"})

# PDF Q&A
@app.post("/ask_pdf/", summary="Ask a question based on the uploaded PDF")
async def ask_pdf(question: str = Form(...)):
    if stored_db is None:
        return JSONResponse(status_code=400, content={"error": "No PDF has been uploaded or processed yet."})
    try:
        answer = ask_question(stored_db, question)
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to answer question: {str(e)}"})

# Image Upload
@app.post("/upload_image/", summary="Upload an interior design image")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(IMAGE_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"message": f"Image '{file.filename}' uploaded successfully."}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# Image Q&A
@app.post("/ask_image/", summary="Ask a question about an uploaded image")
async def ask_image(filename: str = Form(...), question: str = Form(...)):
    try:
        file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "Image file not found."})
        answer = answer_about_image(file_path, question)
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
