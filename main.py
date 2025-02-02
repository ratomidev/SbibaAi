# #9:42from fastapi import FastAPI, HTTPException
from services.pdf_processing import extract_text_from_pdf, chunk_text
from services.zilliz_service import create_vector_store, query_zilliz_database
from services.groq_service import get_groq_response
from utils.file_utils import read_text_from_file
from utils.text_utils import create_prompt_template
from config import UPLOAD_DIR

app = FastAPI()

@app.get('/')
async def root_home():
    return "Hello visitor"

@app.get('/{country}')
async def root(country: str):
    prompt_template = create_prompt_template()
    return get_groq_response(prompt_template.format(country=country))

@app.get('/few_shot_learning/{equations}')
async def calculate(equations: str):
    return get_groq_response(equations)

@app.post('/upload_pdf/')
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    create_vector_store(chunks)
    
    return {"filename": file.filename, "message": "PDF uploaded and processed successfully."}

@app.get('/query_pdf/')
async def query_pdf(query: str, k: int = 3):
    results = query_zilliz_database(query, k)
    return {"query": query, "results": results}
