# 🧠 Mashroom GenAI RAG System

This project was developed for the Mashroom internship technical challenge. The goal was to build a smart assistant capable of answering user questions based on **PDF documents** and **interior design images** — an important step toward improving how users interact with 3D, AR, and AI-driven design tools.

---

## 🚀 What It Does

- ✅ Answers natural language questions based on uploaded **PDF reports**
- ✅ Supports interior **image uploads** and interprets visual queries (e.g., furniture, style)
- ✅ Fully **local** solution — no OpenAI or paid APIs used
- ✅ Clean API built with **FastAPI**, documented and testable via Swagger UI

---

## 🛠️ Tech Stack

| Module         | Tech Used                            |
|----------------|--------------------------------------|
| AI Framework   | LangChain, FAISS                     |
| GenAI (Text)   | HuggingFace `flan-t5-base` (local)   |
| GenAI (Vision) | BLIP-2 (`blip2-flan-t5-xl`)          |
| Backend        | FastAPI                              |
| Utilities      | transformers, torch, Pillow, accelerate |

---

## 📂 Project Structure

mashroom-genai-rag/
│
├── main.py # FastAPI API entrypoint
├── ai_modules/
│ ├── pdf_rag.py # RAG pipeline for PDFs
│ └── image_vision.py # Vision-based Q&A for images
│
├── sample_data/ # Uploaded PDFs & images
│ └── images/
│
├── vectorstore_index/ # FAISS index for PDFs
├── requirements.txt
└── README.md

Example Queries


📄 PDF Mode
POST /upload_pdf/ — upload a report
POST /ask_pdf/ — e.g.:
What is the total budget for this project?
Who is responsible for the implementation phase?


🖼️ Image Mode
POST /upload_image/ — upload JPG/PNG
POST /ask_image/ — e.g.:
What style is this room?
What kind of furniture is used in this layout?


⚠️ Known Limitation (Image Pipeline)
While the image module is fully implemented using BLIP-2 Flan-T5-XL, a runtime error currently blocks execution during tensor expansion:
Error processing image: The expanded size of the tensor (1) must match the existing size (1408) at non-singleton dimension 2.
This issue appears to be related to offloading mechanics under accelerate when using limited-memory hardware (in this case: Apple M1/MPS with ~16GB RAM).


✅ Note: The image pipeline was tested earlier with a smaller BLIP model and successfully returned structured answers such as:
"Answer: Modern"
This confirms that the vision-based flow functions correctly, but requires more GPU or local memory to support larger BLIP models (like blip2-flan-t5-xl).


📌 Deliverables Checklist
 Modular LangChain pipeline with PDF ingestion + retrieval
 Local text generation via flan-t5-base
 Local vision-language model integration (BLIP-2)
 Web-ready FastAPI backend with file upload endpoints
 Swagger docs for testing each step
 GitHub-hosted project with clean structure
 Documented setup + tech rationale


💻 Runtime Note
This was built and tested on:
Device: MacBook Pro M1 (16 GB RAM)
Backend: Apple Metal (MPS) + PyTorch
Vision model: BLIP-2 FLAN-T5-XL (~15 GB total)
While large vision models may not fully run due to memory constraints, everything else works offline and out-of-the-box.


🙏 Thank You
Thank you for reviewing this project. I'm excited by the opportunity to contribute to Mashroom’s mission of building intelligent, immersive tools for design and architecture.