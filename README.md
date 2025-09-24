# Natural Language Coffee Recommendation System

**Live Demo:** [BeanRec.com](https://beanrec.com)

A full-stack semantic search engine for coffee reviews. This application takes a user's natural language description of a coffee (e.g., "a bright, fruity coffee with citrus notes") and returns the most relevant recommendations from a database of expert reviews.

The core of the system is a fine-tuned dual-encoder sentence-transformer model (SBERT) fused with a metadata encoder composed of embedding layers for categorical data and a small dense network for numerical data that learns to map user queries and coffee profiles to the same high-dimensional vector space.

## System Architecture
1. Data Scraping: A python script (`scrape_data.py`) scrapes the data from [CoffeeReview.com](coffeereview.com), taking advantage of their advanced search pages. It first collects the review links, then scrapes the relevant data from each individual link. The script takes optional arguments including `input_file` (if you are appending to an existing file), `scrape_type` (taking on values "all" or "pages" for a manually set number of pages), `num_pages` (the number of pages when `scrape_type = "pages"`, default is 10), and `num_workers` (defines the number of processors for scraping the links in parallel). 

2. Data Preprocessing & Feature Engineering: A python script (`preprocess_data.py`) performs extensive data cleaning and feature engineering, like extracting origins / varietals from unstructure text, normalizing prices, and determining test method. A MySQL Database is then then filled using with both raw and preprocessed data via `populate_db.py`.

3. Synthetic Query Generation: A python script (`generate_llm_queries.py`) calls an LLM to generate more complex, conversational, and nuanced queries for each coffee. `generate_programmatic_queries.py` then systematically generates queries based on key attributes for each coffee to create a large volume of simple, factual queries. 

4. Modeling: Model architecture is defined in (`model.py`). The model is primarily a dual-encoder sentence transformer (SBERT, used primarily for text embeddings like expert review or user query) fused with a metadata encoder (primarily for categorical and numerical data in reviews, like origin, flavor scores, etc.) The training is done via `train.py`, which takes arguments for learning rate (individual rates for the transformer weights and metadata encoder weights), batch size, number of epochs, the epoch at which to switch to semi-hard negative mining, and model path (for a pretrained model, beyond the initial SBERT weights). Validation is performed in `evaluate.py`, where both Recall@K and NDCG@K can be computed. Recommendations can be served using methods in `predict.py`, which allows the model to serve any of the 3,000 coffees.  

5. Indexing: All the coffees in the catalog are run through the model to get their embeddings and stored in an efficient FAISS index for real time search for minimal latency and resources.

6. Backend API: The backend, contained in `app_backend.py`, is containerized along with its dependecies with Docker, stored on Google Cloud Artifact Registry, and deployed on Google Cloud Run with only 1 processor and 4 GB of memory. It loads the trained model and FAISS index, exposing a REST API endpoint for recommendations.

7. Frontend: A static single-page web application (HTML, CSS, JS, `index.html`) served directly from a Google Cloud Storage bucket using Google Cloud Compute. Load balancer is configured to point custom domain name to front end. 

## Tech Stack
- ML & Data: PyTorch, Transformers, FAISS, Sentence-Transformers, Pandas, NumPy Scikit-learn
- Backend: Flask
- Database: SQLite
- Deployment: Docker, Google Cloud Platform (Cloud Run, Artifact Registry, Cloud Storage, Compute)
- Frontend: HTML, Tailwin CSS, Javascript

## Local Setup & Installation
These instructions will guide you through setting up the application to run locally.

**Prerequisites**:
- Docker
1. Clone the Repository
2. Build the .dev Dockerfile (will tak 6-7 minutes)

`docker build -f Dockerfile.dev -t coffee-recommender.dev .`

3. Run the Docker container (will be available on localhost:5001)

`docker run --init -e PORT=5000 -p 5001:5000 coffee-recommender.dev`

If you would instead like to run specific files locally, use env.yml to run the full anaconda environment. 
