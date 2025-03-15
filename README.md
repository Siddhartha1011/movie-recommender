
# Movie Recommender System   
A content-based movie recommendation system using TF-IDF and cosine similarity.  
Built with **FastAPI** for backend and **Gradio** for an interactive UI.  

## Features  
- Get movie recommendations based on plot similarity.  
- FastAPI handles API requests.  
- Gradio provides an easy-to-use web interface.  
- Uses TF-IDF for text vectorization and cosine similarity for ranking.  

## How to Run  
```bash
uvicorn movie_recommender:app --reload  # Start FastAPI  
python movie_recommender.py  # Start Gradio UI

Install the following dependencies on vs terminal using pip install scikit-learn fastapi uvicorn gradio.

Click run the following code on vs code terminal.

To run the fast api open a new teminal bash and run the following line of code in the terminal (uvicorn movie_recommender:app --reload).This will start the FastAPI server on your host link.

Click on the host link in code terminal to open the url on browser.
