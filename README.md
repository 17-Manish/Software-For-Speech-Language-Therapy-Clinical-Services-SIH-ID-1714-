# Software-For-Speech-Language-Therapy-Clinical-Services-SIH-ID-1714-
A simple system to manage speech and language therapy work using Python, Flask, and SQLite. It helps student therapists handle patients and suggests therapy plans using a basic ML (Nearest Neighbor) model. The app makes therapy work easier by organizing data, reducing manual work, and giving useful recommendations.

->Tech Stack
-Backend: Python, Flask
-Database: SQLite
-Machine Learning: Nearest Neighbor (K-NN)
-Frontend: HTML, CSS (Flask templates)

 Structure of the project
 project/
│── app.py
│── model/
│     └── knn_model.py
│── templates/
│── static/
│── database.db
└── README.md

how to run
-Install dependencies
pip install -r requirements.txt
-Run the Flask app
python app.py
-Open in browser
http://127.0.0.1:5000/

