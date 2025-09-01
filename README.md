# SkinSense AI: Skin Cancer Classification Web App

SkinSense AI is a web application designed to help users detect and classify skin cancer from images using deep learning. The project combines a Flask backend (Python) with a modern HTML/CSS/JS frontend, providing an easy-to-use interface for uploading skin lesion images and receiving instant predictions.

## Features
- Upload skin lesion images for analysis
- Deep learning model (Keras/TensorFlow) for skin cancer classification
- User-friendly web interface with responsive design
- Multiple static and dynamic pages (Learn, Forms, etc.)
- Error handling and informative feedback

## How It Works
1. **User uploads an image** of a skin lesion via the web interface.
2. The backend processes the image and uses a trained deep learning model (`SkinCancerClassificationModelFinal.h5`) to predict the type of skin cancer.
3. The result is displayed to the user with additional information and guidance.

## Project Structure
```
├── Working.py                  # Main Flask app (simple version)
├── _updated/                   # Main app folder (modular Flask structure)
│   ├── app/                    # Application code (init, forms, controllers)
│   ├── config/                 # Configuration files
│   ├── tests/                  # Unit tests
│   ├── run.py                  # Entry point for Flask app
│   └── ...
├── static/                     # Static files (CSS, JS, images)
├── templates/                  # HTML templates
├── requirements.txt            # Python dependencies
├── SkinCancerClassificationModelFinal.h5  # Trained ML model
└── README.md                   # Project documentation
```

## Setup & Installation
1. **Clone the repository:**
	```bash
	git clone https://github.com/LokeshWasnik/skinsense-ai
	cd <project-folder>
	```
2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
3. **Run the app:**
	```bash
	python Working.py
	# or for modular version
	python _updated/run.py
	```
4. **Open your browser:**
	Visit `http://localhost:5000` to use the app.

## Deployment
You can deploy this app for free on platforms like Render.com, Railway, or Replit. Make sure to include your `requirements.txt` (if using Render/Heroku-style deployment).

## Contributing
Pull requests and suggestions are welcome! Please open an issue to discuss changes or improvements.

## License
This project is for educational and research purposes only. Not for clinical use.







