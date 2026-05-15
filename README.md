# Lumina Crop Recommendation Project

This is a machine learning project that trains a Random Forest model to recommend crops based on soil and weather conditions (such as Nitrogen, Phosphorous, Potassium, temperature, humidity, pH, and rainfall). 

It also includes an optional integration setup with Google's Gemini Large Language Model for advanced insights.

## Prerequisites

Make sure you have Python installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Open a terminal or command prompt in the project directory.
2. Run the main script:

```bash
python "lumina project.py"
```

### What the script does:
- **Data Generation**: Automatically creates a sample dataset `crop_recommendation.csv` if it doesn't already exist.
- **Model Training**: Trains a Random Forest Classifier on the dataset.
- **Model Evaluation**: Evaluates the model and prints the classification report.
- **Model Saving**: Saves the trained model locally as `crop_recommendation_model.pkl` using `joblib`.

## Using Gemini (Optional)

If you want to use the Gemini AI integration:
1. Open `lumina project.py`.
2. Uncomment the code inside the `setup_gemini()` function.
3. Make sure to provide your Gemini API Key when prompted.
