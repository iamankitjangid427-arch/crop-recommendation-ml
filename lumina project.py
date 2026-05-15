import os
import getpass
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Optional: Import for Gemini if you plan to use it
# from langchain_google_genai import ChatGoogleGenerativeAI

def create_sample_data(filename='crop_recommendation.csv'):
    """Creates a sample crop dataset if it doesn't exist."""
    if os.path.exists(filename):
        print(f"Dataset '{filename}' already exists. Skipping creation.")
        return
    
    print(f"Creating sample dataset: '{filename}'...")
    data = {
        'N': [90, 80, 60, 85, 75],
        'P': [42, 35, 30, 40, 38],
        'K': [43, 40, 45, 42, 41],
        'temperature': [20.87, 22.5, 19.1, 21.0, 23.1],
        'humidity': [82.0, 80.5, 83.1, 81.5, 79.8],
        'ph': [6.5, 6.7, 6.2, 6.4, 6.8],
        'rainfall': [202.93, 185.2, 210.3, 195.0, 180.5],
        'label': ['rice', 'wheat', 'maize', 'rice', 'wheat']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print("Sample dataset created and saved.")

def train_and_save_model(data_filename='crop_recommendation.csv', model_filename='crop_recommendation_model.pkl'):
    """Trains the RandomForest model and saves it."""
    print("\n--- Training Model ---")
    
    # Load dataset
    data = pd.read_csv(data_filename)
    print("Sample Data:\n", data.head(), "\n")
    
    # Prepare features and label
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    # Train-test split
    # Using a small test_size and handling zero_division since sample data is very small.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Evaluation Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # Save the model
    joblib.dump(model, model_filename)
    print(f"Model successfully saved as '{model_filename}'")

def setup_gemini():
    """Placeholder for setting up Gemini API."""
    print("\n--- Setting up Gemini ---")
    # if 'GOOGLE_API_KEY' not in os.environ:
    #     os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter the Gemini API Key: ")
    # llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
    # print("Gemini LLM initialized.")
    print("Gemini code is currently commented out. Uncomment in setup_gemini() to use if needed.")

if __name__ == "__main__":
    # 1. Create Data
    create_sample_data()
    
    # 2. Train and Save Model
    train_and_save_model()
    
    # 3. Setup Gemini (Optional)
    setup_gemini()
