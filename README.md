# ML Model Evaluation Website

This web application allows users to upload a CSV file, and it will train and evaluate five different machine learning models (Logistic Regression, SVM, KNN, Random Forest, and Decision Tree) on the provided data. The evaluation metrics displayed for each model are Accuracy, Precision, Recall, F1 Score, and the Confusion Matrix.

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Setup and Installation

1.  **Clone the repository or download the files.**

2.  **Navigate to the project directory:**
    ```bash
    cd path/to/your/project
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Ensure you are in the project directory and the virtual environment is activated.**

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  **Open your web browser and go to:**
    ```
    http://127.0.0.1:5000/
    ```

## How to Use

1.  Click on the "Choose File" button to select a CSV file from your computer.
    - The CSV file should have features in the initial columns and the target variable in the last column.
    - The application attempts to convert all feature columns to numeric types. Non-numeric values that cannot be converted will be replaced by the mean of their respective columns.
    - The target column will be factorized (converted to numerical labels) if it's not already numeric.
2.  Click the "Upload and Evaluate" button.
3.  The website will display the performance metrics for each of the five models.

## Project Structure

```
/
|-- app.py                  # Main Flask application
|-- requirements.txt        # Python package dependencies
|-- README.md               # This file
|-- uploads/                # Directory for storing uploaded CSV files (created automatically)
|-- templates/
|   |-- index.html          # HTML template for the web page
|-- static/
    |-- style.css           # CSS styles for the web page
```

## Notes

- The application assumes the last column of the CSV is the target variable and all other columns are features.
- Basic preprocessing is applied: features are converted to numeric (NaNs filled with mean), and the target is factorized if not numeric.
- For more complex datasets, further preprocessing might be required outside of what this simple application provides.
- `zero_division=0` is used in precision and recall calculations to handle cases where a class might not be predicted, preventing the application from crashing. 