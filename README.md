AI Model Comparison Tool
An interactive machine learning application built with Python, Scikit-learn, and Streamlit to train, evaluate, and compare classification models through a user-friendly web interface.
Features
* Multiple ML Models: Compare Logistic Regression, Decision Tree, and Random Forest classifiers.
* Performance Evaluation: Measure Accuracy, Precision, Recall, and F1-score.
* Cross-Validation: Evaluate model performance using 5-fold cross-validation.
* Data Upload: Upload custom CSV datasets and select the target column.
* Data Preprocessing: Handle numerical and categorical features using Scikit-learn pipelines.
* Data Visualization: Explore model performance, confusion matrices, and feature importance.
* Live Predictions: Test trained models with new input values.
* Excel Export: Download model comparison and cross-validation results as Excel files.
Technologies Used
* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* OpenPyXL
Getting Started
Prerequisites
Install Python and Git before starting.
Installation
1. Clone the repository:
2. Navigate to the project directory:
3. Create a virtual environment:
4. Activate the virtual environment on Windows:
5. Install the required dependencies:
6. Run the application:
streamlit run app.py
How to Use
1. Launch the application.
2. Choose the built-in Iris dataset or upload a CSV file.
3. Select the target column containing the classes to predict.
4. Configure the test size and model settings.
5. Compare model evaluation metrics and cross-validation results.
6. Explore visualizations, feature importance, and live predictions.
7. Export results to Excel when needed.
Note: The application currently supports classification tasks. The selected target column should contain categorical classes rather than continuous numerical values.
Project Purpose
This project demonstrates practical machine learning workflows, including data preprocessing, model training, performance evaluation, cross-validation, visualization, and interactive predictions.
Future Improvements
* Add regression model support.
* Implement hyperparameter tuning.
* Add more classification algorithms.
* Improve model evaluation and reporting.
 
⸻
 
Author: [Dala Alshammari]