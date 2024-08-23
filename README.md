# Predicting Social Media Post Engagement

## Introduction
This project aims to predict the engagement levels of social media posts based on their descriptions and other relevant features. Engagement is defined as the sum of interactions a post receives, such as likes and comments. By accurately predicting engagement, this project seeks to optimize content strategies, maximize reach, and improve overall social media performance.

## Dataset Overview
The dataset consists of social media posts with the following attributes:
- **Post descriptions**
- **Number of likes**
- **Number of comments**
- **Number of posts by the user**
- **Time of posting**
- **User follower count**

## Project Structure
The project is divided into seven key notebooks, each focusing on different stages of the data science pipeline:

1. **Data Collection:** Gathering and merging the base dataset with embedded text data.
2. **Data Cleaning and Wrangling:** Handling missing values, filtering data, and preparing it for analysis.
3. **Exploratory Data Analysis (EDA):** Understanding data distribution, relationships, and key patterns.
4. **Feature Engineering:** Creating new features from text, time, and categorical data to enhance model performance.
5. **Data Preprocessing:** Scaling and splitting the data into training and testing sets.
6. **Model Selection:** Training multiple models, including XGBoost, Random Forest, and Neural Networks.
7. **Model Evaluation and Hyperparameter Tuning:** Fine-tuning models and selecting the best one based on performance metrics.

## Setup
### Tools and Libraries
- **Python:** Core programming language used for data manipulation, analysis, and modeling.
- **Jupyter Notebook:** Interactive environment for running code and documenting analysis.
- **scikit-learn:** Library for machine learning algorithms and data preprocessing.
- **XGBoost:** Gradient boosting library used for high-performance model training.
- **TensorFlow:** Framework for building and training neural networks.
- **Matplotlib & Seaborn:** Libraries used for creating data visualizations.

## Project Workflow
1. **Data Collection**
   - **Objective:** Collect and merge the base dataset with embedded text data.
   - Gather data from multiple sources, focusing on the integration of large-scale datasets to ensure comprehensiveness.
   - Merge datasets based on common identifiers like `id` and `description`.

2. **Data Cleaning and Wrangling**
   - **Objective:** Prepare the data for analysis.
   - Handle missing values through imputation or removal.
   - Filter data based on criteria such as follower count and post activity.
   - Save the cleaned data for further analysis.

3. **Exploratory Data Analysis (EDA)**
   - **Objective:** Gain insights into the data through visualization and statistics.
   - Analyze descriptive statistics like mean, median, and distribution.
   - Visualize data using histograms, scatter plots, and heatmaps.
   - Identify correlations and patterns between features.

4. **Feature Engineering**
   - **Objective:** Enhance the dataset with new features.
   - Generate text-based features such as word count and sentiment scores.
   - Create time-based features like day of the week and hour of posting.
   - Perform one-hot encoding on categorical variables.
   - Save the engineered features for model training.

5. **Data Preprocessing**
   - **Objective:** Prepare the data for machine learning models.
   - Scale numerical features using `MinMaxScaler`.
   - Split the data into training and testing sets.
   - Save the preprocessed data for model selection.

6. **Model Selection**
   - **Objective:** Train and evaluate different models.
   - **XGBoost:** A powerful gradient boosting model.
   - **Random Forest:** An ensemble of decision trees for robust predictions.
   - **Gradient Boosting:** Another gradient boosting method.
   - **Neural Networks (TensorFlow):** A deep learning approach to capture complex patterns.
   - Evaluate each model using metrics like MAE, MSE, and R-squared.

7. **Model Evaluation and Hyperparameter Tuning**
   - **Objective:** Optimize model performance through tuning.
   - Use `GridSearchCV` to find the best hyperparameters for XGBoost and Random Forest.
   - Evaluate the tuned models on the test set.
   - Save the final model for deployment or further analysis.

## Results and Interpretation
### Model Performance
**Random Forest Performance**:
- Mean Absolute Error (MAE): 29.12
- Mean Squared Error (MSE): 1485.41
- R-squared: 0.38

**Gradient Boosting Performance**:
- Mean Absolute Error (MAE): 34.41
- Mean Squared Error (MSE): 1897.81
- R-squared: 0.21

**TensorFlow Neural Network Performance**:
- Mean Absolute Error (MAE): 34.12
- Mean Squared Error (MSE): 1859.50
- R-squared: 0.23

**XGBoost Performance**:
- Mean Absolute Error (MAE): 29.27
- Mean Squared Error (MSE): 1525.94
- R-squared: 0.37

**Best Model**: Random Forest

### Key Findings
- Positive sentiment and longer descriptions tend to increase engagement.
- Posting during peak hours results in higher interactions.
- Specific keywords (e.g., "exclusive", "free") are strong predictors of engagement.

### Business Impact
The model provides actionable insights that can help marketers optimize their social media strategies, focusing on content that is more likely to engage users. By aligning posting times with peak engagement periods and crafting posts with targeted keywords, brands can significantly enhance their online presence.

### Limitations
- The model is based on specific data and may not generalize across all social media platforms.
- Other engagement metrics like shares and saves were not included due to data limitations.

## Presentation of Findings
### Visualizations
- **Feature Importance:** Visualizations to highlight key predictors of engagement.
- **Predicted vs. Actual Engagement:** Plots to demonstrate model accuracy.

### Recommendations
- **Content Strategy:** Focus on positive, engaging descriptions with key phrases.
- **Posting Schedule:** Post during peak engagement times.
- **A/B Testing:** Use the model to test different content strategies.

### User Interface
- A simple interface where users can input post details and receive predictions.

## Conclusion
### Summary
This project successfully built and optimized a model to predict social media post engagement. The insights can be directly applied to enhance social media content strategies, leading to improved user interaction and brand visibility.

### Next Steps
- Extend the model to include additional engagement metrics like shares and saves.
- Test the model across different social media platforms for generalizability.

## Deliverables
- **GitHub Repository:** [https://github.com/DaniCoco31/Predicting-Social-Media-Post-Engagement/]
- **Presentation Slides:** [https://www.canva.com/design/DAGOnginV-4/jWEpQz7KxToZWa93cqtMoA/edit]

### Installation
To set up the project locally, clone the repository and install the required libraries:

```bash
git clone https://github.com/yourusername/social-media-engagement-prediction.git
cd social-media-engagement-prediction
pip install -r requirements.txt
