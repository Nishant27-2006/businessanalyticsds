# Data Science Salary Prediction

This repository contains Python scripts and resources used for predicting data science salaries using various machine learning models. The dataset used in this project is publicly available on Kaggle and provides a comprehensive overview of salary data for various roles in the data science field.

## Dataset

The dataset used in this project is the "Data Science Job Salaries" dataset, which is available on Kaggle. It includes a wide range of features, such as job titles, locations, company sizes, and other relevant factors, making it ideal for building robust predictive models.

**Kaggle Dataset Citation:**
- J, Shan (2021). Data Science Job Salaries. Kaggle. Retrieved August 14, 2024, from [https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)



## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Nishant27-2006/businessanalyticsds
   cd your-repo-name
Install Dependencies:
Make sure you have Python installed. Install the required Python packages:

pip install -r requirements.txt
Run the Scripts:
You can run each of the scripts to train the respective machine learning models:

python decision_tree.py
python grad_boost.py
python linear_regression.py
python svm.py
Each script will generate results, model files, and figures related to the performance of the respective model.

Evaluation
The project evaluates the performance of the following machine learning models:

Decision Tree
Gradient Boosting
Linear Regression
Support Vector Machine (SVM)
The performance of each model is visualized through scatter plots comparing predicted vs actual salaries and residuals distributions. The results are saved as .txt files and figures are saved as .png images in their respective directories.

Figures
The repository includes the following figures:

Figure 1: Decision Tree - Predicted vs Actual Salaries
Figure 2: Decision Tree - Residuals Distribution
Figure 3: Gradient Boosting - Predicted vs Actual Salaries
Figure 4: Gradient Boosting - Residuals Distribution
Figure 5: Linear Regression - Predicted vs Actual Salaries
Figure 6: Linear Regression - Residuals Distribution
Figure 7: SVM - Predicted vs Actual Salaries
Figure 8: SVM - Residuals Distribution
Results
The results for each model are stored in the results directory and include metrics like Mean Squared Error (MSE) and other evaluation statistics.

Conclusion
This project provides insights into the predictability of data science salaries using various machine learning techniques. The dataset and scripts can be easily adapted for other salary prediction tasks in different industries.

Contact
For any questions or suggestions, feel free to contact me at nishantg2706@gmail.com.
