# Machine Learning Applied to Credit Risk Analysis

This project aims to analyze the applicability of Machine Learning techniques in credit risk analysis, with a particular focus on the selection of relevant variables and the optimization of predictive models. The literature review highlights that these techniques outperform traditional methods in risk prediction and loss reduction. Additionally, there is a growing interest in the applicability of unconventional approaches based on Deep Learning methods, which, together with non-financial alternative data, promote financial inclusion. As methodology, two optimization algorithms are proposed: Local Search (BL) and Cross Entropy (CE), combined with various Machine Learning techniques to enhance the process of identifying optimal combinations of variables, focusing on the most influential ones in credit risk prediction. The results show that data preprocessing, including value imputation and outlier removal, is crucial to improving model accuracy. The comparative analysis between BL and CE suggests that both methods improve model performance, although CE offers greater flexibility in selecting additional variables, leading to further optimization improvements. Moreover, it was found that longer loan payment delays are key predictors in credit risk analysis.

---

## Data

The dataset used for this project comes from Kaggle's popular "Give Me Some Credit" competition. Files labeled _original correspond to the original datasets from the Kaggle competition. Files labeled _modified are the preprocessed versions, generated with the Exploratory-Data-Analysis.ipynb script.

| Variable                                   | Description                                                                                                                      | Type                |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|---------------------|
| SeriousDlqin2yrs                           | Borrower is delinquent or overdue by 90 days or more on a payment.                                                               | Quantitative, Binary (Tarjet)|
| RevolvingUtilizationOfUnsecuredLines        | Total balance on credit cards and personal lines of credit (excluding real estate and installment debt) divided by credit limits. | Quantitative, Continuous |
| Age                                        | Age of the borrower in years.                                                                                                    | Quantitative, Discrete   |
| NumberOfTime30-59DaysPastDueNotWorse       | Number of times borrower has been 30-59 days past due but no worse in the last 2 years.                                          | Quantitative, Discrete   |
| DebtRatio                                  | Monthly debt payments, alimony, and living costs divided by gross monthly income.                                                | Quantitative, Continuous |
| MonthlyIncome                              | Borrower's monthly income.                                                                                                       | Quantitative, Continuous |
| NumberOfOpenCreditLinesAndLoans            | Number of open installment loans (e.g., car loans, mortgages) and lines of credit (e.g., credit cards).                         | Quantitative, Discrete   |
| NumberOfTimes90DaysLate                    | Number of times borrower has been 90 days or more past due.                                                                      | Quantitative, Discrete   |
| NumberRealEstateLoansOrLines               | Number of real estate loans or lines, including home equity lines of credit.                                                     | Quantitative, Discrete   |
| NumberOfTime60-89DaysPastDueNotWorse       | Number of times borrower has been 60-89 days past due but no worse in the last 2 years.                                          | Quantitative, Discrete   |
| NumberOfDependents                         | Number of dependents in the family, excluding the borrower (e.g., spouse, children, etc.).                                      | Quantitative, Discrete   |

*Source: Own elaboration based on the “Give Me Some Credit” dataset.*

---

## Repository Structure

| File/Folder                                      | Description                                                                                  |
|--------------------------------------------------|----------------------------------------------------------------------------------------------|
| `Exploratory-Data-Analysis.ipynb`                | Jupyter notebook for initial data exploration, cleaning, and visualization.                  |
| `CE models.py`                                   | Python script for feature selection using various ML models optimized with Cross Entropy.     |
| `BL models.py`                                   | Python script for feature selection using various ML models optimized with Local Search.      |
| `Research/`                                      | Folder containing the full research report (Machine_Learning_Applied_to_Credit_Risk_Analysis.pdf) and presentation to his defense (Thesis_Presentation.pptx).   |
| `Data/`                                          | Folder containing original and processed datasets of Give Me Some Credit                         |  
| `README.md`                                      | This project description and instructions.                                                   |
| `LICENSE`                                        | License for the code in this repository.                                                     |

---

## Research Paper and Presentation

The folder `Research/` contains:
- `Machine_Learning_Applied_to_Credit_Risk_Analysis.pdf`: The full research report.
- `Thesis_Presentation.pptx`: The slides used for the defense and final presentation of the research project to the academic authorities.

**License:**  
These documents are © David Felipe Vargas Cadena, 2025.  
They are licensed under a [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.  
You may share these documents with attribution, but you may not use them commercially or create derivative works.

---

## Contact

For questions or collaboration, please contact:  
**David Felipe Vargas Cadena**  
https://www.linkedin.com/in/davidfvargasc/  |  https://github.com/eldeivin/

