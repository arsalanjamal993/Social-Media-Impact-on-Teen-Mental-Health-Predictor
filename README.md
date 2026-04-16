# Social Media Impact on Teen Mental Health Predictor

## Project Overview
This project implements a Machine Learning pipeline to analyze behavioral data and predict stress levels in teenagers. By evaluating factors such as social media usage hours, sleep quality, and physical activity, the model identifies patterns that contribute to mental health strain.

## Dataset Features
The model utilizes the following features from the `Teen_Mental_Health_Dataset.csv`:
- **Daily Social Media Hours:** Quantitative measure of usage.
- **Sleep Hours & Screen Time:** Indicators of physical recovery and digital habits.
- **Social Interaction Level:** Categorical (Low, Medium, High).
- **Platform Usage:** Specific apps (Instagram, TikTok, etc.).
- **Target Variable:** `stress_level` (Scaled 1–10).

## Implementation Details

### 1. Data Preprocessing
To ensure the mathematical models can process the data, the following steps were taken:
- **Ordinal Mapping:** `social_interaction_level` was manually mapped (Low: 1, Medium: 2, High: 3) to preserve the logical hierarchy.
- **One-Hot Encoding:** Categorical variables like `gender` and `platform_usage` were transformed using `pd.get_dummies` with `drop_first=True` to avoid the dummy variable trap.
- **Feature Selection:** Non-numeric string columns were removed to prevent `ValueError` during model training.

### 2. Modeling
A **Linear Regression** model was chosen as the baseline to establish the correlation between lifestyle factors and stress.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

## Performance Evaluation
The model is evaluated using two primary metrics:
- **R-Squared ($R^2$):** Measures how much variance in stress is explained by the features.
- **Mean Squared Error (MSE):** Quantifies the average squared difference between predicted and actual stress levels.

## How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas scikit-learn
   ```
3. Run the script:
   ```bash
   python predictor.py
   ```

## Future Roadmap
- Transition from Linear Regression to **Random Forest Regressor** for non-linear patterns.
- Implement a Neural Network using **TensorFlow/PyTorch** as part of the next learning phase.
- Create a visualization dashboard using **Matplotlib** and **Seaborn**.
