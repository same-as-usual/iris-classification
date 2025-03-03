🌸 Iris Flower Classification
A machine learning project using Scikit-Learn to classify Iris flowers into three species based on sepal and petal dimensions.



📌 Project Overview
This project implements a classification model to categorize iris flowers into three species:

Setosa
Versicolor
Virginica
The dataset used is the famous Iris dataset, which contains 150 samples with 4 features:

Sepal Length
Sepal Width
Petal Length
Petal Width
The model achieves 98% accuracy using Logistic Regression.

🚀 Technologies Used
Python 🐍
Scikit-Learn
Pandas
Matplotlib & Seaborn
📊 Data Visualization
To understand patterns in the dataset, various visualization techniques were implemented:

1️⃣ Pair Plot (Feature Correlation)
The pair plot shows the distribution of features for different species.

python
Copy
Edit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Pair plot
sns.pairplot(df, hue="species", diag_kind="kde", palette="husl")
plt.show()
📌 Output:

Insight:

Setosa is clearly separable from the other two species.
Versicolor & Virginica show some overlap but are still distinguishable.
2️⃣ Scatter Plot (Feature Relationships)
A scatter plot helps visualize the relationship between petal length and petal width.

python
Copy
Edit
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["petal length (cm)"], y=df["petal width (cm)"], hue=df["species"], palette="Dark2")
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()
📌 Output:

Insight:

Petal dimensions strongly determine the species.
Setosa is well-separated, while Versicolor and Virginica have slight overlap.
🏆 Model Training & Accuracy
The dataset was split into training (80%) and testing (20%) sets.
A Logistic Regression model was trained and achieved 98% accuracy.

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
📌 Output:

yaml
Copy
Edit
Model Accuracy: 98.00%
✅ Key Observations:

Logistic Regression is highly effective for this dataset.
The model can be further optimized using hyperparameter tuning.
📝 Usage
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/same-as-usual/iris-classification.git
cd iris-classification
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Model
bash
Copy
Edit
python iris_classification.py
🔥 Future Improvements
Implement Decision Trees & Random Forest for comparison.
Use Hyperparameter Tuning to improve accuracy.
Deploy the model using Flask / Streamlit for real-time predictions.
📌 Project Contributors
👨‍💻 Rajat Chattopadhyay

⭐ Support & Feedback
If you like this project, please consider starring ⭐ the repository! 😊
For suggestions, open an issue or pull request.
