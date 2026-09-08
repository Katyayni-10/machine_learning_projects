# machine_learning_projects
# My Machine Learning Projects

A collection of end-to-end ML projects covering supervised, unsupervised, and reinforcement learning. Every project includes the dataset, a trained model (`.pkl`), and an interactive **Streamlit** app (`app.py`) to demo predictions.

## 📁 Repository Structure

```
my-ml-projects/
├── breast_cancer_prediction/
│   ├── data/               # dataset
│   ├── model.pkl           # trained model
│   └── app.py              # streamlit app
├── california_housing/
├── customer_churn_prediction/
├── diabetes_prediction/
├── dry_beans/
├── fish_weight/
├── loan_approval/
├── mall_customers/
├── medical_cost/
├── salary_prediction/
├── taxi_reinforcement/
├── wine/
├── wine_classification/
└── README.md               # you are here
```

---

## 🧠 Supervised Learning

| Project | Task | Algorithm | Tech Stack |
|---|---|---|---|
| [breast_cancer_prediction](./breast_cancer_prediction) | Classification | Support Vector Machine (SVM) | scikit-learn, pandas, Streamlit |
| [california_housing](./california_housing) | Regression | Random Forest Regressor | scikit-learn, pandas, Streamlit |
| [customer_churn_prediction](./customer_churn_prediction) | Classification | Random Forest Classifier | scikit-learn, pandas, Streamlit |
| [diabetes_prediction](./diabetes_prediction) | Classification | Logistic Regression | scikit-learn, pandas, Streamlit |
| [dry_beans](./dry_beans) | Classification | K-Nearest Neighbors (KNN) | scikit-learn, pandas, Streamlit |
| [fish_weight](./fish_weight) | Regression | K-Nearest Neighbors (KNN) | scikit-learn, pandas, Streamlit |
| [loan_approval](./loan_approval) | Classification | Decision Tree | scikit-learn, pandas, Streamlit |
| [medical_cost](./medical_cost) | Regression | Decision Tree Regressor | scikit-learn, pandas, Streamlit |
| [salary_prediction](./salary_prediction) | Regression | Linear Regression | scikit-learn, pandas, Streamlit |
| [wine_classification](./wine_classification) | Classification | Support Vector Classifier (SVC) | scikit-learn, pandas, Streamlit |

## 🔍 Unsupervised Learning

| Project | Task | Algorithm | Tech Stack |
|---|---|---|---|
| [mall_customers](./mall_customers) | Clustering | K-Means | scikit-learn, pandas, Streamlit |
| [wine](./wine) | Dimensionality Reduction | Principal Component Analysis (PCA) | scikit-learn, pandas, Streamlit |

## 🎮 Reinforcement Learning

| Project | Task | Algorithm | Tech Stack |
|---|---|---|---|
| [taxi_reinforcement](./taxi_reinforcement) | Sequential Decision Making | Reinforcement Learning (Q-learning) | Gymnasium, NumPy, Streamlit |

---

## 🛠️ How to Run Any Project

Each folder is self-contained — dataset, trained `.pkl` model, and a Streamlit app.

```bash
cd project-name/
pip install -r requirements.txt
streamlit run app.py
```

This will launch a local interactive demo where you can input feature values and get live predictions from the trained model.

## 📌 Project Folder Convention

Every project folder follows the same layout:

```
project-name/
├── data/            # raw / processed dataset used for training
├── model.pkl         # serialized trained model
├── app.py             # Streamlit app for interactive predictions
```

## 📫 Contact

Feel free to reach out via [your email / LinkedIn / GitHub profile link] for questions or collaboration.
