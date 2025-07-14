

# 🛍️ Global E-Commerce Customer Analytics

**Fraud Detection | Churn Prediction | Customer Segmentation**
🔍 *A Complete ML Workflow with 10 Classification Models, 3 Clustering Models, and Interactive Visualizations*

---

## 📌 Project Overview

This project explores a synthetic dataset simulating global e-commerce customer activity. The focus is on analyzing customer behavior, predicting fraudulent transactions, estimating churn risk, and segmenting customers using machine learning techniques.

It includes a full pipeline from **EDA → Feature Engineering → Modeling → Clustering → Visualization → Dashboard**.

---

## 🧠 Objectives

* 🔍 Perform Exploratory Data Analysis (EDA)
* 📊 Visualize customer patterns & behavior
* 🧠 Build **10 classification models** to detect fraudulent customers
* 💔 Predict customer **churn risk**
* 🎯 Apply **3 clustering models** for customer segmentation
* 📈 Create visual dashboards using Plotly for interactive insights

---

## 📂 Dataset Description

| Column               | Description                               |
| -------------------- | ----------------------------------------- |
| `customer_id`        | Unique ID for each customer               |
| `age`                | Age of the customer                       |
| `gender`             | Gender (Male/Female/Other)                |
| `country`            | Customer's country                        |
| `avg_order_value`    | Average order value in USD                |
| `total_orders`       | Total number of orders                    |
| `last_purchase`      | Days since last purchase                  |
| `is_fraudulent`      | Fraud flag (1 = Yes, 0 = No)              |
| `preferred_category` | Most purchased product category           |
| `email_open_rate`    | Percentage of emails opened               |
| `customer_since`     | Account creation date                     |
| `loyalty_score`      | Score (0–100) indicating customer loyalty |
| `churn_risk`         | Probability of customer churn (0–1)       |

> ⚠️ **Note:** This is synthetic data generated for educational purposes. No real user data is used.

---

## 📊 Exploratory Data Analysis (EDA)

* Distribution of age, gender, churn risk, loyalty score
* Missing value treatment & feature correlation
* Top countries, categories, and customer behaviors
* Email engagement vs loyalty & fraud
* Fraud rate comparison across demographic groups

---

## 🤖 Machine Learning Models

### 🔟 Classification Models (Target: `is_fraudulent`)

| Model                  |
| ---------------------- |
| Logistic Regression    |
| K-Nearest Neighbors    |
| Support Vector Machine |
| Decision Tree          |
| Random Forest          |
| XGBoost                |
| LightGBM               |
| AdaBoost               |
| Gradient Boosting      |
| Naive Bayes            |

**Metrics:** Accuracy, Precision, Recall, F1-score

### 🎯 Target Extension: Churn Prediction

Binary classification on `churn_risk` (thresholded)

---

## 🧩 Clustering Techniques (Customer Segmentation)

### ✅ Features Used:

* `avg_order_value`, `total_orders`, `loyalty_score`, `email_open_rate`, `churn_risk`

### 🔍 Clustering Models:

| Model             | Description                                   |
| ----------------- | --------------------------------------------- |
| **KMeans**        | Center-based customer segmentation            |
| **DBSCAN**        | Density-based outlier & fraud group detection |
| **Agglomerative** | Hierarchical group clustering                 |

🔸 **PCA** used for 2D visualization.

---

## 📈 Interactive Visualization (Plotly)

* Loyalty Score vs Churn Risk
* Fraud Rate by Country
* Cluster Analysis
* Top Product Categories
* Years with Company vs Loyalty



---

## 🛠️ Tools & Libraries

* `Python 3.10+`
* `Pandas`, `NumPy`
* `Matplotlib`, `Seaborn`, `Plotly`
* `Scikit-learn`, `XGBoost`, `LightGBM`
* `PCA`, `KMeans`, `DBSCAN`, `Agglomerative`


---


## ✅ Results & Takeaways

* 💡 High churn customers usually show lower loyalty & email engagement
* ⚠️ Fraudulent users have unique behavioral signatures (high inactivity, low loyalty)
* 🧠 Ensemble models (Random Forest, XGBoost, LightGBM) outperform basic classifiers
* 📊 Clustering reveals distinct customer segments useful for marketing strategies

---

## 🧪 Future Work

* ✅ Extend churn prediction to regression & survival models
* ✅ Deploy top model with Streamlit or Flask
* ✅ Integrate real-time fraud scoring
* ✅ AutoML for model selection and tuning

---

## 📚 License

This project uses **synthetic data** and is distributed under the MIT License.
Free to use for educational and research purposes.

---

## 🙋‍♂️ Author

**Arif Miah**
🎓 BSc in Computer Science & Engineering
🏆 3X Kaggle Expert | Machine Learning Engineer
🔗 [LinkedIn](www.linkedin.com/in/arif-miah-8751bb217) | [GitHub](https://github.com/Arif-miad) | [Kaggle](https://www.kaggle.com/code/miadul/e-commerce-customer-intelligence-ml-clustering) | [YouTube]([https://youtube.com](https://www.youtube.com/@intelliaiworld))

---

