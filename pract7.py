import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample dataset
np.random.seed(42)
n_samples = 1000

# Create two features
X = np.random.randn(n_samples, 2)
# Create non-linear decision boundary
y = (X[:, 0]**2 + X[:, 1]**2 > 2.5).astype(int)

def evaluate_classifier(y_true, y_pred, y_prob, model_name):
    """
    Evaluate classifier performance with multiple metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print classification report
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot confusion matrix
    plt.subplot(131)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name}\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot ROC curve
    plt.subplot(132)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    return accuracy, precision, recall, f1, roc_auc

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
print("\nLogistic Regression Analysis")
print("=" * 50)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
log_reg_pred = log_reg.predict(X_test_scaled)
log_reg_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate logistic regression
log_metrics = evaluate_classifier(y_test, log_reg_pred, log_reg_prob, "Logistic Regression")

# Decision Tree
print("\nDecision Tree Analysis")
print("=" * 50)
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)

# Make predictions
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]

# Evaluate decision tree
dt_metrics = evaluate_classifier(y_test, dt_pred, dt_prob, "Decision Tree")

# Visualize decision tree
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=[f'Feature {i+1}' for i in range(X.shape[1])],
          class_names=['Class 0', 'Class 1'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Compare models
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, log_metrics, width, label='Logistic Regression')
plt.bar(x + width/2, dt_metrics, width, label='Decision Tree')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.xticks(x, metrics, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance for Decision Tree
plt.figure(figsize=(8, 4))
importance = pd.DataFrame({
    'feature': [f'Feature {i+1}' for i in range(X.shape[1])],
    'importance': dt.feature_importances_
})
sns.barplot(data=importance, x='feature', y='importance')
plt.title('Feature Importance (Decision Tree)')
plt.show()
