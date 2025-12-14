from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            
            # SVM doesn't support threshold tuning easily without probability=True
            # We keep it standard
            "SVM": SVC(kernel='rbf', probability=False, random_state=42),
            
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        
        print(f"\n{'Model':<20} | {'Accuracy':<10} | {'Recall':<10}")
        print("-" * 55)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            
            # --- THE TRICK: THRESHOLD TUNING ---
            # Standard prediction uses 0.5 (50%) threshold.
            # We lower it to 0.3 (30%) to catch more cancer cases.
            
            if hasattr(model, "predict_proba"):
                # Get probability of Cancer (Class 1)
                y_prob = model.predict_proba(X_test)[:, 1]
                # If prob > 0.30, predict Cancer (1)
                y_pred = (y_prob > 0.30).astype(int)
            else:
                # Fallback for SVM (if probability=False)
                y_pred = model.predict(X_test)
            # -----------------------------------
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            recall = report['1']['recall'] if '1' in report else 0
            
            results[name] = model
            print(f"{name:<20} | {acc:.2%}     | {recall:.2%} (Boosted)")
            print("-" * 55)
            
        return results