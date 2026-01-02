import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, roc_curve
from numpy import random
import optuna
import shap
import matplotlib.pyplot as plt
import pandas as pd
random.seed(seed=12)
from myRF import read_data
from PRF.PRF import RandomForestClassifier as prf


def calculate_class_accuracy(true_labels, predicted_labels, class_label):
    total_instances = 0
    correct_predictions = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == class_label:
            total_instances += 1
            if predicted_label == class_label:
                correct_predictions += 1

    if total_instances == 0:
        return 0.0
    else:
        accuracy = correct_predictions / total_instances
        return accuracy

def threeClass(thresh_oob, test_probs):
    scores = []
    for id in range(len(test_probs)):
        if np.argmax(test_probs[id]) == 0:
            scores.append(0)
        else:
            val = 1 if  test_probs[id, 1] >= (thresh_oob+1)*test_probs[id, 2] else 2
            scores.append(val)

    return scores
def getprecision(true_labels, predicted_labels, pos_label):
    # Create binary labels for the chosen class
    binary_true_labels = (true_labels == pos_label)
    binary_predicted_labels = (np.array(predicted_labels) == pos_label)

    # Compute precision score for the chosen class
    precision = recall_score(binary_true_labels, binary_predicted_labels)
    return precision

def optimize_threshold_from_oob_predictions(labels_train, oob_probs, thresholds, ThOpt_metrics='Kappa'):
    if ThOpt_metrics == 'Kappa':
        tscores = []
        for thresh in thresholds:
            scores = threeClass(thresh, oob_probs)
            kappa = cohen_kappa_score(labels_train, scores, weights='quadratic')
            _precision = getprecision(labels_train, scores, pos_label=2)
            tscores.append((np.round(kappa*0.8, 6)+_precision, thresh))
        tscores.sort(reverse=True)
        thresh = tscores[0][-1]
    return thresh

def threshold_tune(opt_oob=True, cls=None, labels_train=None, test_probs=None, labels_test=None,
                   thresholds=None, ThOpt_metrics='Kappa', cv=0):
    if opt_oob and hasattr(cls, 'oob_decision_function_'):
        oob_probs = cls.oob_decision_function_
        thresh_oob = optimize_threshold_from_oob_predictions(labels_train, oob_probs, thresholds, ThOpt_metrics)
        scores = threeClass(thresh_oob, test_probs)
        return scores, thresh_oob
    else:
        print("OOB optimization not available or disabled.")
        scores = threeClass(thresholds[0], test_probs)
        return scores, thresholds[0]

def confusion_res(scores, targets):
    targets = targets.reshape(-1)
    CM = confusion_matrix(targets, scores).astype(np.float32)
    total = np.sum(CM, axis=0)
    prob = CM / total
    return CM, prob


def objective(trial, X, y, method, kfold=3):
    # Define hyperparameter search space based on method
    if method == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 15, 25),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            # 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
    elif method == 'PRF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 15, 25),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            # 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'keep_proba': trial.suggest_float('keep_proba', 0.3, 0.7),
            'new_syn_data_frac': trial.suggest_float('new_syn_data_frac', 0.05, 0.3)
        }
    elif method == 'SVM':
        params = {
            'C': trial.suggest_float('C', 0.01, 1, log=True),
            # 'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
    elif method == 'logistic':
        params = {
            'C': trial.suggest_float('C', 0.01, 1, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'max_iter': 1000  # Fixed to ensure convergence
        }
    elif method == 'ensemble':
        params = {
            'rf_n_estimators': trial.suggest_int('rf_n_estimators', 15, 25),
            'rf_max_depth': trial.suggest_int('rf_max_depth', 5, 15),
            'prf_n_estimators': trial.suggest_int('prf_n_estimators', 15, 25),
            'prf_max_depth': trial.suggest_int('prf_max_depth', 5, 15),
            'keep_proba': trial.suggest_float('keep_proba', 0.3, 0.7),
            'new_syn_data_frac': trial.suggest_float('new_syn_data_frac', 0.05, 0.3),
            'threshold': trial.suggest_float('threshold', 0.0, 0.44, step=0.02)
        }

    skf = StratifiedKFold(n_splits=kfold)
    accuracies = []
    class_accuracies = {0: [], 1: [], 2: []}  # For three classes

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        under = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = under.fit_resample(X_train, y_train)

        if method == 'RF':
            dt_clasi = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                # min_samples_split=params['min_samples_split'],
                # min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                class_weight="balanced",
                oob_score=True
            )
        elif method == 'PRF':
            dt_clasi = prf(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                bootstrap=True,
                keep_proba=params['keep_proba'],
                new_syn_data_frac=params['new_syn_data_frac']
            )
        elif method == 'SVM':
            dt_clasi = SVC(
                C=params['C'],
                # kernel=params['kernel'],
                gamma=params['gamma'],
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif method == 'logistic':
            dt_clasi = LogisticRegression(
                C=params['C'],
                solver=params['solver'],
                max_iter=params['max_iter'],
                random_state=42,
                class_weight='balanced'
            )
        elif method == 'ensemble':
            dt_clasi1 = RandomForestClassifier(
                n_estimators=params['rf_n_estimators'],
                max_depth=params['rf_max_depth'],
                random_state=42,
                class_weight="balanced",
                oob_score=True
            )
            dt_clasi2 = prf(
                n_estimators=params['prf_n_estimators'],
                max_depth=params['prf_max_depth'],
                bootstrap=True,
                keep_proba=params['keep_proba'],
                new_syn_data_frac=params['new_syn_data_frac']
            )
            dt_clasi1.fit(X=X_train_res, y=y_train_res)
            dt_clasi2.fit(X=X_train_res, y=y_train_res)
            y_pred1 = dt_clasi1.predict_proba(X_test)
            y_pred2 = dt_clasi2.predict_proba(X_test)
            y_pred = (0.5 * y_pred1 + y_pred2) / 2
            y_pred = threeClass(params['threshold'], y_pred)

        else:
            dt_clasi = None
            print('this is no such method...')
            return 0
        if method != 'ensemble':
            dt_clasi.fit(X=X_train, y=y_train)
            y_pred = dt_clasi.predict(X_test)

        # Calculate overall accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Calculate per-class accuracy
        for class_label in [0, 1, 2]:
            class_acc = calculate_class_accuracy(y_test, y_pred, class_label)
            class_accuracies[class_label].append(class_acc)

    # Calculate mean accuracies
    mean_accuracy = np.mean(accuracies)
    mean_class_accuracies = {
        f'class_{class_label}_accuracy': np.mean(acc_list)
        for class_label, acc_list in class_accuracies.items()
    }

    # Print class-wise accuracies for this trial
    print(f"Trial {trial.number}:")
    print(f"Overall accuracy: {mean_accuracy:.4f}")
    for class_label, acc in mean_class_accuracies.items():
        print(f"{class_label}: {acc:.4f}")

    return mean_accuracy


def run_RF_PRF(X=None, y=None, method=None, n_trials=20):
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, method, kfold=Kfold), n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    print("Best parameters:", best_params)
    print("Best accuracy:", study.best_value)

    # Run final model with best parameters
    skf = StratifiedKFold(n_splits=Kfold)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        if method == 'RF':
            dt_clasi = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                random_state=42,
                class_weight="balanced",
                oob_score=True
            )
        elif method == 'PRF':
            dt_clasi = prf(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                bootstrap=True,
                keep_proba=best_params['keep_proba'],
                new_syn_data_frac=best_params['new_syn_data_frac']
            )
        elif method == 'SVM':
            dt_clasi = SVC(
                C=best_params['C'],
                # kernel=best_params['kernel'],
                gamma=best_params['gamma'],
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif method == 'logistic':
            dt_clasi = LogisticRegression(
                C=best_params['C'],
                solver=best_params['solver'],
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            dt_clasi = None
            print('this is no such method...')

        dt_clasi.fit(X=X_train, y=y_train)
        y_pred = dt_clasi.predict(X_test)

        # Calculate and print class-wise accuracies
        print(f"\nFold {i + 1} Class-wise Accuracies:")
        for class_label in [0, 1, 2]:
            class_acc = calculate_class_accuracy(y_test, y_pred, class_label)
            print(f"Class {class_label} accuracy: {class_acc:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)


def run_ensemblerune(X=None, y=None, method=None, n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, method='ensemble', kfold=Kfold), n_trials=n_trials)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print("Best accuracy:", study.best_value)

    skf = StratifiedKFold(n_splits=Kfold)
    thresholds = np.round(np.arange(0.0, 0.44, 0.02), 2)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        under = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = under.fit_resample(X_train, y_train)

        dt_clasi1 = RandomForestClassifier(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            random_state=42,
            class_weight="balanced",
            oob_score=True
        )
        dt_clasi2 = prf(
            n_estimators=best_params['prf_n_estimators'],
            max_depth=best_params['prf_max_depth'],
            bootstrap=True,
            keep_proba=best_params['keep_proba'],
            new_syn_data_frac=best_params['new_syn_data_frac']
        )

        dt_clasi1.fit(X=X_train_res, y=y_train_res)
        dt_clasi2.fit(X=X_train_res, y=y_train_res)
        y_pred1 = dt_clasi1.predict_proba(X_test)
        y_pred2 = dt_clasi2.predict_proba(X_test)
        y_pred = (0.5 * y_pred1 + y_pred2) / 2

        y_pred, opt_threshold = threshold_tune(
            opt_oob=True,
            cls=dt_clasi1,
            labels_train=y_train_res,
            test_probs=y_pred,
            labels_test=y_test,
            thresholds=thresholds,
            ThOpt_metrics='Kappa',
            cv=i
        )

        print(f"\nFold {i + 1} Class-wise Accuracies:")
        for class_label in [0, 1, 2]:
            class_acc = calculate_class_accuracy(y_test, y_pred, class_label)
            print(f"Class {class_label} accuracy: {class_acc:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)

def run_ensemblerune_shap(X=None, y=None, method=None, n_trials=20, feature_names=None):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, method='ensemble', kfold=Kfold), n_trials=n_trials)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print("Best accuracy:", study.best_value)

    skf = StratifiedKFold(n_splits=Kfold)
    thresholds = np.round(np.arange(0.0, 0.45, 0.02), 2)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = np.array(y[train_index])
        y_test = np.array(y[test_index])

        under = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = under.fit_resample(X_train, y_train)

        dt_cl = RandomForestClassifier(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            random_state=42,
            class_weight="balanced",
            oob_score=True
        )
        dt_clasi2 = prf(
            n_estimators=best_params['prf_n_estimators'],
            max_depth=best_params['prf_max_depth'],
            bootstrap=True,
            keep_proba=best_params['keep_proba'],
            new_syn_data_frac=best_params['new_syn_data_frac']
        )

        dt_cl.fit(X=X_train_res, y=y_train_res)
        dt_clasi2.fit(X=X_train_res, y=y_train_res)
        y_pred1 = dt_cl.predict_proba(X_test)
        y_pred2 = dt_clasi2.predict_proba(X_test)

        # Optimize threshold using OOB predictions
        y_pred, opt_threshold = threshold_tune(
            opt_oob=True,
            cls=dt_cl,
            labels_train=y_train_res,
            test_probs=(0.5 * y_pred1 + y_pred2) / 2,  # Average probabilities
            labels_test=y_test,
            thresholds=thresholds,
            ThOpt_metrics='Kappa',
            cv=i
        )

        print(f"\nFold {i + 1} Class-wise Accuracies:")
        for class_label in [0, 1, 2]:
            class_acc = calculate_class_accuracy(y_test, y_pred, class_label)
            print(f"Class {class_label} accuracy: {class_acc:.4f}")

        # SHAP Analysis with thresholded output
        print(f"\nComputing SHAP values for Fold {i + 1}...")

        # Custom prediction function incorporating threeClass with optimized threshold
        def ensemble_predict_proba(X):
            X = np.atleast_2d(X)
            prob1 = dt_cl.predict_proba(X)
            prob2 = dt_clasi2.predict_proba(X)
            avg_probs = (0.5 * prob1 + prob2) / 2
            # Apply threeClass with optimized threshold
            scores = []
            for id in range(len(avg_probs)):
                if np.argmax(avg_probs[id]) == 0:
                    scores.append(0)
                else:
                    val = 1 if avg_probs[id, 1] >= (opt_threshold + 1) * avg_probs[id, 2] else 2
                    scores.append(val)
            return np.array(scores)  # Return class assignments

        background = shap.sample(X_train_res, 100, random_state=42)
        explainer = shap.KernelExplainer(
            ensemble_predict_proba,
            background,
            link="identity"  # Use identity link since output is class labels, not probabilities
        )
        shap_values = explainer.shap_values(X_test, nsamples=100)

        # Summary plot for Pathological class (Class 2)
        shap.summary_plot(shap_values[2], X_test, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=10)
        plt.title(f"SHAP Feature Importance for Pathological Class (Fold {i + 1})")
        plt.tight_layout()
        plt.savefig(f"shap_summary_fold_{i + 1}.png")
        plt.close()

        # Force plot for the first test instance (Pathological class)
        shap.force_plot(explainer.expected_value[2], shap_values[2][0], X_test[0],
                        feature_names=feature_names, show=False)
        plt.savefig(f"shap_force_fold_{i + 1}_sample.png")
        plt.close()
        break



def shap_plot(X, y):
    dataset = pd.read_csv(r'signalfeat.csv')  # Get the data
    feature_names = dataset.columns[:-1]
    # feature_names = [f"Feature_{i}" for i in range(X.shape[1])]  # Placeholder
    # If read_data() returns a DataFrame, use: feature_names = X.columns
    run_ensemblerune_shap(X=X, y=y, method='ensemble', n_trials=10, feature_names=feature_names)



if __name__ == "__main__":
    Kfold = 3
    X, y = read_data()
    X = np.array(X)
    y = np.array(y)
    # print('This is RF')
    # run_RF_PRF(X=X, y=y, method='RF', n_trials=10)
    # print('This is SVM')
    # run_RF_PRF(X=X, y=y, method='SVM', n_trials=10)
    # print('This is logistic')
    # run_RF_PRF(X=X, y=y, method='logistic', n_trials=10)
    # print('This is PRF')
    # run_RF_PRF(X=X, y=y, method='PRF', n_trials=10)
    # print('This is resemble')
    # run_ensemblerune(X=X, y=y, method='resemble', n_trials=10)
    shap_plot(X, y)


