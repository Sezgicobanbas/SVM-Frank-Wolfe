import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# create a SVM class using soft margin SVM
class SVM: 
    def __init__(self, C, method="fw", iteration=100):

        self.C = C
        self.method = method
        self.iteration = iteration

        self.W = None
        self.b = None

    def compute_dual_lagrangian(self, X, y, alpha):
        l_dual = 0.5 * (np.matmul(y, y.T) * np.matmul(alpha, alpha.T) * np.matmul(X, X.T)).sum() + (
                1 / (2 * self.C)) * np.sum(alpha * alpha)

        return l_dual
## sometimes its hard to minimize primal problem thats why we create a dual problem, 
# bc dual problem has a sparsity 
## it is easy to memorize and computationally efficient 
## we used that for checking loss function inside the frank wolfe algorithm  
# and away step FW  and pairwise FW algorithm
## to check loss  and evaluate the exact line search (step6 line search)
## 
## function calculates the dual form of the Lagrangian for an SVM, which is 
# used in the optimization process to find the optimal sigma
#The function combines the kernel matrix computation and the regularization term 
# to produce the final value of the dual.

    def compute_gradient(self, X, y, alpha):

        gradients = y * np.matmul(alpha * y, np.matmul(X, X.T)).T + (alpha / self.C)

        return gradients
#  function calculates the gradient of the dual Lagrangian with respect to alpha in SVM 
    def linear_minimization_oracle(self, X, y, alpha):

        s = np.zeros(len(y))
        gradient = self.compute_gradient(X, y, alpha)

        i = np.argmin(gradient)
        s[i] = 1

        return s
##  It is used to solve constrained optimization problems by finding 
# a feasible direction to move toward at each iteration
## it used at all  Frank wolfe algorithm to find st 
## 

### LINE SEARCH 
            gamma_search = np.linspace(0, gamma_max, 15)  # Perform line search over 15 points
            gamma_t = 0
            min_loss = self.compute_dual_lagrangian(X, y, alpha)
            for gamma_iter in gamma_search:
                candidate_alpha = alpha + gamma_iter * d_t
                candidate_loss = self.compute_dual_lagrangian(X, y, candidate_alpha)
                if candidate_loss < min_loss:
                    min_loss = candidate_loss
                    gamma_t = gamma_iter

# USED  to find the optimal step size along the given direction dt to minimize the 
# objective function. Goal is to find step size in the greatest reduction in the
#objective function 
# # used in away step frank wolfe and paiewise frank wolfe when we do not use the k/k+2 
#  which is   fixed step size we used as standard frank wolfe algorithm 


    def frank_wolfe(self, X, y):
        history = {}
        n = len(y)
        # Normalize the array so that its elements sum to 1
        alpha = np.random.rand(n)
        alpha /= np.sum(alpha)

        for k in range(0, self.iteration):
            gamma = 4 / (k + 2)
            s = self.linear_minimization_oracle(X, y, alpha)
            alpha = (1 - gamma) * alpha + gamma * s

            if k % 5 == 0:
                train_loss = self.compute_dual_lagrangian(X, y, alpha)
                history[k] = train_loss

        return alpha, history

    def away_step_frank_wolfe(self, X, y, epsilon=1e-6):
        history = {}
        n = len(y)

        # Step 1: Initialize the alpha , alpha  is the purpuse of the algorithm
        #  we are tying to find alpha  
        alpha = np.random.rand(len(y))
        alpha /= np.sum(alpha) # Feasible initial solution
        active_set = {tuple(alpha.copy())}  # Ensure at least two distinct atoms
        # active set is for tracking the st we find in each iteration 
        # AFW uses the active set to perform away steps, which move away from the
        #  bad vertex in the current solution. 

        for t in range(self.iteration):
            #print(f"Iteration {t}: Active Set Size = {len(active_set)}")  # Debugging

            # Step 2: Compute gradient
            gradient = self.compute_gradient(X, y, alpha)

            # Step 3: Find best and worst vertex
            s_t = self.linear_minimization_oracle(X, y, alpha)  # Best vertex 
            d_fw_t = s_t - alpha  # FW direction
            # we are looking at the direction for FW 

            worst_idx = np.argmax([np.dot(gradient, np.array(v)) for v in active_set])
            v_t = np.array(list(active_set)[worst_idx])  # Worst vertex
            d_away_t = alpha - v_t  # Away direction
            # this is use  for finding the away direction, it use active set to 
            # find the point 


            # Step 4: Check Frank-Wolfe gap
            g_fw_t = -np.dot(gradient, d_fw_t)
            if g_fw_t <= epsilon:
                return alpha, history
            # if the gap is smaller enough we can get out from the  function 


            # Step 5: Choose direction (FW or Away)
            if g_fw_t >= -np.dot(gradient, d_away_t):
                d_t = d_fw_t  # Frank-Wolfe step
                gamma_max = 1
            else:
                d_t = d_away_t  # Away step
                gamma_max = min(alpha[worst_idx] / (1 - alpha[worst_idx] + 1e-10), 1) 
            # decide which direction we are going 

            # Step 6: Line Search
            gamma_search = np.linspace(0, gamma_max, 15) # gamma max is equal to 1 and we 
                                                #took 15 number in exact distance  from 0 to 1 
            gamma_t = 0
            min_loss = self.compute_dual_lagrangian(X, y, alpha)
            for gamma_iter in gamma_search:
                candidate_alpha = alpha + gamma_iter * d_t
                candidate_loss = self.compute_dual_lagrangian(X, y, candidate_alpha)
                if candidate_loss < min_loss:
                    min_loss = candidate_loss
                    gamma_t = gamma_iter  # Update gamma_t properly

            # Ensure gamma_t is always nonzero
            if gamma_t == 0:
                gamma_t = gamma_max * 0.01  # Take a small step instead of staying stuck
            # bc of the dataset we used 
            # in the away step FW it was not updating the loss thats why we put this code line 
            # in every iteration gamma t was 0 bc it was not less than min loss 

            # Step 7: Update alpha
            alpha = alpha + gamma_t * d_t

            # Step 8: Active Set Update
            if tuple(s_t) not in active_set:
                active_set.add(tuple(s_t)) # Ensure new vertex is added
            
            if gamma_t == gamma_max in active_set:  # Ensure safe removal
                active_set.remove(tuple(v_t))
                
            # If the step size equals tto maximum possible value the weights of
            #  vt (worst vertex) drops to zero  
            # and it should be removed from the active set. this is called a drop step 

            # Track convergence
            if t % 5 == 0:
                train_loss = self.compute_dual_lagrangian(X, y, alpha)
                history[t] = train_loss

        return alpha, history

    def pairwise_frank_wolfe(self, X, y, epsilon=1e-6):
        history = {}
        n = len(y)

        # Step 1: Initialize
        alpha = np.random.rand(len(y))
        alpha /= np.sum(alpha)  # Normalize to make the initial point feasible
        active_set = {tuple(alpha.copy())}  # Store initial active set as a set of tuples
        ## 
        for t in range(self.iteration):
            # Step 2: Compute gradient
            gradient = self.compute_gradient(X, y, alpha)

            # Step 3: Linear Minimization Oracle (LMO)
            s_t = self.linear_minimization_oracle(X, y, alpha)  # Find best vertex \( s_t \)

            # Step 4: Select worst vertex \( v_t \)
            worst_idx = np.argmax([np.dot(gradient, np.array(v)) for v in active_set])
            v_t = np.array(list(active_set)[worst_idx])  # Convert tuple back to NumPy array

            # Step 5: Pairwise Direction
            d_t = s_t - v_t  # Pairwise direction


            # Step 6: Compute Maximum Step Size
            gamma_max = alpha[worst_idx]  # Maximum feasible step size for pairwise direction

            # Stopping Criterion (Frank-Wolfe Gap)
            g_fw_t = -np.dot(gradient, d_t)
            if g_fw_t <= epsilon:
                return alpha, history

            # Step 7: Line Search (to find the gamma )
            gamma_search = np.linspace(0, gamma_max, 15)  # Perform line search over 15 points
            gamma_t = 0
            min_loss = self.compute_dual_lagrangian(X, y, alpha)
            for gamma_iter in gamma_search:
                candidate_alpha = alpha + gamma_iter * d_t
                candidate_loss = self.compute_dual_lagrangian(X, y, candidate_alpha)
                if candidate_loss < min_loss:
                    min_loss = candidate_loss
                    gamma_t = gamma_iter



            # Step 8: Update alpha
            alpha = alpha + gamma_t * d_t

            # Step 9: Update Active Set
            if gamma_t == gamma_max:  # Ensure removal of worst atom
                active_set.remove(tuple(v_t))

            active_set.add(tuple(s_t))  # Ensure we add the new best atom

            # Track convergence every 5 iterations
            if t % 5 == 0:
                train_loss = self.compute_dual_lagrangian(X, y, alpha)
                history[t] = train_loss

        return alpha, history

    def fit(self, X, y):
        if self.method == "fw":
            alpha, history = self.frank_wolfe(X, y)
        elif self.method == "afw":
            alpha, history = self.away_step_frank_wolfe(X, y)
        elif self.method == "pfw":
            alpha, history = self.pairwise_frank_wolfe(X, y)
        else:
            raise ValueError("Invalid method. Choose 'fw', 'afw', or 'pfw'.")

        self.W = np.dot(alpha.T * y, X) ## coming from the KKT conditions, Soft margin SVM 
        chi = alpha / self.C  # slack variable 
        b = (1 - chi) / y - np.dot(self.W, X.T)  # bias 
        self.b = np.mean(b[alpha > 0]) if np.any(alpha > 0) else 0

        return history, alpha

    def predict(self, X):
        y_hat = np.sign(np.dot(self.W, X.T) + np.full(X.shape[0], self.b))

        return np.array(y_hat)


def preprocess_and_train(file_path, target_column, model, test_size, random_state):
    data = pd.read_csv(file_path)

    # Separate features (X) and labels (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert y_train and y_test to NumPy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Track Training Time
    start_time = time.time()
    history, alpha = model.fit(X_train, y_train) # fit the model
    training_time = time.time() - start_time

    # Make predictions
    y_predicted = model.predict(X_test)
    report = classification_report(y_test, y_predicted, output_dict=True)

    return {
        "history": history,  # For convergence comparison
        "accuracy": report["accuracy"],
        "F1": report['macro avg']['f1-score'],
        "training_time": training_time  # Store training time
    }


import matplotlib.pyplot as plt

def plot_accuracy_and_time(results):
        datasets = ["is_dead", "is_cancer", "is_exited"]
        methods = ["FW", "AFW", "PFW"]

        # Initialize figure with 2 rows (one for accuracy, one for time) & 3 columns (one per dataset)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for i, dataset in enumerate(datasets):
            accuracies = []
            times = []

            for method in methods:
                key = f"{method}_{dataset}"
                accuracies.append(results[key]["accuracy"])
                times.append(results[key]["training_time"])

            # Plot Accuracy Comparison
            axes[0, i].plot(methods, accuracies, marker="o", linestyle="-", label="Accuracy")
            axes[0, i].set_title(f"Accuracy Comparison - {dataset}")
            axes[0, i].set_xlabel("Method")
            axes[0, i].set_ylabel("Accuracy")
            axes[0, i].grid(True)

            # Plot Training Time Comparison
            axes[1, i].plot(methods, times, marker="o", linestyle="-", color="r", label="Training Time")
            axes[1, i].set_title(f"Training Time Comparison - {dataset}")
            axes[1, i].set_xlabel("Method")
            axes[1, i].set_ylabel("Time (seconds)")
            axes[1, i].grid(True)

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Initialize models
    fw_SVM = SVM(C=1, method="fw", iteration=100)  # Standard Frank-Wolfe
    afw_SVM = SVM(C=0.5, method="afw", iteration=100)  # Away-Step Frank-Wolfe
    pfw_SVM = SVM(C=1, method="pfw", iteration=100)  # Pairwise Frank-Wolfe
    # we did not use fine tunning to find optimal C but tried each possible numbers and take the best ones 
    # Datasets and target columns
    datasets = [
        {"file_path": "Data_final/heart_failure.csv", "target_column": "is_dead"},
        {"file_path": "Data_final/breast_cancer.csv", "target_column": "is_cancer"},
        {"file_path": "Data_final/churn.csv", "target_column": "is_exited"}
    ]

    # Results dictionary to store the results
    results = {}



    for dataset in datasets:
        file_path = dataset["file_path"]
        target_column = dataset["target_column"]
        for model, method in [(fw_SVM, "FW"), (afw_SVM, "AFW"), (pfw_SVM, "PFW")]:
            result = preprocess_and_train(
                file_path=file_path,
                target_column=target_column,
                model=model,
                test_size=0.4,
                random_state=42
            )
            results[f"{method}_{target_column}"] = result  # Store with a clear key


    # Display results
    for method_dataset, result in results.items():
        print(f"Results for {method_dataset}: Accuracy: {result["accuracy"]} F1-Score:{result["F1"]} Training Time = {result['training_time']:.4f} seconds")

    plot_accuracy_and_time(results)


