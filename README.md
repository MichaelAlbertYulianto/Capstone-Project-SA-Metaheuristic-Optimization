# Metaheuristic Optimization with LazyPredict

This project demonstrates the use of metaheuristic optimization techniques along with LazyPredict for regression and classification tasks. The datasets used in this project include medical cost, iris, and mall customer datasets.

## Project Structure

## Datasets

- `insurance.csv`: Medical cost dataset
- `Iris.csv`: Iris flower dataset
- `Mall_Customers.csv`: Mall customer segmentation dataset

## Libraries Used

- `numpy`
- `pandas`
- `seaborn`
- `lazypredict`
- `scikit-learn`
- `mealpy`

## Steps

1. **Library Importing**: Import necessary libraries and modules.
2. **Load Data Set**: Load datasets from CSV files.
3. **Exploratory Data Analysis**: Perform EDA on the datasets.
4. **Preprocessing**: Clean and preprocess the datasets.
5. **Encoding**: Encode categorical features.
6. **Scaling**: Scale the data using standardization.
7. **Model Training with LazyPredict**: Use LazyPredict to find the best models for regression and classification tasks.
8. **Metaheuristic Optimization**: Optimize model parameters using Simulated Annealing (SA).

## Usage

1. **Load the Notebook**: Open `MetaheuristicOPT.ipynb` in Jupyter Notebook or JupyterLab.
2. **Run the Cells**: Execute the cells step-by-step to perform data loading, preprocessing, model training, and optimization.
3. **View Results**: Observe the results of the best models and optimized parameters.

## Example

### LazyRegressor

```python
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
model_reg, predictions_reg = lazy_reg.fit(all_df_dict['train_set'][0][0], all_df_dict['test_set'][0][0], all_df_dict['train_set'][0][1], all_df_dict['test_set'][0][1])
model_reg
```

### LazyClassifier

```python
lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
model_clf, predictions_clf = lazy_clf.fit(all_df_dict['train_set'][1][0], all_df_dict['test_set'][1][0], all_df_dict['train_set'][1][1], all_df_dict['test_set'][1][1])
model_clf
```

### Metaheuristic Optimization for Regression

```python
data = {
    "X_train": X_train_reg,
    "X_test": X_test_reg,
    "y_train": y_train_reg,
    "y_test": y_test_reg
}

class LinRegOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        n_jobs, fit_intercept = x_decoded["n_jobs"], x_decoded["fit_intercept"]
        lr = LinearRegression(n_jobs=n_jobs, fit_intercept=fit_intercept)
        lr.fit(self.data["X_train"], self.data["y_train"])
        y_predict = lr.predict(self.data["X_test"])
        return metrics.mean_squared_error(self.data["y_test"], y_predict)

my_bounds = [
    IntegerVar(lb=1, ub=50, name="n_jobs"),
    BoolVar(n_vars=1, name="fit_intercept"),
]

problem = LinRegOptimizedProblem(bounds=my_bounds, minmax="max", data=data)
model = SA.OriginalSA(epoch=200, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")
print(f"Best solution: {model.g_best.solution}")
print(f"Best Mean Squared Error: {model.g_best.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")
```

