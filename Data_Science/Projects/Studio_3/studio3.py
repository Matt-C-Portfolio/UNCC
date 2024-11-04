# Import Packages
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def boxplot(df, column_name, 
            box_color="#EBD8D4", 
            median_color="#323335", 
            whisker_color="#EBD8D4", 
            flier_color="#EBD8D4", 
            hist_color="#ddd0ca", 
            kde_color="#fef176", 
            background_color="#323335"):
    """
    Plots a boxplot and a histogram with a KDE overlay for the specified column in the DataFrame with a custom color scheme.
    
    Parameters:
    - df: DataFrame containing the data.
    - column_name: The name of the column to plot.
    - box_color: Color of the boxplot.
    - median_color: Color of the median line.
    - whisker_color: Color of the whiskers.
    - flier_color: Color of the fliers.
    - hist_color: Color for histogram bars.
    - kde_color: Color for the KDE line.
    - background_color: Color of the plot background.
    """
    plt.figure(figsize=(10, 12), facecolor=background_color)  # Set the figure background color

    # Row 1: Boxplot
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.boxplot(df[column_name], vert=False,
                patch_artist=True,
                boxprops=dict(facecolor=box_color, color=box_color),
                medianprops=dict(color=median_color, linewidth=1.5),
                whiskerprops=dict(color=whisker_color, linewidth=2),
                capprops=dict(color=whisker_color, linewidth=2),
                flierprops=dict(marker='o', markerfacecolor=flier_color))

    plt.title(f'Boxplot of {column_name}', color=box_color)
    plt.xlabel(column_name, color=box_color)
    plt.ylabel(' ', color=box_color)

    # Set the background color and spine colors for the boxplot
    plt.gca().set_facecolor(background_color)
    plt.gca().spines['bottom'].set_color(box_color)
    plt.gca().spines['top'].set_color(background_color)
    plt.gca().spines['right'].set_color(background_color)
    plt.gca().spines['left'].set_color(box_color)

    # Set x and y tick colors
    plt.tick_params(axis='x', colors=box_color)
    plt.tick_params(axis='y', colors=box_color)

    # Row 2: Histogram with KDE
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    sns.histplot(df[column_name], kde=True, color=hist_color, fill=True)

    # Customize KDE line color and width
    kde_lines = plt.gca().get_lines()  # Get the KDE lines
    if kde_lines:
        kde_lines[0].set_color(kde_color)
        kde_lines[0].set_linewidth(3)

    plt.title(f'Distribution of {column_name}', color=box_color)
    plt.xlabel(column_name, color=box_color)
    plt.ylabel(' ', color=box_color)
    
    # Set the background color for the histogram
    plt.gca().set_facecolor(background_color)
    plt.gca().spines['bottom'].set_color(box_color)
    plt.gca().spines['top'].set_color(background_color)
    plt.gca().spines['right'].set_color(background_color)
    plt.gca().spines['left'].set_color(box_color)

    # Set x and y tick colors for the histogram
    plt.tick_params(axis='x', colors=box_color)
    plt.tick_params(axis='y', colors=box_color)

    # Show the plot
    plt.tight_layout()
    plt.show()


def colToNumeric(col):
    try: return pd.to_numeric(col)
    except ValueError: return col

#Function to remove outliers based on IQR
def removeOutliersIQR(df, column):

    q1 = np.percentile(df[column], 25)
    q3 = np.percentile(df[column], 75)
    
    iqr = q3 - q1
    upperThreshold = q3 + (1.5 * iqr)
    lowerThreshold = q1 - (1.5 * iqr)
    
    df = df[(df[column] <= upperThreshold) & (df[column] >= lowerThreshold)]    
    
    return df

def plot_quantitative_data(df, columns, 
                           box_color="#EBD8D4", 
                           median_color="#323335", 
                           whisker_color="#EBD8D4",
                           flier_color="#EBD8D4", 
                           hist_color="#ddd0ca", 
                           kde_color="#fef176", 
                           background_color="#323335",
                           suptitle_color="#fef176"):
    # Calculate the number of columns and rows needed for the plots
    num_columns_per_row = 4
    num_rows = (len(columns) + num_columns_per_row - 1) // num_columns_per_row  # Total rows needed for boxplots
    total_rows = num_rows * 2  # Total rows for boxplots and histograms

    # Set up the figure with the necessary number of rows and columns
    fig, ax = plt.subplots(nrows=total_rows, ncols=num_columns_per_row, figsize=(28, 5 * total_rows))
    fig.patch.set_facecolor(background_color)
    fig.suptitle('Quantitative Data Visualizations', color=suptitle_color, fontsize=22)

    # Loop through the columns and set up plots
    for i, col in enumerate(columns):
        row_index = i // num_columns_per_row  # Row index for boxplots and histograms
        col_index = i % num_columns_per_row    # Column index for boxplots and histograms
        
        # Boxplot
        df.boxplot(column=col, patch_artist=True,
                   boxprops=dict(facecolor=box_color, color=box_color),
                   medianprops=dict(color=median_color, linewidth=1.5),
                   whiskerprops=dict(color=whisker_color, linewidth=2),
                   capprops=dict(color=whisker_color, linewidth=2),
                   flierprops=dict(marker='o', markerfacecolor=flier_color),
                   ax=ax[row_index * 2, col_index])  # Boxplot in the first row

        ax[row_index * 2, col_index].set_title(col, color=box_color)
        ax[row_index * 2, col_index].spines['bottom'].set_color(background_color)
        ax[row_index * 2, col_index].spines['top'].set_color(background_color)
        ax[row_index * 2, col_index].spines['right'].set_color(background_color)
        ax[row_index * 2, col_index].spines['left'].set_color(box_color)
        ax[row_index * 2, col_index].set_facecolor(background_color)
        ax[row_index * 2, col_index].tick_params(axis='y', colors=box_color)
        ax[row_index * 2, col_index].grid(visible=False)
        ax[row_index * 2, col_index].set_xticklabels([])

        # Histogram
        sns.histplot(df[col], kde=True, color=hist_color, ax=ax[row_index * 2 + 1, col_index])  # Histogram in the second row

        ax[row_index * 2 + 1, col_index].set_title(col, color=box_color)
        ax[row_index * 2 + 1, col_index].spines['bottom'].set_color(background_color)
        ax[row_index * 2 + 1, col_index].spines['top'].set_color(background_color)
        ax[row_index * 2 + 1, col_index].spines['right'].set_color(background_color)
        ax[row_index * 2 + 1, col_index].spines['left'].set_color(box_color)
        ax[row_index * 2 + 1, col_index].set_facecolor(background_color)
        ax[row_index * 2 + 1, col_index].tick_params(axis='y', colors=box_color)
        ax[row_index * 2 + 1, col_index].grid(visible=False)

        # Set KDE line color and width
        if ax[row_index * 2 + 1, col_index].get_lines():
            ax[row_index * 2 + 1, col_index].get_lines()[0].set_color(kde_color)
            ax[row_index * 2 + 1, col_index].get_lines()[0].set_linewidth(3)

    # Hide any unused subplots
    for j in range(len(columns) * 2, total_rows * num_columns_per_row):
        fig.delaxes(ax.flatten()[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    

def additiveOLS(df, y_name, threshold=0.001):
    X = df.drop(columns=[y_name])
    y = df[y_name]
    
    best_vars = []  # aggregate variables that increase r^2 the most
    remaining_vars = list(X.columns)  
    r_2 = 0  # current r_2
    
    # Iterate until no more variables improve R^2 above the threshold
    while remaining_vars:
        best_var = None
        best_r2 = r_2
        
        # Test each remaining variable
        for var in remaining_vars:
            x = sm.add_constant(df[best_vars + [var]])
            model = sm.OLS(y, x).fit()
            new_r2 = model.rsquared
            
            # Check r^2 against previous best r_2
            if new_r2 - r_2 > best_r2 - r_2:
                best_r2 = new_r2
                best_var = var
        
        # if best var improves model r^2 over threshold, save the variable
        if best_r2 - r_2 > threshold:
            best_vars.append(best_var)
            remaining_vars.remove(best_var)
            r_2 = best_r2
        else:
            break
    
    # create and return final model
    x_final = sm.add_constant(df[best_vars])
    final_model = sm.OLS(y, x_final).fit()
    
    return final_model


def plotCorrelationMatrices(df, target_variable, correlation_threshold=0, group_size=9,
                            background_color="#323335", text_color="#EBD8D4", 
                            cmap=sns.diverging_palette(240, 60, l=65, center="dark", as_cmap=True)):
    """
    Plots correlation matrices for features correlated with the target variable.

    Parameters:
    - df: DataFrame containing the data.
    - target_variable: The target variable for correlation analysis.
    - correlation_threshold: Minimum correlation value to include a feature.
    - group_size: Number of features to display in each correlation matrix.
    """
    # Select only numeric columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlations with the target variable
    correlations = numerical_df.corr()[target_variable]

    # Subset features with correlation above the threshold or below its negative
    selected_features = correlations[correlations.abs() > correlation_threshold].abs().sort_values(ascending=False).index.tolist()

    # Remove the target variable from the list of features
    if target_variable in selected_features:
        selected_features.remove(target_variable)

    # Split features into groups of size `group_size`
    feature_groups = [selected_features[i:i + group_size] for i in range(0, len(selected_features), group_size)]

    # Create and display correlation matrices using Seaborn heatmap
    for idx, group in enumerate(feature_groups):
        features_to_use = [target_variable] + group
        correlation_matrix = numerical_df[features_to_use].corr()

        plt.figure(figsize=(10, 8), facecolor=background_color)
        
        # Create heatmap with annotations
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap=cmap, 
                              linewidths=0.5, cbar_kws={"shrink": .8}, 
                              linecolor=background_color, 
                              fmt=".2f",  # Format for the correlation coefficients
                              annot_kws={"size": 10, "color": text_color})  # Set text size and color

        plt.title(f'Correlation Matrix {idx + 1}', color=text_color)
        plt.gca().set_facecolor(background_color)
        plt.gca().spines['bottom'].set_color(text_color)
        plt.gca().spines['top'].set_color(background_color)
        plt.gca().spines['right'].set_color(background_color)
        plt.gca().spines['left'].set_color(text_color)
        plt.tick_params(axis='x', colors=text_color)
        plt.tick_params(axis='y', colors=text_color)

        # Set the color of the colorbar ticks and labels
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelcolor=text_color)  # Change color of ticks

        plt.show()


# function to calculate means and standard deviations of model coefficient intervals
def calcIntervals(confidence_intervals):
    """
    Calculate means and standard deviations for the upper and lower bounds 
    of confidence intervals for model coefficients.

    Parameters:
    -----------
    confidence_intervals : list of list of tuples
        A list containing confidence intervals for model coefficients. 
        Each interval is a tuple of (lower_bound, upper_bound).

    Returns:
    --------
    upper_means : numpy.ndarray
        The mean of the upper bounds of the confidence intervals.
    
    lower_means : numpy.ndarray
        The mean of the lower bounds of the confidence intervals.
    
    upper_stds : numpy.ndarray
        The standard deviation of the upper bounds of the confidence intervals.
    
    lower_stds : numpy.ndarray
        The standard deviation of the lower bounds of the confidence intervals.
    """
    # Initialize lists for means and standard deviations
    uppers = []
    lowers = []
    upper_means = []
    lower_means = []
    upper_stds = []
    lower_stds = []

    # populate uppers and lowers with list of uppers and lowers for each model
    for model in confidence_intervals:
        upper = []
        lower = []
        for interval in model:
            lower.append(interval[0])
            upper.append(interval[1])
        uppers.append(upper)
        lowers.append(lower)
    # find mean and standard deviation for each coefficient
    upper_means = np.mean(uppers, axis=0)
    upper_stds = np.std(uppers, axis=0)
    lower_means = np.mean(lowers, axis=0)
    lower_stds = np.std(lowers, axis=0)

    return upper_means, lower_means, upper_stds, lower_stds


def crossValidationLR(X, y, folds=5, random_state=42, constant=True):
    """
    Perform K-Fold cross-validation on a linear regression model and 
    calculate various model statistics including coefficients, 
    R-squared values, and p-values.

    Parameters:
    -----------
    X : pandas.DataFrame
        The feature matrix used for training the model.

    y : pandas.Series
        The target variable corresponding to the feature matrix.

    folds : int, optional
        The number of folds to use for cross-validation (default is 5).

    random_state : int, optional
        The random seed used for reproducibility (default is 42).

    constant : bool, optional
        If True, adds a constant term to the model (default is True).

    Returns:
    --------
    lm_final : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted linear regression model after cross-validation.

    models_dict : dict
        A dictionary containing the fitted models for each fold of the cross-validation.
    """

    # Get date and time
    date = datetime.now()
    day = date.strftime("%a, %d %b %Y")
    time = date.strftime('%H:%M:%S')

    # List to store params and metrics of each model
    model_params = []
    r_squared = []
    adj_r_squared = []
    f_stat = []
    f_p_value = []
    durb_wat = []
    standard_error = []
    conf_intervals = []
    std_err = []
    
    # Create a dictionary to store models
    models_dict = {}

    # Create list for storing p-values
    if constant:
        p_values = [[] for x in range(X.shape[1] + 1)]
    else:
        p_values = [[] for x in range(X.shape[1])]
    
    # Initialize the cross-validator
    kf = KFold(n_splits=folds, random_state=random_state, shuffle=True)
    
    # Saving column names for reference because they are dropped in conversion
    columns = X.columns
    
    # Convert to numpy array
    X = X.to_numpy()

    # Model each fold
    for fold_index, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.reset_index(drop=True)[train_index], y.reset_index(drop=True)[test_index]
        
        if constant:
            # Add Constants
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)

        # Train model
        lm = sm.OLS(y_train, X_train).fit()

        # Save params and metrics for averaging
        model_params.append(lm.params)
        r_squared.append(lm.rsquared)
        adj_r_squared.append(lm.rsquared_adj)
        f_stat.append(lm.fvalue)
        f_p_value.append(lm.f_pvalue)
        durb_wat.append(durbin_watson(lm.resid))
        standard_error.append(lm.bse)
        conf_intervals.append(lm.conf_int().values)
        std_err.append(lm.bse)

        # Store the model in the dictionary
        models_dict[f'lm{fold_index}'] = lm
        
        for x in range(len(lm.params)):
            p_values[x].append(lm.pvalues.iloc[x])

    # Save Statistic Means and stds
    r_squared_mean = np.mean(r_squared)
    r_squared_std = np.std(r_squared)

    adj_r_squared_mean = np.mean(adj_r_squared)
    adj_r_squared_std = np.std(adj_r_squared)

    f_stat_mean = np.mean(f_stat)
    f_stat_std = np.std(f_stat)

    f_p_value_mean = np.mean(f_p_value)
    f_p_value_std = np.std(f_p_value)

    p_values_means = [np.mean(values) for values in p_values]

    durb_wat_mean = np.mean(durb_wat)
    durb_wat_std = np.std(durb_wat)
    
    # Calculate params from averaging all models params
    params_final = np.mean(model_params, axis=0)
    # Calculate standard deviation of params
    params_std = np.std(model_params, axis=0)
    std_err_final = np.mean(std_err, axis=0)

    # Calculate confidence interval means and stds
    conf_intervals = [sublist.tolist() for sublist in conf_intervals]
    upper_means, lower_means, upper_stds, lower_stds = calcIntervals(conf_intervals)
    
    # Print the formatted information with right alignment
    print(f'{"OLS Cross-Validation Averaging":^98}')
    print('====================================================================================================')
    print(f'{"Num Folds:"} {folds:>89}')
    print(f'{"Dep. Variable:"} {y.name:>85}')
    print(f'{"Model:"} {"OLS":>93}')
    print(f'{"Method:"} {"Least Squares":>92}')
    print(f'{"Date:"} {day:>94}')
    print(f'{"Time:"} {time:>94}')
    print('====================================================================================================')
    
    # Print Statistic Means
    print(f'{"mean":>89}  {"std":>9}')
    print('----------------------------------------------------------------------------------------------------')
    print(f'{"R-squared":>71} {r_squared_mean:>17.3f} {r_squared_std:>10.3f}')
    print(f'{"Adj. R-squared":>71} {adj_r_squared_mean:>17.3f} {adj_r_squared_std:>10.3f}')
    print(f'{"F-statistic":>71} {f_stat_mean:>17.3f} {f_stat_std:>10.3f}')
    print(f'{"Prob (F-statistic)":>71} {f_p_value_mean:>17.3f} {f_p_value_std:>10.3f}')
    print(f"{'Durbin-Watson':>71} {durb_wat_mean:>17.3f} {durb_wat_std:>10.3f}")

    if constant:
        # Print Coefficients and P-values with constant
        print('====================================================================================================')
        print(f'{"coef":>46} {"std":>8} {"P>|t|":>10} {"0.025":>10} {"std":>6} {"0.975":>10} {"std":>6}')
        print('----------------------------------------------------------------------------------------------------')
        print(f'{"const":>35} {params_final[0]:>10,.3f} {params_std[0]:>6.3f} {p_values_means[0]:>10,.3f} {lower_means[0]:>10.3f} {lower_stds[0]:>6.3f} {upper_means[0]:>10.3f} {upper_stds[0]:>6.3f}')
        for x in range(0, len(columns)):
            print(f'{columns[x]:>35} {params_final[x+1]:>10.3f} {params_std[x+1]:>6.3f} {p_values_means[x+1]:>10.3f} {lower_means[x+1]:>10.3f} {lower_stds[x+1]:>6.3f} {upper_means[x+1]:>10.3f} {upper_stds[x+1]:>6.3f}')
        print('\n====================================================================================================\n')
    
    else:
        # Print Coefficients and P-values without constant
        print('====================================================================================================')
        print(f'{'std err':>36}{"coef":>10} {"std":>6} {"P>|t|":>10} {"0.025":>10} {"std":>6} {"0.975":>10} {"std":>6}')
        print('----------------------------------------------------------------------------------------------------')
        for x in range(0, len(columns)):
            print(f'{columns[x]:>25} {std_err_final[x]:>10.3f} {params_final[x]:>10.3f} {params_std[x]:>6.3f} {p_values_means[x]:>10.3f} {lower_means[x]:>10.3f} {lower_stds[x]:>6.3f} {upper_means[x]:>10.3f} {upper_stds[x]:>6.3f}')
        print('\n====================================================================================================\n')
    
    # Create Model using cross-validation params
    # Train/Test Split of all data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # Add constant
    if constant:
        X_train = sm.add_constant(X_train)
    # Initialize new model 
    lm_final = sm.OLS(y_train, X_train).fit()
    
    # Update the model parameters with the average coefficients
    lm_final.params = params_final
    # Make durbin watson mean accessible
    lm_final.durb_wat_mean = durb_wat_mean
    # Make standard error mean accessible
    lm_final.bse = np.mean(standard_error)

    return lm_final, models_dict


def evaluateModels(models, X_test, y_test):
    """
    Evaluates multiple linear models and returns their evaluation metrics in a DataFrame.

    Parameters:
    models: Dictionary of fitted linear models with model names as keys.
    X_test: Test feature data.
    y_test: True values for the target variable.

    Returns:
    pd.DataFrame: DataFrame containing evaluation metrics for each model.
    """
    metrics = {}

    for name, model in models.items():
        # Predicting the target variable using the test features
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        r_squared = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Store metrics in the dictionary
        metrics[name] = {
            'R-squared': r_squared,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'Models'
    
    return metrics_df



def crossValidationDT(X_train, y_train, random_state=42, folds=5, criterion='squared_error', n_jobs=1):
    """
    Train multiple Decision Tree Regressor models using k-fold cross-validation with GridSearchCV.

    Parameters:
    - X_train: DataFrame, feature data for training
    - y_train: Series, target variable for training
    - random_state: int, random state for reproducibility
    - folds: int, number of folds for k-fold cross-validation
    - criterion: str, the function to measure the quality of a split ('mse' or 'mae')
    - n_jobs: int, number of jobs to run in parallel (1 means single processor, -1 means using all processors)

    Returns:
    - models: dict, a dictionary of trained Decision Tree Regressor models with best parameters
    """
    
    # Initialize KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    # Define the model parameters to tune
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    models = {}  # Dictionary to hold the fitted models
    dt_counter = 1  # Counter for naming models
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Initialize the Decision Tree Regressor
        dt_regressor = DecisionTreeRegressor(random_state=random_state, criterion=criterion)
        
        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=dt_regressor, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=folds, n_jobs=n_jobs)
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Store the fitted model with the best parameters in the dictionary
        models[f'dt{dt_counter}'] = grid_search.best_estimator_
        dt_counter += 1

    return models


def crossValidationRF(X_train, y_train, random_state=42, folds=5, n_jobs=1):
    """
    Train multiple Random Forest Regressor models using k-fold cross-validation with GridSearchCV.

    Parameters:
    - X_train: DataFrame, feature data for training
    - y_train: Series, target variable for training
    - random_state: int, random state for reproducibility
    - folds: int, number of folds for k-fold cross-validation

    Returns:
    - models: dict, a dictionary of trained Random Forest Regressor models with best parameters
    """
    
    # Initialize KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    # Define the model parameters to tune
    param_grid = {
        'n_estimators': [5, 10, 20],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    models = {}  # Dictionary to hold the fitted models
    rf_counter = 1  # Counter for naming models
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Initialize the Random Forest Regressor
        rf_regressor = RandomForestRegressor(random_state=random_state)
        
        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=folds, n_jobs=n_jobs)
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Store the fitted model with the best parameters in the dictionary
        models[f'rf{rf_counter}'] = grid_search.best_estimator_
        rf_counter += 1

    return models


def crossValidationXGB(X_train, y_train, random_state=42, folds=5, n_jobs=1):
    """
    Train multiple XGBoost Regressor models using k-fold cross-validation with GridSearchCV.

    Parameters:
    - X_train: DataFrame, feature data for training
    - y_train: Series, target variable for training
    - random_state: int, random state for reproducibility
    - folds: int, number of folds for k-fold cross-validation
    - n_jobs: int, number of parallel jobs to run (-1 means using all processors)

    Returns:
    - models: dict, a dictionary of trained XGBoost Regressor models with best parameters
    """
    
    # Initialize KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    # Define the model parameters to tune
    param_grid = {
        'n_estimators': [5, 10, 20],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    models = {}  # Dictionary to hold the fitted models
    xgb_counter = 1  # Counter for naming models
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Initialize the XGBoost Regressor
        xgb_regressor = XGBRegressor(random_state=random_state)
        
        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=folds, n_jobs=n_jobs)
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Store the fitted model with the best parameters in the dictionary
        models[f'xgb{xgb_counter}'] = grid_search.best_estimator_
        xgb_counter += 1

    return models