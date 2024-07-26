import wandb

from src.data_preprocessing import DataProcessor
from src.model import LinearRegressionModel
from src.plot import Diagnostics

wandb.require("core")

if __name__ == '__main__':
    # Tracking
    wandb.init(
        # Project name
        project='Implementing Multiple Linear Regression using JAX',
        # Hyperparameters
        config={
            'model': 'Multiple Linear Regression',
            'dataset': 'Student_Performance.csv',
            'epochs': 10000,
            'learning_rate': 5e-3,
            'lambda': 0.01,
            'dropout': 0.1,
            'max_patience': 100,
            'L_p regularization': 'L_2' 
        }
    )

    data_extractor = DataProcessor('/Users/liibanmohamud/Downloads/Machine Learning/Practice/MLOps/Advanced Multiple Linear Regression/data/Student_Performance.csv')
    # Run the pipeline
    X, y = data_extractor.preprocess_data(y_column='Performance Index')
    X_train, X_val, X_test, y_train, y_val, y_test = data_extractor.split_data_into_training_validation_testing(X, y)
    
    model = LinearRegressionModel(weights_init='random')
    model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    model.plot_losses()

    # Plot Diagnostic plots
    y_pred = model.predict(X_test=X_test)
    diagnostics = Diagnostics(X=X_test, y_pred=y_pred, y_true=y_test)
    diagnostics.diagnostic_plots(figsize=(8,8))

    wandb.finish()