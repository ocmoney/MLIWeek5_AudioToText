import wandb

sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {
            'values': [64]  # Current is 64
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.00005,  
            'max': 0.001,   
            
        },
        'epochs': {
            'values': [20]  # Current is 20
        },
        'nhead': {
            'values': [4,8]  # Current is 4
        },
        'num_encoder_layers': {
            'values': [2, 3, 4]  # Current is 3
        },
        'dim_feedforward': {
            'values': [1024, 2048]  # Current is 1024
        },
        'dropout': {
            'values': [0.1, 0.15, 0.2]  # Current is 0.1
        }
    }
}

# Define the initial parameters that we know work well
initial_params = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 20,
    'nhead': 4,
    'num_encoder_layers': 3,
    'dim_feedforward': 1024,
    'dropout': 0.1
}

def init_sweep():
    sweep_id = wandb.sweep(sweep_config, project="audio-classification")
    return sweep_id, initial_params 