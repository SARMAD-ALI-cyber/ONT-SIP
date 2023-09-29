import flwr as fl
import tensorflow as tf

# Configuration dictionary
# Se this documentation for clearance I have used custom config dict
# this is an easy way to control hyperparameters of clients, we access this dict on client side fit function
# there is also on_evaluate_config_fn argument in strategy like for fit it is done in same manner.
# https://flower.dev/docs/framework/how-to-configure-clients.html

def fit_config(server_round: 2):
    config = {
        "current_round": server_round,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "local_epochs": 40,
    }
    return config


# Aggregation strategy
strategy = fl.server.strategy.FedAvg(
    on_fit_config_fn=fit_config
)

# star flower server
server = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)


# 0.0.0.0 is used by server to listen to any ip address 8080 is a port number
# here num_rounds means how many times you want to repeat a step
# like if I want to train for 40 epoch and num_round is 3 it wil train 3 times on 40 epochs

