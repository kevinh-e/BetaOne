# BetaOne

AlphaZero-style chess engine

1.
    Setting up the paper implementation
    tabular rasa approach, using mcts and resnet.

2.
    implementing multiprocessing for mcts to speed it up
        implementing uci interface for GUI support
        checkpointing

3.
    optimizing mcts and multiprocessing
        implementing batch inferencing
        optimised mcts search
        mixed precision training

4.
    drastically improving performance with supervised pretraining
        using pgn data from engine games + lichess high level games
        considered c++ parse script
        implemented an iteratable dataset for parsing large data into tensors
        changed to a cosineannealing learning rate scheduler for high number of iterations
