[wins-losses-ties]  time

# Random vs Random, 100000 games
```
python arena.py random random 100000
[55215-44780-5]  50.0s
```

# MCTS-1000 vs Random, 100 games
```
python arena.py mcts-1k random 100
[100-0-0]  180.2s
```

# MCTS-2000 vs MCTS-1000, 20 games
```
python arena.py mcts-2k mcts-1k 20
[13-7-0]  384.3s
```

# MCTS-5000 vs MCTS-1000, 10 games
```
python arena.py mcts-5k mcts-1k 10
[8-2-0]  284.9s
```