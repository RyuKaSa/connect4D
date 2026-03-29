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


```
python arena.py tournament -n 20 random mcts-1k mcts-2k mcts-5k

python arena.py ratings

────────────────────────────────────────────────────
  Rank  Agent           Rating     RD  Games
────────────────────────────────────────────────────
     1  mcts-5k         1832.1  125.2     15
     2  mcts-2k         1590.6  125.2     15
     3  mcts-1k         1530.2  125.2     15
     4  random          1047.1  125.2     15
────────────────────────────────────────────────────
```