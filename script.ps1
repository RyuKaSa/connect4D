# generate data
python generate_data.py --games 1000 --iters 1000 --output training_data.npz

# Delete the corrupted model
Remove-Item neural_model.pt, neural_model_pre_rl.pt -ErrorAction SilentlyContinue

# Retrain BC
python train_neural.py bc --data training_data.npz --epochs 100

# DAgger
python train_neural.py dagger --rounds 2 --games 100

# RL
python train_neural.py rl --games 1000