# generate data
# python generate_data.py --games 1000 --iters 1000 --output training_data.npz

# Train V2 (3D CNN) — reuses existing training_data.npz, does NOT touch neural_model.pt

# BC
python train_neural.py --arch cnn bc --data training_data.npz --epochs 100

# DAgger
python train_neural.py --arch cnn dagger --rounds 2 --games 100

# RL
python train_neural.py --arch cnn rl --games 1000