# Locomotion Project

A reinforcement learning project for robot locomotion using RSL-RL framework.

## Setup

1. Clone the repository:
```bash
git clone git@github.com:FedeAi/locomotion-p0.git --recursive
cd locomotion-p0
```
2. Initialize the submodules:
```bash
git submodule update --init --recursive
```

4. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

5. Install dependencies:
```bash
# Core dependencies
pip install -r requirements.txt

# Install RSL-RL
cd rsl_rl
pip install -e .
cd ..


# Install Genesis
cd genesis
pip install -e .
cd ..
```

## Usage

Train the Go2 robot:
```bash
python src/go2_train.py --exp_name="my_experiment" --max_iterations=1000
```

## Project Structure
```
locomotion-p0/
├── src/
│   ├── go2_train.py    # Training script
│   └── go2_env.py      # Environment
├── rsl_rl/             # RSL-RL framework
└── requirements.txt     # Project dependencies
```


## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request