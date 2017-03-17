# Deep Reinforcement Learning


## Requirements

* Platform: Ubuntu 16 LTS
* 3rd-part package: pygame, tensorflow, opencv, numpy, matplotlib

## Double Q-learning 

This project applies double Q-learning architecture to obtain high performance in playing the famous flappy bird game. The game comes from [Yen](https://github.com/yenchenlin/DeepLearningFlappyBird.git) on github

To run the demo

```python
python main.py
```

To see the report

```python
python report.py
```

To retrain the model, set ```TRAIN=True``` in ```main.py```

```python
# TRAIN = False
TRAIN = True
```

## Training Result
The unit of x-axis is 1e6.

![image](01-DoubleDQN-flappy-bird/report.png)


