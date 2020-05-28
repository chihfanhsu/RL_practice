# Practicing reinforcement learning algorithm on a small grid world
Contributors: silver and helen

## A Small Grid World 
A small grid world, you can define the parameter in the [maze_env.py](https://github.com/chihfanhsu/RL_practice/blob/master/maze_env.py).
For example, <br />
![alt text](https://github.com/chihfanhsu/RL_practice/blob/master/README_fig/world_example.png)<br />
```python
self.action_space = ['0','1','2','3'] # agent's actions, move up, right, down, and left
self.MAZE_Limit = np.array([4, 4]) # (Width,Height) of the world
self.origin = np.array([0, 0]) # world's original location
self.start_state =  np.array([1, 2]) # initial location of the agent
self.oval_state = np.array([[0, 0],[3, 3]]) # reward!
self.block_state = np.array([[2, 1]]) # you shall not pass!
self.hell_state =  np.array([[4, 4]]) # you die!
```
## Implemented Algorithms
You can find the functions in the [utility.py](https://github.com/chihfanhsu/RL_practice/blob/master/utility.py).
* Planning
  * Dynamic programing
* Model-free Prediction
  * Monte-Carlo
  * Temporal Difference (Action-value)
    * TD(0)
    * TD(n)
    * TD(<img src="http://chart.googleapis.com/chart?cht=tx&chl=\lambda" style="border:none;">)
* Model-free Control
  * Monte-Carlo
  * Sarsa(<img src="http://chart.googleapis.com/chart?cht=tx&chl=\lambda" style="border:none;">), on/off-policy
