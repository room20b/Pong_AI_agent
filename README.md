# Pong_AI_agent
AI agent learns pong with reinforcement learning

Python 3.6.1 :: Anaconda 4.4.0

#### Have virtualenv installed, if not:
$ conda install virtualenv                                                                                                               


#### Create a virtualenv:
- some_dir/
   - env_name/
   - Game_of_breakthrough/
      - requirements.txt

#### Activate virtualenv:
$ source activate env_name


#### Now cd into it and run:
$ pip install -r requirements.txt


#### Once all packages are installed simply run the following to get the server running:
$ python pong_AI.py

#### Note: It might take some time for the simulation to produce results.

#### The pictue below shows the accuracy and the rate at whice the AI agent learns. The different colors show the different state space of the game. Blue has a lot fewer states than red, so it learns and reaches the optimal value faster. But at the same time it is more volatile.

![alt text](https://github.com/room20b/Pong_AI_agent/blob/master/Screen%20Shot%202016-12-05%20at%2011.24.49%20PM.png?raw=true)
