import math
import numpy as np
import random
import sys
import time

#constants
ball_x = 0      
ball_y = 1
velocity_x = 2
velocity_y = 3
paddle_y = 4

paddle_height = 0.2
paddle_x = 1

ACT={0.04:0,-0.04:1,0:2}
actions = [0.04, -0.04, 0]
game=0
max_val = -sys.maxsize


#Q = [[[[[[0 for x5 in ACT]for x4 in range(12)]for x3 in range(3)]for x2 in range(2)]for x1 in range(12)]for x0 in range(12)]

min_sa_freq = 8   
C = 30        
gamma = 0.8

def d_action(action): #map 0.04, -0.04, 0 to 0,1,2
	ret_action=0
	if (action==0.04):
		ret_action=0
	elif (action==-0.04):
		ret_action=1
	else:
		ret_action=2
	return ret_action

def d_state(state):
    dis_state = [0]*5  
    dis_state[ball_x] = min(int(12*state[ball_x]),11) #12 by 12 board location
    dis_state[ball_y] = min(int(12*state[ball_y]),11)
    dis_state[paddle_y] = min(int(12*state[paddle_y]/(1-paddle_height)),11) #paddle
    dis_state[velocity_x] = int(np.sign(state[velocity_x])) #x velocity of ball
    if (abs(state[velocity_y]) < 0.015): # y velocity of ball 
        dis_state[velocity_y] = 0
    else:
        dis_state[velocity_y] = int(np.sign(state[velocity_y]))
    return (dis_state[0],dis_state[1],dis_state[2],dis_state[3],dis_state[4])

def rebound(state):
	if (state[ball_x]<0):
		state[ball_x]=-state[ball_x]
		state[velocity_x]=-state[velocity_x]
		#print('Bounce right')

	if (state[ball_y]>1):
		state[ball_y]=2-state[ball_y]
		state[velocity_y]=-state[velocity_y]
		#print('Bounce up')

	if (state[ball_y]<0):
		state[ball_y]=-state[ball_y]
		state[velocity_y]=-state[velocity_y]
		#print('Bounce down')

	if ((state[ball_x]>=paddle_x) and (state[ball_y]>=state[paddle_y]) and (state[ball_y]<=(state[paddle_y]+paddle_height))):
		#print('Rebound success')
		state[ball_x]=2*paddle_x-state[ball_x]
		while True:
			temp=np.random.uniform(-0.015,0.015)-state[velocity_x]
			if(abs(temp)>0.03):
				state[velocity_x]=temp
				state[velocity_y]=state[velocity_y]+np.random.uniform(-0.03,0.03)
				break
		return 1
	return 0

def end_case(state):
	#global game
	miss_y=(state[ball_y]<state[paddle_y] or state[ball_y]>(state[paddle_y]+paddle_height))
	miss_x=(state[ball_x]>paddle_x)
	if(miss_x and miss_y):
		#print('Reached end case!')
		return True
	else:
		return False

def next_move(state,Q,N_freq):
	action_val = 0
	max_Q = max_val
	for action in actions:
		sa_val = (d_state(state), action)
		if ((sa_val not in Q) or (sa_val not in N_freq) or (N_freq[sa_val] < min_sa_freq)):
			return action
		if (Q[sa_val] > max_Q):
			#select action with le max value
			max_Q = Q[sa_val]
			action_val = action
	return action_val
	'''qmax=0.001
	discrete_state=d_state(state)
	bx,by,vx,vy,paddle_pos=discrete_state
	#print(state)
	#print(discrete_state)
	#val=0
	for a in ACT:
		#discrete_action=d_action(a)
		#print(a)
		q_val=Q[bx][by][vx][vy][paddle_pos][ACT[a]]
		#print(q_val)
		if(q_val>qmax):
			#val+=1
			qmax=q_val
			ret_val=a
		else:
			#if(val==0):
			ret_val=0
	#print(ret_val)
	return ret_val'''

def reward(state):
	#R=0
	#print('hahahahahaah') #donot rebound or check end case here
	if ((state[ball_x]>=paddle_x) and (state[ball_y]>=state[paddle_y]) and (state[ball_y]<=(state[paddle_y]+paddle_height))):
		#print('bleh')
		R=50
	elif(end_case(state)==True):
		#print('dammit')
		R=-50
	else:
		#print('meh')
		R=0
	#R=RE
	return R

def max_next_Q(state,ACTION):
	global Q
	discrete_state=d_state(state)
	bx,by,vx,vy,paddle_pos=discrete_state
	Q0=Q[bx][by][vx][vy][paddle_pos][ACT[0.04]]
	Q1=Q[bx][by][vx][vy][paddle_pos][ACT[-0.04]]
	Q2=Q[bx][by][vx][vy][paddle_pos][ACT[0]]
	Qmax=max(Q0,Q1,Q2)
	print('Qmax ', Q0, Q1, Q2)
	return Qmax

def max_next_state(state,Q):
	#print(state)
	max_Q=max_val
	if (state is False):
		return 0
	for action in actions:
		sa_val = (d_state(state), action)
		if (sa_val in Q):
			#select the Q wih max value
			max_Q = max(max_Q, Q[sa_val])
		else:
			max_Q = max(max_Q, 0)
	#print(max_Q)
	return max_Q

#need to update state and actions
def update_state(state,act):
	#updateing the state 
	state[0]+=state[2]
	state[1]+=state[3]
	state[4]+=act
	#checking for boundry conditions
	if(rebound(state)==1):
		return (state, 1)	
	if(end_case(state)==True):
		return(False,-1)
	if(rebound(state)==0):
		return (state, 0)

def Q_learn(state, Q, N, reward, state_action):
	#first iteration when everythin is empty
	if(state_action not in N):
		N[state_action]=1
		alpha=C/(C+N[state_action])
		Q[state_action]=alpha*(reward+gamma*max_next_state(state,Q))
	#Stuff from the equations online 
	else:
		N[state_action]+=1
		alpha=C/(C+N[state_action])
		Q[state_action]+=alpha*(reward+gamma*max_next_state(state,Q)-Q[state_action])
	return Q[state_action]
	'''global Q
	discrete_state=d_state(state)
	bx,by,vx,vy,paddle_pos=discrete_state
	AHPHA=0.8
	GAMMA=0.5
	Q[bx][by][vx][vy][paddle_pos][ACTION]+=AHPHA*(reward(state)+GAMMA*(max_next_state(state,ACTION)-Q[bx][by][vx][vy][paddle_pos][ACTION]))
	#print('reward ', RE)
	print(Q[bx][by][vx][vy][paddle_pos][ACTION])'''

#####add plot stuff and .csv file
def main():
	#state=[0.5, 0.5, 0.03, 0.01, 0.4]
	Q_val_state_action = {}
	N_val_state_action = {}
	game_no=0
	train=10000 #change this however you want
	bounces_sum=0.0
	t=0
	while(game_no<train):

		if(game_no%(train-1000)==0):
			print('game: ', game_no)
			print('average bounces in last 1000 games ', bounces_sum/1000.0)
			bounces_sum=0
		#initial state
		state_rn=[0.5,0.5,0.03,0.01,0.4]
		#current bounce values
		cb_ha=0
		#its a given
		hmm_should_i_stop=True

		while(hmm_should_i_stop):
			#select next action
			action=next_move(state_rn,Q_val_state_action,N_val_state_action)
			#state action ket value as a tuple for Q
			state_action_val=(d_state(state_rn),action)
			#get the next state and the possible reward values
			next_state, reward_val=update_state(state_rn,action)
			#print(next_state)

			#if we reached le trminal value we assign the state to be a bool
			#and update all our bounce and game values
			if(next_state is False):
				hmm_should_i_stop=False
				bounces_sum+=cb_ha
				cb_ha=0
				game_no+=1
			#do le dank Q learning
			#print(state_action_val)
			Q_val_state_action[state_action_val]=Q_learn(next_state,Q_val_state_action,N_val_state_action,reward_val,state_action_val)
			#update the bounce lel 
			if(reward_val==1):
				cb_ha+=1

if __name__=='__main__':
    lol = time.time()
    main()
    print((time.time() - lol))


