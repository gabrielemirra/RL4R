import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, Huber, MeanSquaredError
import numpy as np
from a2c import *
from truss_environment import *

print('set')
print('set_2')
state_size = [64,64]
base_env = env(state_size)
grid_size = 32*32
stock_size = 20
action_shape = [1024,stock_size,2]
batch_size = 1
episode_len = 50
norm = 40
n_episodes = 500000
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'


#define loss objects and optimizers

huber = Huber(reduction=tf.losses.Reduction.NONE)
mse = MeanSquaredError(reduction=tf.losses.Reduction.NONE)
categorical_crossentropy = SparseCategoricalCrossentropy(from_logits = True,reduction=tf.losses.Reduction.NONE)
ac_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# construct metric objects

a_loss_m = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
e_loss_m = tf.keras.metrics.Mean('entropy_loss', dtype=tf.float32)
reward_m = tf.keras.metrics.Mean('mean_reward', dtype=tf.float32)
c_loss_m = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
ac_grads = tf.keras.metrics.Mean('ac_gradients', dtype=tf.float32)


#define a2c training loss functions

def actor_loss(actions, logits, advantages):
    advantages = tf.stop_gradient(advantages)
    cross_entropy = tf.zeros((logits.shape[0],logits.shape[1])) 
    split_logits = tf.split(logits,action_shape,axis=-1) #4 lists with shapes (B,T,1024), (B,T,1024), (B,T,2), (B,T,3)

    for i in range(len(split_logits)):
        cross_entropy += categorical_crossentropy(actions[:,:,i],split_logits[i])   
    cross_entropy = cross_entropy*advantages
    cross_entropy = tf.math.reduce_sum(cross_entropy, axis=-1) #SUM ALONG T
    loss = tf.math.reduce_sum(cross_entropy) #MEAN ALONG B
    return loss #must have shape B,T,1

def entropy_loss(logits):
    entropy = tf.zeros((logits.shape[0],logits.shape[1]))
    split_logits = tf.split(logits,action_shape,axis=-1) #4 lists with shapes (B,T,1024), (B,T,1024), (B,T,2), (B,T,3)
    for i in range(len(split_logits)):
        policy = tf.nn.softmax(split_logits[i], axis=-1)
        log_probs = tf.nn.log_softmax(split_logits[i], axis=-1)        
        entropy += tf.math.reduce_sum(policy*log_probs, axis=-1) 
    entropy = tf.math.reduce_sum(entropy, axis=-1)
    loss = tf.math.reduce_sum(entropy)
    return loss #must have shape B,T,1

def critic_loss(values, returns):
    l2 = mse(values,returns)
    l2 = tf.math.reduce_sum(l2, -1)
    loss = tf.math.reduce_sum(l2)#tf.math.reduce_sum(advantages**2)
    return loss

def mean_grads(grads):
    mean_grad = tf.zeros(())
    for grad in grads:
        mean_grad += tf.reduce_mean(grad)
    mean_grad /= len(grads)
    return mean_grad

@tf.function
def actor_critic_sgd_step(actor_critic, states, stocks, actions,rewards,la, actions_mask,noise,ht,ct,behaviour_policy_logits, entropy_decay):

    print(f'states: {states.shape}, stocks: {stocks.shape}, actions: {actions.shape}, rewards: {rewards.shape}, last action: {la.shape}, action mask: {actions_mask.shape}, noise vector: {noise.shape}, policy logits: {behaviour_policy_logits.shape}')
    states_ = tf.cast(states, tf.float32)
    stocks_ = tf.cast(stocks, tf.float32)
    actions_ = tf.cast(actions, tf.float32)
    rewards_ = tf.cast(rewards, tf.float32)
    la_ = tf.cast(la, tf.float32)
    actions_mask_ = tf.cast(actions_mask, tf.float32)
    noise_ = tf.cast(noise, tf.float32)



    with tf.GradientTape() as tape:

        actions, target_policy_logits, values = actor_critic([states_, stocks_, la_, actions_mask_, noise_, actions_, ht, ct]) ####################################
        # values = critic([states_, la_, actions_mask_, noise_, actions_])

        rewards,values = tf.reshape(rewards_,(batch_size,episode_len)),tf.reshape(values,(batch_size,episode_len))
        actions,behaviour_policy_logits = tf.reshape(actions,(batch_size,episode_len,len(action_shape))),tf.reshape(behaviour_policy_logits,(batch_size,episode_len,np.sum(action_shape))) 
        target_policy_logits = tf.reshape(target_policy_logits,(batch_size,episode_len,np.sum(action_shape)))

        advantage = rewards-values
        
        a_loss = actor_loss(actions, target_policy_logits, advantage)
        e_loss = entropy_loss(target_policy_logits)#tf.stop_gradient(target_policy_logits))
        c_loss = critic_loss(values, rewards)

        ac_loss = a_loss+(0.005*e_loss)+(c_loss)

    grads = tape.gradient(ac_loss, actor_critic.trainable_variables)
    grads = [tf.clip_by_norm(g,norm) for g in grads]
    ac_opt.apply_gradients(zip(grads, actor_critic.trainable_variables))


    return a_loss, c_loss, e_loss, grads

def train_actor_critic(actor_critic, \
    states_episodes, stocks_episodes, actions_episodes, rewards_episodes, core_states, noise_vectors, behaviour_policy_logits,entropy_decay):
    
# states, actions, rewards, la, action_mask , noise_vectors, behaviour_policy_logits

# (320, 64) (5, 4) (5, 1) (5, 4) (5, 4) (5, 10) (5, 2052)

# (1280, 64, 64) (1280, 5) (1280, 1) (1280, 5) (1280, 5) (1280, 10) (64, 20, 2056)

    states = np.array(states_episodes)
    stocks = np.array(stocks_episodes)
    actions,rewards = np.vstack(actions_episodes),np.expand_dims(np.reshape(rewards_episodes,-1),-1)
    noise_vectors = np.vstack(noise_vectors)
    behaviour_policy_logits = np.expand_dims(np.array(behaviour_policy_logits),0)
    core_states = np.squeeze(np.array(core_states)) #32,10,2,256
    ht,ct = core_states[-1,0,:],core_states[-1,1,:]
    ht,ct = np.reshape(ht,(batch_size,256)), np.reshape(ct,(batch_size,256))



    ########################################## DOUBLE CHECK
    prev_action = np.reshape(actions,(batch_size,episode_len,len(action_shape))) #32,20,4
    la = np.insert(prev_action,0, np.zeros(len(action_shape)), axis=-2)
    la = np.delete(la, -1, -2)

    
    action_mask = np.ones_like(la)
    action_mask[la[:,:,2] == 0] = [1,1]+[0 for i in range(len(action_shape)-2)]
    action_mask[:,0] = np.zeros(len(action_shape))



    la = np.reshape(la,(batch_size*episode_len,len(action_shape)))
    action_mask = np.reshape(action_mask,(batch_size*episode_len,len(action_shape)))
    ##########################################

    noise_vectors = np.reshape(noise_vectors,(batch_size*episode_len,10))
    #stocks = np.reshape(stocks,(batch_size*episode_len,stock_size))
    entropy_decay = tf.constant(entropy_decay, tf.float32)

    a_loss, c_loss, e_loss, grads = \
    actor_critic_sgd_step(actor_critic, states, stocks, actions, rewards, la, action_mask , noise_vectors, ht, ct, behaviour_policy_logits, entropy_decay)

    ac_grads(mean_grads(grads))

    a_loss_m(a_loss)
    e_loss_m(e_loss)
    c_loss_m(c_loss)

    return a_loss, c_loss, e_loss 


# if is_inference:
#     return Model(inputs = [X_input, S_input, A_input, M_input, N_input], outputs = [actions_, logits_, value, hidden_state, cell_state])
# else:
#     return Model(inputs = [X_input, S_input, A_input, M_input, N_input, stored_actions, ht, ct], outputs = [actions_, logits_, value]) #, Model(inputs = [X_input, A_input, M_input, N_input, stored_actions], outputs = value)

@tf.function
def actor_critic_inference(model, state, component_stock, prev_action, action_mask, noise):
    state, component_stock, prev_action, action_mask, noise = \
    tf.cast(state, tf.float32),tf.cast(component_stock, tf.float32),tf.cast(prev_action, tf.float32), tf.cast(action_mask, tf.float32), tf.cast(noise, tf.float32)
    actions, logits, _, ht, ct = model([state, component_stock, prev_action,  action_mask, noise])
    return actions, logits, _, ht, ct


##############################################################################

class A2C_Agent():
    def __init__(self, agent_id, environment, ep_counter, ep_memory):
        np.random.seed()

        self.entropy_decay = 0.1
        self.Actor_Inference = build_policy(batch_size, episode_len, state_size, action_shape, stock_size, is_inference=True)
        self.Actor_Critic = build_policy(batch_size, episode_len, state_size, action_shape, stock_size, is_inference=False)

        self.agent_id = agent_id
        self.env = environment
        self.states, self.next_states, self.stocks, self.actions, self.rewards, self.core_states, self.noise_vectors, self.policy_logits = [], [], [], [], [], [], [], []

        self.debug_state = None

        self.ep_counter = ep_counter
        self.ep_memory = ep_memory

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    def remember(self,state,next_state, stock, action,reward, core_states, noise, policy_logits):
       # store episode actions to memory
        state = np.reshape(state,[1,*state_size])
        next_state = np.reshape(next_state,[1,*state_size])  

        self.states.append(state)
        self.next_states.append(next_state)
        self.stocks.append(stock)
        self.actions.append(action)
        self.rewards.append(reward)
        self.core_states.append(core_state)
        self.noise_vectors.append(noise)
        self.policy_logits.append(tf.squeeze(tf.concat(policy_logits,axis=-1)))

    def act(self,state, stock, prev_action,noise):
        state = np.reshape(state,[1,*state_size])
        # prev_action = np.array(prev_action)
        prev_action = np.expand_dims(prev_action,axis=0) #shape is [1,5]
        stock = np.expand_dims(stock,axis=0)
        # noise = np.random.normal(loc=0.0, scale=1, size=(1,10))

        #define action mask based on previous action content
        if np.sum(prev_action) == 0:
            action_mask = np.zeros((1,len(action_shape)))
        elif prev_action[:,2] == 0: #no placement
            action_mask = np.ones((1,len(action_shape)))
            action_mask[:,2:] = 0
        else:
            action_mask = np.ones((1,len(action_shape)))

        actions, logits, _, ht, ct = actor_critic_inference(self.Actor_Inference,state, stock, prev_action, action_mask, noise) #this function is performed using a single actor inference model
        core_state = [ht,ct]

        return actions, logits, core_state     # actions should be integers
    
    def store_trajectory(self):

        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        stocks = np.vstack(self.stocks)
        actions = np.vstack(self.actions)
        next_states = np.vstack(self.next_states)
        rewards = self.rewards #undiscounted raw rewards (if no rewards come from the environment, all 0s)
        core_states = self.core_states
        noise_vectors = self.noise_vectors
        policy_logits = self.policy_logits

        # store into replay memory and empty arrays

        self.ep_memory.append((states, next_states, stocks, actions, rewards, core_states, noise_vectors, policy_logits)) 
        self.states, self.next_states, self.stocks, self.actions, self.rewards, self.core_states, self.noise_vectors, self.policy_logits = [], [], [], [], [], [], [], []


    def discounted_reward(self,reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99
        running_add = 0
        discounted_r = np.zeros_like(reward, dtype=np.float32)
        print(f'reward collected during the episode: {reward}')
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        return discounted_r

    def train(self): 

        experience = self.ep_memory[0]
        states, _, stocks, actions, rewards, core_states, noise, policy_logits = experience#[0],experience[2],experience[3],experience[4],experience[5], experience[6]

        # rew = np.array(rewards)[:,0]
        # pen = np.array(rewards)[:,1]
        
        # print('rew',rew)
        # print('pen',pen)
        
        # rewards_prev = np.insert(rew[:-1],0,0)
        # tca = rew-rewards_prev

        # print('tca', tca)
        # if self.ep_counter >= 1500:
        #     tca += pen
        #     print('tca + pen', tca)

        discounted_rewards = self.discounted_reward(rewards)

        self.entropy_decay = max(0.00001,self.entropy_decay)



        a_loss,c_loss,e_loss = train_actor_critic(self.Actor_Critic, states, stocks, actions, discounted_rewards, core_states, noise, policy_logits,self.entropy_decay)
        print('n_episodes: ', self.ep_counter , 'a_loss: ',a_loss,'c_loss: ',c_loss,'e_loss: ',e_loss, 'entropy_value: ',self.entropy_decay)
        a_weights = self.Actor_Critic.get_weights()
        self.Actor_Inference.set_weights(a_weights)

        reward_m(rewards[-1])

        if self.ep_counter % 64 == 0:
            self.entropy_decay -= 0.00018002
            e = self.ep_counter

            with self.train_summary_writer.as_default():
                tf.summary.scalar('actor_loss', a_loss_m.result(), step=e)
                tf.summary.scalar('entropy_loss', e_loss_m.result(), step=e)
                tf.summary.scalar('reward', reward_m.result(), step=e)
                tf.summary.scalar('critic_loss', c_loss_m.result(), step=e)                                                              
                tf.summary.scalar('actor_critc_gradients', ac_grads.result(), step=e)


            a_loss_m.reset_states()
            e_loss_m.reset_states()
            reward_m.reset_states()
            c_loss_m.reset_states()
            ac_grads.reset_states()
    
    def plot_graph(self, input_array,color,filename,out_format):
        plt.figure(figsize=(10,5))
        plt.grid()
        axes = plt.gca()        

        if isinstance(color, list): #multiple graphs in the same plot
            n = np.arange(0,len(input_array[0]))            
            for i in range(len(color)):
                plt.plot(n,input_array[i],color[i])
        else:
            n = np.arange(0,len(input_array))            
            plt.plot(n,input_array,color)
        plt.savefig(filename + '.' + out_format, format=out_format) 
        plt.close()

    def load(self,name):
        self.Actor_Critic.load_weights(name)

    def save(self,name):
        self.Actor_Critic.save_weights(name) 


reward_record = []
reward_pen_record = []
mean_reward_record = []
mean_reward_pen_record = []

end_state_reward_record =[]
mean_end_state_reward_record = []
states_coll = []


ep_memory = deque(maxlen=1)
environment = env(state_size)
A2C_Agent = A2C_Agent(agent_id = 0, environment = environment, ep_counter = 0, ep_memory = ep_memory)


for e in range(n_episodes):

    itr = 1
    total_reward = []
    total_pen_reward = []


    A2C_Agent.env.reset()
    stock = A2C_Agent.env.component_stock
    noise = A2C_Agent.env.noise
    state = 1-A2C_Agent.env.state
    done = False

    prev_action=[0 for i in range(len(action_shape))] 


    # run episode
    while itr <= episode_len:

        if itr == episode_len:
            A2C_Agent.env.done = True

        # print(f'*************************************************************        {prev_action}')
        action, logits, core_state = A2C_Agent.act(state, stock, prev_action,noise)
        action = K.eval(action)[0]

        prev_action = action

        # print('action samples ______________________', action)
        state,next_state,stock,reward,done,noise = A2C_Agent.env.step(action.tolist())
        state = 1-state
        next_state = 1-next_state           
        
        total_reward.append(reward)
        #total_pen_reward.append(reward[1])
        itr += 1

        if done:
            A2C_Agent.Actor_Inference.get_layer('lstm').reset_states()
            A2C_Agent.debug_state = next_state

        A2C_Agent.remember(state,next_state,stock,action,reward,core_state,noise,logits)
        

    A2C_Agent.store_trajectory() #append to queue
    
    
    A2C_Agent.ep_counter += 1 #update global counter
    local_ep_value = A2C_Agent.ep_counter
   
    if local_ep_value % 10 == 0: #plot every 16 updates
        A2C_Agent.env.plot_state(A2C_Agent.debug_state,'episode_{}_from_agent_{}'.format(local_ep_value,A2C_Agent.agent_id))
        # A2C_Agent.env.solver.visualise_model(view_deflected=False)
        # print('extra reward is: ',reward)


    reward_record.append(np.sum(total_reward))
    #reward_pen_record.append(np.sum(total_pen_reward))
    end_state_reward_record.append(A2C_Agent.env.reward)#fem_reward[0]) # only the last reward before end of the episode

    # print('sequence length', len(total_reward))

    A2C_Agent.train()


    if e % 50 == 0:
        last_mean_reward = np.mean(reward_record[-50:])
        last_mean_pen_reward = np.mean(reward_pen_record[-50:])
        
        mean_reward_record.append(last_mean_reward)
        mean_reward_pen_record.append(last_mean_pen_reward)

        mean_end_state_reward_record.append(np.mean(end_state_reward_record[-50:]))



        if last_mean_reward>= max(mean_reward_record) and e > 1500:
            A2C_Agent.save(f'checkpoints/{e}.h5')



    if e % 100 == 0:


        A2C_Agent.plot_graph(mean_reward_record,'FireBrick','mean_reward','pdf')
        A2C_Agent.plot_graph(mean_reward_pen_record,'g','mean_reward_pen','pdf')
        A2C_Agent.plot_graph(mean_end_state_reward_record,'b','end_state_mean_reward','pdf')

        # env.plot_state(next_state,'episode_{}'.format(e)) 



    #dones = [m[4] for m in agent.memory if m[4]][-100:]
    #print('mean_score', sum(dones)/len(dones), e) #average every 100 steps stored in memory