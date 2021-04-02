import gym
import numpy as np
import tensorflow as tf
import os
from RenderTimer import waitForNextFrame
from LLC import LLC
import cv2


def randomNormalClamped(mean,sd,size=None):
    result=np.random.normal(mean,sd,size)
    result=np.clip(result,mean-2.0*sd,mean+2.0*sd)
    return result

rendering_buffer = []
def render_scene(sim, size=600):
    rendering_buffer.append(sim.render(width=size, height=size, mode="rgb_array"))

def save_video(folder='', file='video', size=800, fps=50):
	path = os.path.join(os.getcwd(), 'videos')
	os.makedirs(os.path.join(os.getcwd(), 'videos'), exist_ok=True)
	if folder != '':
		path = os.path.join(path, folder)
		os.makedirs(path, exist_ok=True)

	path = os.path.join(path, file + '.mp4')

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	writer = cv2.VideoWriter(path, fourcc, fps // 5, (size, size))

	for img in rendering_buffer:
		writer.write(img)

	# cv2.destroyAllWindows()
	writer.release()

	rendering_buffer.clear()

	print('Saved video: %s' % path)


aliveBonusMultipliers={"Hopper-v2":5,"Walker2d-v2":20,"Humanoid-v2":2,"HalfCheetah-v2":2,"Ant-v2":1}

class EnvWrapper:
    def __init__(self,envName,taskName="",actionFrequency=10.0,ignoreDone=False, timeout=0):
        if taskName=="":
            self.task=None
        else:
            self.task=taskName
        assert(taskName=="" or taskName=="walk" or taskName=="standup" or taskName=="walkbwd" or taskName=="run" or taskName=="balance")

        self.env=gym.make(envName)
        self.sim=self.env.unwrapped.sim

        self.actionFrequency = actionFrequency
        self.actionRepeat=int(1.0/self.env.dt/actionFrequency)
        self.envName=envName
        self.timeout = timeout
        #xIdx and yIdx are for 2D plotting of movement trajectories
        if envName=="HumanoidStandup-v2" or envName=="Humanoid-v2" or envName=="Ant-v2":
            self.xIdx=1
            self.yIdx=2
            self._excludedFirstObs=2  #of global position, the agent only observes y
            self.nRootRot=4
            self.is3d=True
            self.nRootPos=3
        elif envName=="Hopper-v2" or envName=="HalfCheetah-v2" or envName=="Walker2d-v2":
            self.xIdx=0
            self.yIdx=1
            self.nRootRot=1
            self._excludedFirstObs=1  #of global position, the agent only observes y
            self.is3d=False
            self.nRootPos=2
        else:
            raise Exception("Unsupported environment:",envName)
        self.action_space=self.env.action_space
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=self.observe().shape)
        self.done=False
        self.ignoreDone=ignoreDone
        self.obsDim=self.observation_space.shape[0]
        self.actionDim=self.action_space.shape[0]

        self.resetQpos=self.env.init_qpos.copy()
        self.resetQvel=self.env.init_qvel.copy()
        if self.task=="standup":
            if envName=="HalfCheetah-v2":
                #The HalfCheetah has stable balance in the initial position.
                #Thus, we reset it on its back.
                self.resetQpos[2]=1.0*np.pi 
                #lift up a bit, to allow the agent to fall initially, giving it a bit of momentum to utilize. otherwise, this task is too hard.
                self.resetQpos[1]+=0.2 
            elif envName=="Hopper-v2" or envName=="Walker2d-v2":
                self.resetQpos[2]=-0.5*np.pi 
                self.resetQpos[1]-=0.8 #drop down closer to floor 
            elif envName=="Humanoid-v2":
                self.resetQpos[3:7]=[0.707,0,-0.707,0]
                self.resetQpos[2]-=0.5 #drop down closer to floor 
            else:
                raise Exception("Stand up task not supported on this env")
        self.len = 0
        self.rew = 0
        self.max_ep_len = self.env._max_episode_steps
        #if random_init:
        #    self.max_ep_len = self.max_ep_len // 1

    def observe(self):
        if self.sim is None:
            return self.env.observe()
        else:
            return np.concatenate([self.sim.data.qpos[self._excludedFirstObs:],self.sim.data.qvel])
    #For the predefined tasks, this returns the target observation used in computing the reward.
    #When using the LLC, this can be used as a reasonable prior mean for the actions (i.e., target observations the LLC tries to reach).
    def getTargetObs(self):
        if self.task is None:
            return None
        targetObs=np.concatenate([self.env.init_qpos[self._excludedFirstObs:],self.env.init_qvel])
        #if self.envName=="HalfCheetah-v2":
        #    targetObs[1]-=np.pi*0.3  #HalfCheetah wants to balance on one leg
        diffWeights=np.ones(self.obsDim)
        #self.done=diff[0]>0.3                                           #terminate if y much lower than in init
        if self.task=="walk":
            targetObs[self.env.model.nq-self._excludedFirstObs]=1.0      #target x vel 
        elif self.task=="walkbwd":
            targetObs[self.env.model.nq-self._excludedFirstObs]=-1.0     #target x vel 
        elif self.task=="run":
            targetObs[self.env.model.nq-self._excludedFirstObs]=4      #target x vel 
        #if self.envName=="HalfCheetah-v2":
        #    targetObs[1]-=np.pi*0.3  #HalfCheetah wants to balance on one leg
        return targetObs

    def mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def step(self, action, render=False, render_online=True):
        totalReward = 0
        info = dict()
        if self.ignoreDone or not self.done:
            for subStep in range(self.actionRepeat):
                _, r, self.done, info = self.env.step(action)
                if self.task is not None:
                    # Recompute the task-specific rewards
                    actionCost=0.01*np.mean(np.square(action))
                    targetObs=self.getTargetObs()
                    diff=targetObs-self.observe()
                    if self.ignoreDone or self.task == "standup":
                        self.done = False
                    else:
                        self.done = diff[0] > 0.2 # Terminate if y much lower than in init
                    if self.len >= self.max_ep_len:
                        self.done = True
                    diffWeights=np.ones(self.obsDim)
                    if self.task=="walk" or self.task=="walkbwd":
                        #for locomotion tasks, increas velocity difference weight
                        diffWeights[self.env.model.nq-self._excludedFirstObs]=50 
                    elif self.task=="run":
                        diffWeights[self.env.model.nq-self._excludedFirstObs]=100
                    elif self.task=="standup":
                        #in get up task, give higher weight to root rotation and vertical position
                        diffWeights[:self.nRootPos+self.nRootRot-self._excludedFirstObs]=50
                   
                    #if self.envName=="HalfCheetah-v2":
                    #    diff[1]*=50.0
                    #observation cost = weighted mean of observation variable differences to target
                    obsCost=np.sum(np.square(diff*diffWeights))/np.sum(np.square(diffWeights)) 
                    r=-obsCost-actionCost
                    #r=(1.0+r/(1+abs(r)))*self.env.dt #maximum reward = 1 per second
                    if self.ignoreDone or self.task=="standup":
                        aliveBonus=0
                    else:
                        aliveBonus=aliveBonusMultipliers[self.envName]*self.obsDim
                        r = (aliveBonus + r) / aliveBonus * self.env.dt   #maximum reward = 1 per second
                    if r < 0 and not (self.ignoreDone or self.task=="standup"):
                        print("Warning! Negative reward used with termination, cost=",actionCost+obsCost)
                        r = 0  #to prevent early termination exploits
                if render:
                    waitForNextFrame(self.env.dt)
                    if render_online:
                        self.env.render()
                    else:
                        render_scene(self.env, size=600)
                totalReward += r
                self.len += 1
                if self.len * self.env.dt >= self.timeout > 0:
                    self.done = True
                self.rew += r
                if self.done and not self.ignoreDone:
                    break
        if self.done:
            info['episode'] = { 'l': self.len, 'r': self.rew }
        return self.observe(), totalReward, self.done, info
    def getState(self):
        if self.sim is None:
            return self.env.getState()
        else:
            return self.sim.get_state()
    def setState(self,state):
        self.reset()
        if self.sim is None:
            return self.env.setState(state)
        else:
            self.sim.set_state(state)
    def reset(self,fixed=False):
        self.env.reset()
        if fixed or self.task == "standup":
            self.env.set_state(self.resetQpos,self.resetQvel)
        self.len = 0
        self.rew = 0
        self.done=False
        return self.observe()
    def setStateFromObs(self,obs):
        self.reset(fixed=True)
        sim=self.env.sim
        state=sim.get_state()
        state.qpos[self._excludedFirstObs:]=obs[:self.nPosAndRotObs]
        state.qvel[:]=obs[self.nPosAndRotObs:]
        sim.set_state(state)
        self.done=False

    @property
    def dt(self):
        return self.actionRepeat*self.env.dt
    @property
    def nPosAndRotObs(self):
        if self.sim is None:
            return 1
        else:
            return self.env.model.nq-self._excludedFirstObs
    def getRootObsIndices(self):
        if self.envName=="HumanoidStandup-v2" or self.envName=="Humanoid-v2" or self.envName=="Ant-v2":
            #root y position, quaternion
            result=[0,1,2,3,4] 
            #root velocity, angular velocity
            for i in range(6):
                result.append(self.env.model.nq-self._excludedFirstObs+i)
        elif self.envName=="HalfCheetah-v2" or self.envName=="Walker2d-v2" or self.envName=="Hopper-v2":
            #root y position, rotation
            result=[0,1]
            #root velocity, angular velocity
            for i in range(3):
                result.append(self.env.model.nq-self._excludedFirstObs+i)
        else:
            raise Exception("Invalid environment:",self.envName)

'''
Wrapper for a number of EnvWrapper instances, supports batched observations and actions of shape [N,action dims] and [N,observation dims].
The wrapper also provides an optional modified actions space, where actions denote normalized target observations and a low-level controller (LLC)
drives the agent to follow a target observation sequence. If the LLC has not been trained (no file found), the constructor trains it.

Params:

envName             MuJoCo environment name
N                   Number of wrapped environments. If N=1, all actions and observations are expected to not be batched, i.e., of shape
                    [action dims] and [observation dims]. Thus, the wrapper behaves just like a single environment and the instance can be passed to
                    RL algorithms expecting a single environment.
taskName            "" (default MuJoCo), or one of: "balance", "walk", "run", "standup", "walkbwd"
actionMode          "default" or "LLC". Default means the original MuJoCo environment's action space, LLC denotes using the low-level controller.
                    If LLC has not been trained, instantiating the wrapper will train it.
llcData             The exploration data used for trining the LLC, either "ContactExplorer2" or "NaiveExplorer"
H                   LLC horizon, in time steps.
singleTargetObs     If True, the LLC takes in only a single target observation which it tries to reach in H steps
sess                Tensorflow session. Only needed if actionMode=="LLC"
actionFrequency     The wrapper repeats the wrapped environment's actions to approximate this target action frequency. Note: action frequencies
                    higher than the wrapped environment's default have no effect.
ignoreDone          If True, the episode termination signaled by the wrapped environments is ignored. This is needed, e.g., 
                    for random state exploration.
'''
class MultiEnvWrapper:
    def __init__(self, envName, N, reset_mode=None, taskName="", actionMode="default", llcData=None, H=3
                 , singleTargetObs=False, sess:tf.Session=None, actionFrequency=10.0, ignoreDone=False
                 , render_online=True, timeout=0, **kwargs):
        self.envs=[]
        self.render_online = render_online
        for n in range(N):
            self.envs.append(EnvWrapper(envName,taskName,actionFrequency,ignoreDone=ignoreDone, timeout=timeout))
            if not render_online and n == 0:
                self.envs[0].env.render(mode="human")
            self.envs[n].reset(fixed=True)
        self.initialState=self.envs[0].getState()
        self.obsDim=self.envs[0].observe().shape[0]
        self.N=N
        self.H=H
        self.Heff=1 if singleTargetObs else H
        self.singleTargetObs=singleTargetObs
        self.dones=np.ndarray(dtype=np.bool,shape=self.N)
        self.dones[:]=False
        self.obsRangesExplored=False
        self.ignoreDone=ignoreDone
        self.envName=envName
        self.actionMode=actionMode
        self.action_space=self.envs[0].action_space
        self.observation_space=self.envs[0].observation_space
        self.metadata = ""  # Dummy variable
        self.reset_mode = reset_mode
        self.timeout = timeout

        #If needed, create a low-level controller
        if actionMode=="LLC":
            if "Humanoid" in envName:
                nNeurons=128
                nHidden=3
                trainIterBudget=16384
            else:
                nNeurons=64
                nHidden=2
                trainIterBudget=8192
            self.actionMode="default"  #while training, temporarily disable the llc
            self.llc=LLC(sess,self,H=H,dataFile="ExplorationData/{}_{}.npy".format(envName,llcData),singleTargetObs=singleTargetObs,trainIterBudget=trainIterBudget,nNeuronsPerLayer=nNeurons,nHidden=nHidden, **kwargs)
            self.actionMode="LLC"
            #The LLC takes in normalized observation space. The bounds correspond to +- 1 standard deviation of the training data observations
            self.action_space=gym.spaces.Box(-1,1,shape=[self.obsDim*self.Heff])

            self.targetObsMean=None
            self.targetObsSd=None

    def save_video(self, name='untitled'):
        save_video(folder=os.path.join(os.getcwd(), 'Videos'), file=name, size=600, fps=int(1 / self.envs[0].env.model.opt.timestep))

    def getRootObsIndices(self):
        return self.envs[0].getRootObsIndices()
    def setIgnoreDone(self,ignoreDone):
        self.ignoreDone=ignoreDone
        for env in self.envs:
            env.ignoreDone=ignoreDone
    def observe(self):
        obs=np.zeros([self.N,self.obsDim])
        for n in range(self.N):
            obs[n,:]=self.envs[n].observe()
        if self.N==1:
            obs=obs[0]
        return obs
    def step(self,actions,render=False,realize_idx=-1):
        if len(actions.shape)==1:
            #Internally, we always use batched actions
            actions=np.reshape(actions,[1,-1])
        obs=np.zeros([self.N,self.obsDim])
        rewards=np.zeros(self.N)
        nSimulated=0
        futures=[]

        #Convert target observations to actions
        if self.actionMode=="LLC":
            actions=np.reshape(actions,[self.N,self.Heff,-1])
            actions=self.llc.driveTowards(self.observe(),actions,self.targetObsMean,self.targetObsSd)

        #Clamp actions to range
        actions = np.clip(actions, np.reshape(self.envs[0].action_space.low, [1,-1]), np.reshape(self.envs[0].action_space.high, [1,-1]))

        info = dict()
        for n in range(self.N):
            if self.ignoreDone or not self.dones[n]:
                 # and (n==0))
                obs[n],rewards[n],self.dones[n],info=self.envs[n].step(action=actions[n],render=render and n == 0
                                                                       , render_online=self.render_online)
                nSimulated+=1
            else:
                obs[n]=self.envs[n].observe()
        info['nSimulated'] = nSimulated
        if realize_idx >= 0:
            info['realized_actions'] = actions[realize_idx]
        if self.N==1:
            return obs[0],rewards[0],self.dones[0],info
        else:
            return obs,rewards,self.dones,info
    def step_single_env(self,actions, idx,render=False):
        if len(actions.shape)==1:
            #Internally, we always use batched actions
            actions=np.reshape(actions,[1,-1])
        obs=np.zeros([1,self.obsDim])
        rewards=np.zeros(1)

        #Convert target observations to actions
        if self.actionMode=="LLC":
            actions=np.reshape(actions,[1,self.Heff,-1])
            actions=self.llc.driveTowards(self.observe()[idx:idx+1,:],actions,self.targetObsMean,self.targetObsSd)

        #Clamp actions to range
        actions=np.clip(actions,np.reshape(self.envs[0].action_space.low,[1,-1]),np.reshape(self.envs[0].action_space.high,[1,-1]))

        info = dict()
        if self.ignoreDone or not self.dones[idx]:
            obs[0], rewards[0], self.dones[idx], info = self.envs[idx].step(action=actions[0]
                                                                            , render=render and idx == 0
                                                                            , render_online=self.render_online)
            info['nSimulated'] = 1
        else:
            obs[0]=self.envs[idx].observe()
            info['nSimulated'] = 0
        return obs[0],rewards[0],self.dones[idx],info
    def reset(self,mode="", target_start_height=0.5, start_vel_scale=1.0, start_rot_scale=1.0):
        if self.timeout > 0:
            mode = "fixed"
        if self.reset_mode is not None:  # Override mode with self.reset_mode if it exists (necessary for RL tests)
            mode = self.reset_mode
        if mode!="" and mode!="random" and mode!="fixed":
            raise Exception("Invalid reset mode: ",mode)
        if mode=="random":
            if not self.ignoreDone:
               raise Exception("Random state reset incompatible with episode termination. To avoid strange results, use ignoreDone=True.")
            result=self._contactExplorationStateInit(target_start_height,start_vel_scale,start_rot_scale)
        else:
            result=np.zeros([self.N,self.obsDim])
            for n in range(self.N):
                self.envs[n].reset()
                if mode=="fixed":
                    self.envs[n].setState(self.initialState)
                result[n]=self.envs[n].observe()
        self.dones[:]=False
        if self.N==1:
            result=result[0]
        return result
    def close(self):
        self.envs = []  # Dummy function
    def copyState(self,src,dst):
        self.envs[dst].setState(self.envs[src].getState())
    def setStateForAll(self, state):
        self.reset()
        for n in range(self.N):
            self.envs[n].setState(state)
    @property
    def dt(self):
        return self.envs[0].dt
    @property
    def nPosAndRotObs(self):
        return self.envs[0].nPosAndRotObs
    def setStateFromObs(self,obs):
        for n in range(self.N):
            self.envs[n].setStateFromObs(obs[n])

    #Initialize the character randomly such that it will explore all possible contact configurations well.
    #Note: we implement this in MultiEnvWrapper instead of EnvWrapper to avoid doing the initial exploration phase multiple times.
    def _contactExplorationStateInit(self,target_start_height,start_vel_scale,start_rot_scale):
        if self.envs[0].sim is None or self.envs[0].nRootPos==0:
            raise Exception("Contact exploration not supported with this environment!")
        yIdx=self.envs[0].yIdx  #convenience
        nRootPos=self.envs[0].nRootPos
        nRootRot=self.envs[0].nRootRot
        is3d=self.envs[0].is3d
        epsilon=0.01
        y_max=1.0       #something high enough such that the character is not in contact with ground
        #At first call: To measure joint angle and angular velocity observation limits, 
        #hold the character in the air, while driving the joints with minimum and maximum actions
        if not self.obsRangesExplored:
            self.obsRangesExplored=True

            print("Phase 1: Measuring qpos and qvel min and max while the character is not in contact with ground...")
            env=self.envs[0]
            sim=env.sim
            env.reset()
            minQpos=sim.data.qpos.copy()
            maxQpos=sim.data.qpos.copy()
            minQvel=sim.data.qvel.copy()
            maxQvel=sim.data.qvel.copy()
            state=sim.get_state()
            state.qpos[yIdx]+=y_max
            initialY=state.qpos[yIdx]
            sim.set_state(state)

            for i in range(500):
                state=sim.get_state()
                action=env.action_space.sample()
                env.step(action, render=True, render_online=self.render_online)
                minQpos=np.minimum(minQpos,sim.data.qpos)
                maxQpos=np.maximum(maxQpos,sim.data.qpos)
                minQvel=np.minimum(minQvel,sim.data.qvel)
                maxQvel=np.maximum(maxQvel,sim.data.qvel)
                if inContactWithGround(env.env):#sim.data.ncon>0:
                    #print_contact_info(env)
                    #reset back to air when the character makes contact with ground
                    state.qpos[:nRootPos]=0
                    state.qpos[yIdx]=initialY
                    state.qvel[:nRootPos]=0
                    sim.set_state(state)
            print("min qpos",minQpos)
            print("max qpos",maxQpos)
            print("min qvel",minQvel)
            print("max qvel",maxQvel)
            self.minQpos=minQpos
            self.maxQpos=maxQpos
            self.maxJointAngularSpeed=0.5*(np.abs(minQvel)+np.abs(maxQvel))

        #Determine randomization parameters           
        speedSd=4.0*start_vel_scale
        rootAngularSpeedSd=0.5*np.pi*start_vel_scale
        jointAvelSd=np.pi*np.ones(self.envs[0].sim.data.qvel.shape[0])*start_vel_scale
        #jointAvelSd=self.maxJointAngularSpeed*0.25*start_vel_scale
        jointAngleMean=0.5*(self.minQpos+self.maxQpos)[nRootPos+nRootRot:]
        jointAngleSd=0.25*(self.maxQpos-self.minQpos)[nRootPos+nRootRot:]
        jointAngleMin=self.minQpos[nRootPos+nRootRot:]
        jointAngleMax=self.maxQpos[nRootPos+nRootRot:]

        #Randomize all wrapped envs
        result=np.zeros([self.N,self.obsDim])  #will hold the observations
        for n in range(self.N):
            env=self.envs[n]
            sim=env.sim
            env.reset()
            state=sim.get_state()

            #Randomize root rotation
            if is3d and start_rot_scale>0:
                #random quaternion
                quat=np.random.normal(size=4)
                quat/=np.sqrt(np.dot(quat,quat))
                #lerp from original neutral reset quaternion towards the randomized one
                quat=start_rot_scale*quat+(1.0-start_rot_scale)*state.qpos[nRootPos:nRootPos+nRootRot]
                quat/=np.sqrt(np.dot(quat,quat))    #normalize the lerped quaternion
                #apply
                state.qpos[nRootPos:nRootPos+nRootRot]=quat  
            else:
                state.qpos[nRootPos]=np.random.uniform(-np.pi*start_rot_scale,np.pi*start_rot_scale)

            #Randomize root velocity
            state.qvel[:nRootPos]=randomNormalClamped(0,speedSd,size=nRootPos)

            #Randomize root angular velocity
            if is3d:
                state.qvel[nRootPos:nRootPos+3]=np.random.normal(0,1.0,size=3)
                state.qvel[nRootPos:nRootPos+3]/=np.linalg.norm(state.qvel[nRootPos:nRootPos+3])
                state.qvel[nRootPos:nRootPos+3]*=randomNormalClamped(0,rootAngularSpeedSd)
            else:
                state.qvel[nRootPos]=randomNormalClamped(0,rootAngularSpeedSd)

            #Randomize joint angular velocities
            state.qvel[nRootPos:]=randomNormalClamped(np.zeros(state.qvel.shape[0]-nRootPos),jointAvelSd[nRootPos:])
                                                   #-start_vel_scale*self.maxJointAngularSpeed[nRootPos:],
                                                   
                                                  #start_vel_scale*self.maxJointAngularSpeed[nRootPos:])

            #Randomize joint angles. Note: need to clip to avoid violating joint limits, which might cause simulation glitches
            #state.qpos[nRootPos+nRootRot:]=np.random.normal(jointAngleMean,jointAngleSd)
            #state.qpos[nRootPos+nRootRot:]=np.clip(state.qpos[nRootPos+nRootRot:],jointAngleMin,jointAngleMax)
            randomAngles=np.random.uniform(jointAngleMin,jointAngleMax)
            state.qpos[nRootPos+nRootRot:]+=start_rot_scale*(randomAngles-state.qpos[nRootPos+nRootRot:])
  

            #Uncomment to simply test and visualize the initialization
            #sim.set_state(state)
            #for i in range(100):
            #    env.step(np.zeros(actionDim))
            #    if render:
            #        env.render()
            #        time.sleep(1.0/100.0)
            #Move the initial state downwards until the character comes into contact with ground
            #(MuJoCo doesn't give us the character's bounding box, so we need to resort to this "empirical" method)
            state.qpos[yIdx]+=y_max   #To start, lift the character upwards from the default initial position 
                                      #(otherwise, it might be penetrating the ground because of the randomization above)
            while (state.qpos[yIdx]>0):
                sim.set_state(state)
                env.env.step(np.zeros(env.action_space.shape[0])) #we use the unwrapped env to step only one real timestep instead of repeating
                #if render:
                #    env.render()
                #    time.sleep(1.0/100.0)
                if inContactWithGround(env.env):
                    #revert to previous state to avoid initializing in a ground-penetrating state
                    sim.set_state(state)
                    break
                state.qpos[yIdx]-=epsilon
  
            #Readjust to desired start height
            #state.qpos[yIdx]+=epsilon
            #sim.set_state(state)
            if target_start_height>0:
                state.qpos[yIdx]+=target_start_height
                sim.set_state(state)
            result[n]=env.observe()
        return result

def inContactWithGround(env):
    d = env.data
    for coni in range(d.ncon):
        con = d.contact[coni]
        if con.geom1==0 or con.geom2==0:
            return True
    return False


#Helper for debugging contacts
def str_mj_arr(arr):
    return ' '.join(['%0.3f' % arr[i] for i in range(arr.shape[0])])

def print_contact_info(env):
    d = env.unwrapped.data
    for coni in range(d.ncon):
        print('  Contact %d:' % (coni,))
        con = d.contact[coni]
        print('    dist     = %0.3f' % (con.dist,))
        print('    pos      = %s' % (str_mj_arr(con.pos),))
        print('    frame    = %s' % (str_mj_arr(con.frame),))
        print('    friction = %s' % (str_mj_arr(con.friction),))
        print('    dim      = %d' % (con.dim,))
        print('    geom1    = %d' % (con.geom1,))
        print('    geom2    = %d' % (con.geom2,))
