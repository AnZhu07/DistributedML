import torch
import torchvision
import torch.nn.functional as F
import copy

class decentralised:
#aim to include function for step training and averaging

    def __init__(self, train_loaders, model, optimizers, graph):
        self.train_loaders = train_loaders #list of train loaders
        self.model = model   #models  ->dictionary list
        self.optimizers = optimizers #list of optimizers for each agent
        self.graph = graph  #graph struction in adj matrix
        
        ###class variable below
        self.len_agents = len(train_loaders) #number of agents in the network
        self.batch_idx_set = [0]*self.len_agents  #list of counters for each agent to record which batch has been trained
        self.iterator_set = [iter(train_loader) for train_loader in train_loaders]   #list of iterated train_loader for each agent
        self.train_finish = False  #flag indicate if batch idx reached the last batch for training, initialsed to be false
        
    def step_train(self, step_interval):
        
        if self.batch_idx_set[self.len_agents-1] < len(self.train_loaders[self.len_agents-1]):
            self.train_finish = False  #starting step train or not finished step train 
        
        for agent_no in range(self.len_agents):
            self.model[agent_no].train()
            while self.batch_idx_set[agent_no] < len(self.train_loaders[agent_no]):
                (data, target) = next(self.iterator_set[agent_no])
                self.optimizers[agent_no].zero_grad()
                output = self.model[agent_no](data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizers[agent_no].step()
                if (self.batch_idx_set[agent_no]+1) % step_interval == 0:
                    self.batch_idx_set[agent_no] += 1
                    break
                self.batch_idx_set[agent_no] += 1

        if self.batch_idx_set[self.len_agents-1] == len(self.train_loaders[self.len_agents-1]):
            #this means the all agents trained through all batch
            #change flag, initialise counters, re-iter train_loaders for nect epoch
            self.iterator_set = [iter(train_loader) for train_loader in self.train_loaders]
            self.batch_idx_set = [0]*self.len_agents
            self.train_finish = True


    def avg_n(self):
        print('avg')
        model_set = []
        for idx in range(self.len_agents):
            model_set.append(copy.deepcopy(self.model[idx].state_dict()))  #copy all model
        avg_dict = {}
        for key in model_set[0]:
            avg_dict[key]=0  #initialise copy dict
        for j in range(self.len_agents):
        #for each agent: perform local averaing base on the connection graph
            for index in range(len(model_set)):
                count = 0
                for key in model_set[0]:
                    avg_dict[key]=0 
                for i in range(len(self.graph[index])):
                    count = sum(node for node in self.graph[index])  #count the number of agents connected
                    if self.graph[index][i] == 1:  
                        for key in model_set[i]:
                            avg_dict[key]+=(model_set[i][key])*(1/count)  #avg each parameter for agents connected
               
                self.model[j].load_state_dict(avg_dict) #update to the network list
        #print('averaging')
