# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:37:27 2017

@author: Dimitri Cabaud
"""

from mesa.datacollection import DataCollector
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.time import RandomActivation
import random as rd
from scipy.stats import poisson
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt


class CallCenterAgent(Agent):
    
    def __init__(self, unique_id, N, model):
        super().__init__(unique_id, model)
        
        self.wealth = 0
        self.nb_step = 0
        self.mu = 1
        self.loc = 2
        self.scale = 1.5
        self.schedule = SimultaneousActivation(self)
        self.list_taxis = []
        self.list_taxis_wealth = []
        self.number_taxi_working = 0

        # Create taxis agents        
        for i in range(N):
            a = TaxiAgent(i, self)
            self.schedule.add(a)
            self.list_taxis.append(a)
            self.list_taxis_wealth.append(a.wealth)
        
    
    def changement_day(self):
        if self.nb_step == 1440:
            self.nb_step = 0
            
    def number_agent_working(self):
        self.number_taxi_working = 0 
        for j in range(len(self.list_taxis)):  
            self.number_taxi_working += self.list_taxis[j].is_working
        
    
    def call_client(self):
        list_nodes = G.nodes()
        list_clients = [poisson.rvs(self.mu/60) for x in range(len(list_nodes))]
        list_dst = [abs(norm.rvs(self.loc, self.scale)*x) for x in list_clients]
        list_call = [[list_nodes[x],list_clients[x], list_dst[x]] for x in range(len(list_clients)) if list_clients[x]!=0]        
        return list_call
        
    
    def launch_bid(self): 
        list_call  = self.call_client()
        offers, taxis_winners, dst_clients, client_nodes, taxis_wealth = [],[],[],[],[]
        for j in range(len(self.list_taxis)):
            if self.list_taxis[j].respect_fair_rule >= 1:
                self.list_taxis[j].respect_fair_rule -= 1                  

        for i in range(len(list_call)):
            if list_call[i][2] == 0 :
                pass
            else:
                best_offer = 0
                best_taxi = self.list_taxis[0]
                dst_client = 0
                client_node = '8_3'
                taxi_wealth = 0
                for j in range(len(self.list_taxis)):                
                    self.list_taxis[j].is_enough_time_for_bid()
                    if self.list_taxis[j].is_enough_time == 0 or self.list_taxis[j].is_working == 0 or self.list_taxis[j].respect_fair_rule>0:
                        pass
                    else:
                        current_offer, potential_taxi_wealth = self.list_taxis[j].bid(self.list_taxis[j].stack_position[-1],self.list_taxis[j].stack_node_start_1[-1], self.list_taxis[j].stack_node_start_2[-1],list_call[i][0], list_call[i][2])
                        if  current_offer > best_offer:
                            best_offer = current_offer
                            best_taxi = self.list_taxis[j]
                            dst_client = list_call[i][2]
                            client_node = list_call[i][0]
                            taxi_wealth = potential_taxi_wealth
                            self.list_taxis[j].respect_fair_rule += int((dst_client/30)*60)
                offers.append(best_offer)
                taxis_winners.append(best_taxi)
                dst_clients.append(dst_client)
                client_nodes.append(client_node)
                taxis_wealth.append(taxi_wealth)
                self.wealth += best_offer
        return taxis_winners, dst_clients, client_nodes, taxis_wealth
    
    def step(self):
        
        self.changement_day()
        
        self.nb_step += 1
        if self.nb_step < 420 or self.nb_step >= 1380:
            self.mu = 1
        elif (self.nb_step >= 540 and self.nb_step < 1020) or (self.nb_step >= 1140 and self.nb_step < 1380):
            self.mu = 2
        else:
            self.mu = 3 
        
        self.number_agent_working()
            
        taxis_winners, dst_clients, client_node, taxis_wealth = self.launch_bid()
        j = 0
        for i in taxis_winners:

            path, position, a = i.len_and_path_position_to_node(i.stack_position[-1], i.stack_node_start_1[-1], i.stack_node_start_2[-1], client_node[j])
            i.stack_path.append(path)
            i.stack_position.append(position)
            
            path, position, noeud_dest1, noeud_dest2 = i.path_node_to_dst(client_node[j], dst_clients[j])
            i.stack_path.append(path)
            i.stack_position.append(position)
            i.stack_node_start_1.append(noeud_dest1)
            i.stack_node_start_2.append(noeud_dest2)
            
            i.wealth += taxis_wealth[j]
            j+=1
        self.schedule.step()
        
    
class TaxiAgent(Agent):
 
    G = nx.read_edgelist("edges_graph.txt", delimiter=",", data=[("weight", int)]) 
    G.edges(data=True)
    number_agent = 0
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 0 
        self.nb_step = 0
        self.is_enough_time  = 1
        self.respect_fair_rule = 0
        
        self.stack_position = [0]
        self.stack_node_start_1 = ['8_3']
        self.stack_node_start_2 = ['8_3']
        self.stack_client_node = ['8_3']
    
        self.stack_path = []

        self.state_move = 0 
        self.position_travelling = 0

        self.charge_rate = 60
        self.gas_cost = 4
        self.commission_company = 0.3

        if self.unique_id <= 3:
            self.group = 1
        elif self.unique_id >= 4 and self.unique_id <= 7:
            self.group = 2
        else:
            self.group = 3
            
        self.is_working = 0
        self.restriction_rule = 0
        self.is_travelling = 0
        
        TaxiAgent.number_agent += 1
        self.number_agent = TaxiAgent.number_agent
    
    def changement_day(self):
        if self.nb_step > 1440:
            self.nb_step = 0
    
    def del_first_element_stack_path(self, size_path):
        if size_path >= 2:
            del self.stack_path[0][0]
        else:
            del self.stack_path[0]

    def estimation_time(self):
        time_path = 0
        for i in range(len(self.stack_path)-1):            
            for j in range(len(self.stack_path[i])):
                if len(self.stack_path[i])>j+1:
                    second_node = self.stack_path[i][j+1]
                else:
                    second_node = self.stack_path[i+1][0]
                dst = nx.dijkstra_path_length(G,self.stack_path[i][j],second_node)
                time_path+=dst
        return time_path/0.5
                
        
    def is_enough_time_for_bid(self):
        time_path = self.estimation_time()
        if self.nb_step+time_path <= 780 and (self.group == 1):
            self.is_enough_time = 1
        
        elif self.nb_step+time_path <= 1140 and (self.group == 2):
            self.is_enough_time =  1
            
        elif (self.nb_step+time_path <= 240) and (self.group == 3) and (self.nb_step <= 240):
            self.is_enough_time =  1
        
        elif (self.nb_step+time_path <= 1440) and (self.group == 3) and (self.nb_step > 1000):
            self.is_enough_time =  1

        else:
            self.is_enough_time =  0

    def move(self):
        one_step = 0.5
        max_while = 0
        while(max_while < 10):
            try:
                size_path = len(self.stack_path[0])
            except:
                size_path = 0
            try:
                if size_path >= 2:
                    second_node = self.stack_path[0][1]
                else:
                    second_node = self.stack_path[1][0]
            except:
                break

            if second_node == self.stack_path[0][0]:
                self.del_first_element_stack_path(size_path)


            elif self.state_move == 0 and len(self.stack_path)>0: 
                if  G[self.stack_path[0][0]][second_node]['weight'] > (self.position_travelling + one_step):
                    self.position_travelling += one_step
                    break
                else:
                    self.position_travelling = self.position_travelling + one_step - G[self.stack_path[0][0]][second_node]['weight']
                    if size_path == 1:
                        del self.stack_position[0]
                        self.state_move+=1
                    self.del_first_element_stack_path(size_path)
                    one_step = 0
            
            elif self.state_move == 1 and len(self.stack_path)>0:
                if  G[self.stack_path[0][0]][second_node]['weight'] > (self.position_travelling + one_step):
                    self.position_travelling += one_step
                    break
                else:
                    self.position_travelling = self.position_travelling + one_step - G[self.stack_path[0][0]][second_node]['weight']
                    if size_path == 1:
                        self.state_move+=1
                    self.del_first_element_stack_path(size_path)
                    one_step = 0
    
            elif self.state_move == 2 and len(self.stack_path)>0:
                if  G[self.stack_path[0][0]][second_node]['weight'] > (self.position_travelling + one_step):
                    self.position_travelling += one_step
                    break
                else:
                    self.position_travelling = self.position_travelling + one_step - G[self.stack_path[0][0]][second_node]['weight']
                    if size_path == 1:
                        self.state_move+=1
                    self.del_first_element_stack_path(size_path)
                    one_step = 0
    
            elif self.state_move == 3 and len(self.stack_path)>0:
                if  self.stack_position[0] > (self.position_travelling + one_step):
                    self.position_travelling += one_step
                    break
                else:
                    self.position_travelling = self.position_travelling + one_step - self.stack_position[0]
                    self.state_move+=1
                    del self.stack_position[0]
                    one_step = 0
    
            elif self.state_move == 4 and len(self.stack_path)>0:
                if  self.stack_position[0] > (self.position_travelling + one_step):
                    self.position_travelling += one_step
                    break
                else:
                    self.position_travelling = self.position_travelling + one_step - self.stack_position[0]
                    self.state_move=1
                    del self.stack_position[0]
                    one_step = 0
            max_while+=1
                        
        
    def len_and_path_position_to_node(self, position, node_1, node_2, client_node):
        dst_node_1_to_node_2 = nx.dijkstra_path_length(G,node_1,node_2)
        len_path_1 = nx.dijkstra_path_length(G,node_1,client_node) + position
        len_path_2 = nx.dijkstra_path_length(G,node_2,client_node) + dst_node_1_to_node_2 - position
        if len_path_1 < len_path_2:
            path = nx.dijkstra_path(G,node_1,client_node)
            return path, position,len_path_1
        else:
            path = nx.dijkstra_path(G,node_2,client_node)
            return path, dst_node_1_to_node_2 - position,len_path_2
        
            
    def path_node_to_dst(self, client_node, dst):
        initial_client_node, last_node = client_node, client_node 
        dst_construction = 0
        position = 0
        path = [client_node]
        lists_neighbors = list(nx.all_neighbors(G, client_node))
        ramdom_neighbor = rd.choice(lists_neighbors)
        len_start_end_before = 0
        len_start_end_after = nx.dijkstra_path_length(G,initial_client_node,ramdom_neighbor)
        
        while (G[client_node][ramdom_neighbor]['weight'] + dst_construction) <= dst:
            dst_construction += G[client_node][ramdom_neighbor]['weight']
            path.append(ramdom_neighbor)
            client_node = ramdom_neighbor
            len_start_end_before = nx.dijkstra_path_length(G,initial_client_node,client_node)
            lists_neighbors = list(nx.all_neighbors(G, client_node))
            max_while = 0
            while (ramdom_neighbor in path) or (len_start_end_before > len_start_end_after): 
                ramdom_neighbor = rd.choice(lists_neighbors)
                len_start_end_after = nx.dijkstra_path_length(G,initial_client_node,ramdom_neighbor)
                if(max_while==5):
                    ramdom_neighbor = rd.choice(lists_neighbors)
                    len_start_end_after = nx.dijkstra_path_length(G,initial_client_node,ramdom_neighbor)
                    break
                max_while+=1
                

        if (dst_construction <= dst):
            position = dst-dst_construction
            last_node = ramdom_neighbor
        return path, position, path[-1], last_node

    
    def bid(self, position_start, node_start_1, node_start_2, client_node, client_dst):
        a, b, dst_go_to_client = self.len_and_path_position_to_node(position_start, node_start_1, node_start_2, client_node)
        payoff_agent_before_auction = -self.gas_cost*dst_go_to_client + (self.charge_rate-self.gas_cost)*client_dst
        winning_payment_no_bid  = self.commission_company*(self.charge_rate-self.gas_cost)
        if payoff_agent_before_auction > winning_payment_no_bid:
            best_offer_possible = payoff_agent_before_auction - winning_payment_no_bid
        else:
            best_offer_possible = 0  
        bid = rd.uniform(0, best_offer_possible)  
        potential_wealth_agent = payoff_agent_before_auction - bid
        return bid, potential_wealth_agent
        
    def reinitiate(self):
        self.is_working = 0
        self.stack_position = [0]
        self.stack_node_start_1 = ['8_3']
        self.stack_node_start_2 = ['8_3']
        self.stack_client_node = ['8_3']
        self.stack_path = []
        


    def step(self):
        self.changement_day()
        self.nb_step += 1
        if self.nb_step <= 780 and self.nb_step >= 180 and (self.group == 1):
            self.is_working = 1          
        elif self.nb_step <= 1140 and self.nb_step >= 540 and (self.group == 2):
            self.is_working = 1            
        elif (self.nb_step <= 240 or self.nb_step >= 1080) and (self.group == 3):
            self.is_working = 1
        else:
            self.reinitiate()
                    
        self.move()
        self.advance()
        
    def advance(self):
        pass
        

class TaxiModel(Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.schedule = RandomActivation(self)
        
        call_center = CallCenterAgent(1,N,self)
        self.schedule.add(call_center)
        
        self.datacollector = DataCollector(
            agent_reporters={
            "Wealth": lambda call_center: call_center.wealth, 
            "Mu": lambda call_center: call_center.mu,
            "Number_Agent": lambda call_center: call_center.number_taxi_working,
            "Taxi": lambda call_center: call_center.list_taxis_wealth
            })
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

"""
Main
""" 

G = nx.read_edgelist("edges_graph.txt", delimiter=",", data=[("weight", int)]) 
G.edges(data=True)

"""  
Show the graph     
pos=nx.spring_layout(G)               
nx.draw(G, pos, with_labels=True)
plt.show()

list_nodes = G.nodes()
"""

#final model = TaxiModel(12)
model = TaxiModel(12)
#range(1440)
for i in range(2880):
    model.step()
    
call_center_wealth = model.datacollector.get_agent_vars_dataframe()
call_center_wealth['Wealth'].plot()
call_center_wealth[['Number_Agent', 'Mu']].plot()

call_center_wealth['Taxi'].plot()