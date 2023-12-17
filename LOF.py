k = 4
k_distance = []
k_Neigbour = []
# Step1 --> calculate K-Distance and K-Neighbour
for i in range(stan_data.shape[0]):
    lst = list_[i]
    sorted_lst = np.sort(lst)
    count = 0
    temp_lst = []
    for j in range(len(sorted_lst)):
        if sorted_lst[j] <= k:
            count += 1
            temp_lst.append(j)
        else:
            break
    k_distance.append(sorted_lst[count])
    k_Neigbour.append(temp_lst)
    
# Step2 --> Reachibility Distance 
reachibility_distance = np.zeros((stan_data.shape[0], stan_data.shape[0]))
for sample in range(len(stan_data)):
    for nbr in range(len(stan_data)):
        if sample != nbr:
            reachibility_distance[sample][nbr] = max(k_distance[nbr], list_[sample][nbr])
        else:
            reachibility_distance[sample][nbr] = 0
            
# Step3 --> Local Reachibility Distance
local_reachibility_distance = []
for i in range(stan_data.shape[0]):
    count = 0
    for j in k_Neigbour[i]:
        count += (reachibility_distance[i][j]/len(k_Neigbour[i]))
        
    local_reachibility_distance.append(count)
    
# Step4 --> Local Outlier Factor
lofs = []
for i in range(stan_data.shape[0]):
    count = 0
    for j in k_Neigbour[i]:
        count += local_reachibility_distance[j]
        
    lofs.append(count/(local_reachibility_distance[i] * len(k_Neigbour[i])))
    