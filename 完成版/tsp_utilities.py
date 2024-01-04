#########ComputethelengthofaTSPtour
def tsp_length(d, tour):
    n = len(tour)
    length = d[tour[n - 1]][tour[0]] 
    for i in range(n - 1):
        length += d[tour[i]][tour[i + 1]]
    return length

######### Build solution representation by successors
def tsp_tour_to_succ(tour):
    n = len(tour)
    succ = [-1] * n
    for i in range(n):
        succ[tour[i]] = tour[(i+1)%n]
    return succ

######### Build solution representation by predeccessors
def tsp_succ_to_pred(succ):
    n = len(succ)
    pred = [-1] * n 
    for i in range(n):
         pred[succ[i]] = i
    return pred

#########Convertsolutionfromsuccessorofeachcitytocityorder
def tsp_succ_to_tour(succ):
    n = len(succ)
    tour = [-1] * n 
    j=0
    for i in range(n):
        tour[i] = j
        j = succ[j] 
    return tour

#########Convertasolutiongivenby2-optdatastructuretoastandardtour
def tsp_2opt_data_structure_to_tour(t):
    n = int(len(t)/2 + 0.5)
    tour = [-1] * n
    j=0
    for i in range(n):
        tour[i] = j>>1
        j = t[j]
    return tour

#########Compare2directedtours;returnsthenumberofdifferentarcs
def tsp_compare(succ_a,succ_b):
    n = len(succ_a)
    count = 0
    for i in range(n):
        if succ_a[i] != succ_b[i]:
            count += 1
    return count