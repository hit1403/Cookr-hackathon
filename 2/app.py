
"""
    2. Statement - Last Mile Delivery Batching. 60 marks

    Description: It is crucial in today’s last-mile delivery ecosystem to optimize for speed, and cost efficiencies. Smarter algorithms play a crucial
    role in the ecommerce marketplace deliveries We need to group/batch the delivery of multiple items to the same rider without losing time. Here are
    several scenarios where we need smart operational research algorithms.

        Rule # 1:
            • Two orders - From the same kitchen
            • Assign the pick-up to the same rider.
            • To the same customer.
            • Ready at the same time (10 mins apart).
        Rule # 2:
            • Two orders.
            • From two different kitchens (1 km apart).
            • To the same customer.
            • Ready at the same time (10 mins apart).
            • Assign the pick-up to the same rider.
        Rule # 3:
            • Two orders.
            • From the same kitchen.
            • To two different customers (1 km apart).
            • Ready at the same time (10 mins apart).
            • Assign the pick-up to the same rider.

        Rule # 6:
            • Two orders.
            • To the same customer.
            • 2nd kitchens pick up on the way to the customer.
            • Ready at the time the rider reaches the second kitchen (10 mins apart).
            • Assign the pick-up to the same rider.
        Rule # 7:
            • Two orders.
            • 2nd customers drop on the way to the 1st customer (Vice Versa).
            • 2nd kitchens pick up on the way to the customer.
            • Ready at the same time (10 mins apart or by the time rider reaches the kitchen).
            • Assign the pick-up to the same rider.
        Rule # 8:
            • Two orders.
            • From the same kitchen.
            • 2nd customers drop on the way to the customer 1st (Vice Versa).
            • Ready at the same time (10 mins apart).
            • Assign the pick-up to the same rider.
"""



from math import sin, cos, acos
from numpy import deg2rad
from collections import defaultdict

# Function to find dist between two locations.
def distance(lat1, lon1, lat2, lon2):
    lat1 = deg2rad(lat1)
    lon1 = deg2rad(lon1)
    lat2 = deg2rad(lat2)
    lon2 = deg2rad(lon2)

    return acos( sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2 - lon1) ) * 6371


# Function to assign riders to the customers :
def assign(L, K, R) :

    K1 = K.copy()
    R1 = R.copy()
    L1 = L.copy()

    custAssign = defaultdict(list)

    dists = []
    flag = 0

    #print('lenk', len(K1))
    for i in range(len(K1)) :
        if i>0:
            currCord = K1[i]

            for k in range(0, i - 1) :

                if currCord == K1[k] :
                    flag = 1
                    

                    # Rectangle window :
                    # curr Cust out :

                    if (currCord[0][0] <= L1[k][0] and currCord[0][1] <= L1[k][1]) and (L1[k][0] <= L1[i][0] and L1[k][1] <= L1[i][1]) :
                        custAssign['c' + str(i + 1)] = custAssign['c' + str(k + 1)]
                    
                    elif (currCord[0][0] <= L1[i][0] and currCord[0][1] <= L1[i][1]) and (L1[i][0] <= L1[k][0] and L1[i][1] <= L1[k][1]) :
                        custAssign['c' + str(i + 1)] = custAssign['c' + str(k + 1)]
                        
                    elif (currCord[0][0] >= L1[k][0] and currCord[0][1] >= L1[k][1]) and (L1[k][0] >= L1[i][0] and L1[k][1] >= L1[i][1]) :
                        custAssign['c' + str(i + 1)] = custAssign['c' + str(k + 1)]

                    elif (currCord[0][0] >= L1[i][0] and currCord[0][1] >= L1[i][1]) and (L1[i][0] >= L1[k][0] and L1[i][1] >= L1[k][1]) :
                        custAssign['c' + str(i + 1)] = custAssign['c' + str(k + 1)]

                    break
                        

        # Only one order :
        if len(K1[i]) == 1 and flag == 0:

            # if a previous customer's assigned rider passes 
            dist = [[distance(v[0] ,v[1], K1[i][0][0], K1[i][0][1]), k] for k, v in R1.items()]

            dist.sort()

            rider = dist[0][1]

            custAssign['c' + str(i + 1)] = [rider]

            #print('r', rider)
            #print('cA', custAssign)

        # Same customer two orders
        elif len(K1[i]) == 2 and flag == 0:
            
            # Distance between kitchens
            kitDist = distance(K1[i][0][0], K1[i][0][1], K1[i][1][0], K1[i][1][1])

            # Distance between two kitchens :
            if kitDist <= 1 :

                dist1 = [[distance(K1[i][0][0], K1[i][0][1], v[0], v[1]), k] for k, v in R1.items()]
                dist2 = [[distance(K1[i][1][0], K1[i][1][1], v[0], v[1]), k] for k, v in R1.items()]

                dist1.sort()
                dist2.sort()

                # DIstance of first kitchen from the closes rider is smaller than the closest rider from the 2nd kitchen.
                if dist1[0][0] <= dist2[0][0] :
                    custAssign['c' + str(i + 1)] = [dist1[-1]]

                else :
                    custAssign['c' + str(i + 1)] = [dist2[-1]]
            
            # Distance between two kitchens is > 1km :
            else :
                dist1 = [[distance(K1[i][0][0], K1[i][0][1], v[0], v[1]), k] for k, v in R1.items()]
                dist2 = [[distance(K1[i][1][0], K1[i][1][1], v[0], v[1]), k] for k, v in R1.items()]

                dist1.sort()
                dist2.sort()

                custAssign['c' + str(i + 1)] += [dist1[0][-1], dist2[0][-1]]


            '''tmp = []

            for j in range(len(K1[i])) :

                tmp += [[distance(v[0] ,v[1], K1[i][j][0], K1[i][j][1]), k] for k, v in R1.items()]
                print(f'cust{i + 1}', tmp)

            dists += tmp
            print('dists')
            print(dists)'''

  
    return custAssign


# Dictionary to store the coordinates of each rider.
rider = {
         "r1" : [11.022592, 77.003575],  # PSG Tech.
         "r2" : [11.030088, 77.027600],  # CIT.
         "r3" : [11.020384, 76.970828],  # Gandhipuram.
         "r4" : [11.008058, 76.958934],  # Brooks.
         "r5" : [11.058638, 77.087703],  # Neelambur Royal Care Hospital.
        }


# List to store the coordinates of each customer.
customer_loc = [
                [11.031047, 77.037839],  # Airport.
                [11.054687, 76.995046],  # Prozone Mall.
                [11.025860, 76.951378],  # Ganga Hospital.
                [10.988170, 76.962050],  # Ukkadam Bus Stand.
                [11.002274, 77.029230]   # Singanalur.
                ]

# List to store the coordinates of each hotel/kitchen.
kitchen = [
            [10.997107, 76.995870],  # Ramanathapuram.
            [10.996851, 76.967892],  # Railway station.
            [11.025913, 76.941858],  # Annapoorna Saibaba Colony.
            [11.045755, 77.041797],  # Broadway mall.
            [11.078568, 77.035901],  # Kallapati.
            [11.079727, 76.942602],  # Thudiyalur.
            [11.075668, 76.985087]   # KCT
        ]

#print("distance : ",distance(deg2rad(ridr[0][0]),deg2rad(ridr[0][1]),deg2rad(ridr[1][0]),deg2rad(ridr[1][1])))

# Inputting number of customers.
customerCount = int(input("\nEnter number of customers: "))

# Dict to store customer locations.
custLoc = {}

# List to store the order coordinaes of each customer.
custKitchen = []

for i in range(customerCount) :

    custLoc['c' + str(i + 1)] = list(map(float, input('\nEnter your location coordinates (input with ",") : ').split(',')))

    # Inputting number of orders for each customer.
    orderCount = int(input(f'\nEnter number of orders for customer {i+1} : '))

    # Orders of each customer.
    orders = []

    # Inputting choices for each order.
    print("\n\t1. kitchen1\n\t2. kitchen2\n\t3. kitchen3\n\t4. kitchen4\n\t5. kitchen5\n\t6. kitchen6\n\t7. kitchen7\n")
    for j in range(orderCount) :
        choice = int(input(f"\tEnter your choice of kitchen for order {j+1} : "))
        orders += [kitchen[choice - 1]]

    # Appending order of each customer to the main list.
    custKitchen += [orders]

#print('custK', custKitchen)


final_assign = assign(custLoc, custKitchen, rider)

print("final assignment: ",final_assign)

for cust,rid in final_assign.items():
    print(f"For Customer {cust} \n\t The riders assigned are : ")
    for j in rid:
        print("\t",j,"\n")
