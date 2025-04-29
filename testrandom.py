import random

while (True):
    goal = random.sample(range(1, 5*5), 1)

    goal[0] // 5, goal[0] % 5
        
    print(goal[0] // 5, goal[0] % 5)