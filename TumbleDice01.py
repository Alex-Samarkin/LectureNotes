# generate tumbling dice value

import random
import seaborn as sns
import matplotlib.pyplot as plt

def roll_dice():
    # roll dice 
    dice_value = random.randint(1,6)
    return dice_value

# roll dice with different count of sides
def roll_diceN(N=6):
    # roll dice 
    dice_value = random.randint(1,N)
    return dice_value

# roll dice with different count of sides using random choise
def roll_dice_custom(sides):
    # roll dice 
    dice_value = random.choice(range(1, sides+1))
    return dice_value

# roll dice with different count of sides using random choise and custom weight
def roll_dice_custom_weight(sides, weights):
    # roll dice 
    dice_value = random.choices(range(1, sides+1), weights=weights, k=1)
    return dice_value

count = 1000

coins = []
for i in range(count):
    coins.append(roll_diceN(2))

#sns.histplot(coins,stat="count",binwidth=0.5)
#plt.show()

count = 10000

dice1 = []
dice2 = []
points = []

for i in range(count):
    dice1.append(roll_diceN(6))
    dice2.append(roll_diceN(6))
    points.append(dice1[i]+dice2[i])

#sns.histplot(points,discrete=True)
#plt.show()

count = 10000

points = []
# append first attempt if it >15 or
# append second attempt
for i in range(count):
    pt = roll_diceN(20)
    if pt > 15:
        points.append(pt)
    else:
        points.append(roll_diceN(20))
        
res = []
# check results
for pt in points:
    if pt>15:
        res.append(1)
    else:
        res.append(0)

sns.histplot(res,discrete=True)
plt.show()