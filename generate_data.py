import random

ndigit = 100

with open('data.txt', 'w') as file:
    for i in range(10000):
        a = random.randint(1, ndigit)
        b = random.randint(1, ndigit)
        file.write(a.__str__()+' + '+b.__str__()+' = '+ (a+b).__str__()+'\n')
