'''
a = range(10)
print(a)
b = a[0:9]
print(b)
c = a[9]
print(c)
'''
import numpy as np

d = [2, 1, 3]
e = [6, 5, 4]
f = [7, 8, 9]

g = np.array([d, e, f])
print(g)
print('-------------------')
print(g[0, :])
order = g[0, :].argsort()
print(order)
print(g[0, :])
print('------------')
print(g[0][:][order])
print(g[1][:][order])
print(g[2][:][order])
print(order[1:])

list1 = [1, 2, 3, 4, 5, 6]
list2 = [3, 5, 7, 9]

commonalities = set(list1) - (set(list1) - set(list2))
num = len(list(commonalities))
print(num)

print 'num =', num

aaa = np.intersect1d(list1,list2)
print(aaa)
print(len(aaa))

bbb = range(10)
print(bbb)

ccc = bbb[-4:]
print(ccc)

d = list()
d.append(1)
d.append(2)
print(d)
print(d[0])

e = [1, 2, 3]
print(sum(e))

'''
import numpy as np
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

import matplotlib.pyplot as plt
#plt.hist(x, bins=50)
x = [1, 2, 3]
y = [3, 4, 5]
plt.bar(x, y)
plt.show()
'''

import matplotlib.pyplot as plt;

#plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Test 6']
y_pos = np.arange(len(objects))
performance = [10, 8, 6, 4, 2, 1]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Hits@10')

plt.show()

gg = list()
f = [1, 2, 3]
h = 3
gg.append(f)
gg.append(h)
print(h)

h = [1, 3, 4]
h.append(2.2)
print(h)

q = "foo"
w = "bar"
e = q + w
print(e)