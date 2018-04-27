import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

sizes = []

for i in range(1378, 15610):
	try:
		im = Image.open("/home/guy/rsna-bone-age/boneage-training-dataset/"+str(i)+".png")
		sizes.append(im.size)
	except:
		print(".")

sizes.sort()

classifiedsizes = [[(0,0),0]]

for i in sizes:
    new_elem = sizes.pop()
    if new_elem != classifiedsizes[0][0]:
        classifiedsizes.insert(0,[new_elem,0])
    else:
        classifiedsizes[0][1]+=1
#print(classifiedsizes)

x = np.zeros(1)
y = np.zeros(1)
s = np.zeros(1)
i = 0

for size in classifiedsizes:
    x = np.append(x,classifiedsizes[i][0][0])
    y = np.append(y,classifiedsizes[i][0][1])
    s = np.append(s,classifiedsizes[i][1])
    i+=1
    
plt.scatter(x,y,s=s/5)
plt.xlabel('width (pixel number)')
plt.ylabel('height (pixel number)')
plt.title('IMAGE SIZE DISTRIBUTION - rsna-bone-age')
plt.show()
