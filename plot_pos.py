import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import imageio.v2 as imageio 

df_x = pd.read_excel('./x_pos.xlsx',header=None, index_col=None) 
df_y = pd.read_excel('./y_pos.xlsx',header=None,index_col=None) 

print(df_x.shape[1])
x_coords=[]
ax=plt.axes()
ax.set_xlim([-0.5,2])
ax.set_ylim([-0.5,1])
for i in range(1,df_x.shape[1]): #iterating over each coln starting from 1 (first col is empty)
	temp=[]
	temp=df_x.iloc[:, i].tolist()
	x_coords.append(temp)
	
y_coords=[]
for j in range(1,df_y.shape[1]):
	temp=[]
	temp=df_y.iloc[:, j].tolist()
	y_coords.append(temp)
	
r=len(x_coords)
# for i in range(r):
# 	ax=plt.axes()
# 	ax.set_xlim([-1,2])
# 	ax.set_ylim([-1,1])

# 	ax.scatter(x_coords[i],y_coords[i])
# 	print(i)
# 	plt.tight_layout()
# 	plt.draw()
# 	plt.pause(0.9)
# 	plt.clf()
"""
def animate_func(i):
	
	fig.clear()
	ax.set_xlim([-1,2])
	ax.set_ylim([-1,1])


	ax.scatter(x_coords[i],y_coords[i])
	plt.draw()


	
# Plotting the Animation

fig = plt.figure()
ax = plt.axes()
line_ani = animation.FuncAnimation(fig, animate_func, interval=100,frames=10)
plt.show()
"""	
	
num_fluid_particles=625
x_coords=np.array(x_coords)
y_coords=np.array(y_coords)

for i in range(r):
	fig = plt.figure()
	#fig.set_size_inches(18.5, 10.5)
	ax=plt.axes()
	ax.set_xlim([-0.1,1.8])
	ax.set_ylim([-0.1,0.85])

	ax.scatter(x_coords[i,:num_fluid_particles],y_coords[i,:num_fluid_particles],s=0.8)
	ax.scatter(x_coords[i,num_fluid_particles:],y_coords[i,num_fluid_particles:],s=0.8)

	plt.savefig(f'frames/frame-{i}.png',dpi=150)
	plt.close()

with imageio.get_writer('frames.gif', mode='i') as writer:
    for i in range(r):
        image = imageio.imread(f'frames/frame-{i}.png')
        writer.append_data(image)
 





