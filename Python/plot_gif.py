from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

points = genfromtxt("GB.csv", delimiter=",")

x = list(range(0))
y = list(range(0))
for i in range(0, len(points)):
    x.append(points[i, 0])
    y.append(points[i, 1])

y1 = list(range(0))
for i in range(0, len(points)):
    y1.append(points[i, 646])

for i in range(0, 647):
    x = list(range(0))
    y = list(range(0))
    for j in range(0, len(points)):
        x.append(points[j, 0])
        y.append(points[j, i])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    plt.ylim(-200, 600)

    ax.plot(x, y)
    ax.plot(x, y1)

    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='r', labelsize='medium', width=3)
    plt.pause(0.01)
    fig.savefig(f'./img/{i}.jpeg')

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ims = []

for i in range(0, 647):
    im = ax.imshow(plt.imread(f'./img/{i}.jpeg'), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=150)
ani.save('./img/Function.gif')
