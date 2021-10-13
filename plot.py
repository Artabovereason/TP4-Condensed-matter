import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import scipy.optimize
import scipy.integrate   as sintgr

sns.set(rc={'axes.facecolor':'whitesmoke'})

AREA_CALCULATION     = []
area_calculation     = []
area_calculation_min = []
area_calculation_max = []
angles               = []
step                 = 45 #step in measurements
number_of_data       = 5 #number of data to plot

'''s
with open("Modified_data/Dh214_45.dat", "r+") as text_file:
    texts = text_file.read()
    texts = texts.replace(",", ".")
with open("Modified_data/Dh214_45.dat", "w") as text_file:
    text_file.write(texts)

with open("Modified_data/Dh214_90.dat", "r+") as text_file:
    texts = text_file.read()
    texts = texts.replace(",", ".")
with open("Modified_data/Dh214_90.dat", "w") as text_file:
    text_file.write(texts)
'''

for kk in range(number_of_data):
    name        = "Dh84_"+str((kk)*step)
    file        = "Modified_data/"+name+".dat"
    theta       = kk*step
    list_1      = np.loadtxt(file)[:, 0]
    list_2      = np.loadtxt(file)[:, 1]
    list_3      = np.loadtxt(file)[:, 2]
    maxlist2    = max(list_2[:int(len(list_2)/2)])
    maxlist2bis = max(list_2[int(len(list_2)/2):])
    minlist2    = -min(list_2[:int(len(list_2)/2)])
    minlist2bis = -min(list_2[int(len(list_2)/2):])

    '''
    Renormalization in respect to Msaturation :
    '''
    for w in range(len(list_2)):
        if w < int(len(list_2)/4):
            list_2[w]*=1/maxlist2
        elif w < int(len(list_2)/2):
            list_2[w]*=1/minlist2
        elif w < int(3*len(list_2)/4):
            list_2[w]*=1/minlist2bis
        else:
            list_2[w]*=1/maxlist2bis

    angles.append(theta)
    AREA_CALCULATION.append(np.abs(scipy.integrate.simps(list_2,list_1)))

    plt.scatter(list_1,list_2,label='data',marker='x')
    plt.xlabel('Magnetic Field (in Oe)')
    plt.ylabel('Magnetization $M/M_{sat}$')
    plt.legend()
    plt.title('Magnetization in function of the external magnetic field $H$ for DH84, theta=%.1f'%theta)
    plt.savefig('output_fig/DH84/hysteresis'+name+'.png',dpi=300)
    plt.clf()

plt.plot(angles,AREA_CALCULATION,color='black')
plt.scatter(angles,AREA_CALCULATION,color='black',marker='x',label='data')
plt.xlabel('Angle theta in deg')
plt.ylabel('Area')
plt.legend()
plt.title('Value of the area of the hysteric cycle in function of the angle theta')
plt.savefig('output_fig/DH84/DH84-Value-Area')
