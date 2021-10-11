import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import scipy.optimize
import scipy.integrate   as sintgr

sns.set(rc={'axes.facecolor':'whitesmoke'})

def arctan_fit(x,a,b,c,d):
    return np.arctan(b*x+d)*a+c

area_calculation = []
area_calculation_min = []
area_calculation_max = []
angles=[]
for kk in range(13):
    name = "Co,"+str((kk)*15)
    #if kk*45 == 315:
    #    name= "Fe,360"
    file = "Modified_data/"+name+".dat"
    list_1  = np.loadtxt(file)[:, 0]
    list_2 =  np.loadtxt(file)[:, 1]
    maxlist2 = max(list_2[:int(len(list_2)/2)])
    maxlist2bis = max(list_2[int(len(list_2)/2):])
    minlist2 = -min(list_2[:int(len(list_2)/2)])
    minlist2bis = -min(list_2[int(len(list_2)/2):])
    for w in range(len(list_2)):
        if w < int(len(list_2)/4):
            list_2[w]*=1/maxlist2
        elif w < int(len(list_2)/2):
            list_2[w]*=1/minlist2
        elif w < int(3*len(list_2)/4):
            list_2[w]*=1/minlist2bis
        else:
            list_2[w]*=1/maxlist2bis
    list_3 =  np.loadtxt(file)[:, 2]
    plt.scatter(list_1,list_2,label='hysteresis',marker='x')

    plt.xlabel('Magnetic Field (in Oe)')
    plt.ylabel('Magnetization $M/M_{sat}$')
    popt1, pcov1 = scipy.optimize.curve_fit(arctan_fit, list_1[:int(len(list_1)/2)], list_2[:int(len(list_2)/2)])
    t1 = np.linspace(list_1[0],list_1[int(len(list_1)/2)],1000)
    for i in range(len(pcov1)):
        pcov1[i][i]=np.sqrt(np.abs(pcov1[i][i]))
    #plt.plot(t1,(2/np.pi)*np.arctan(t1*(popt1[1]+pcov1[1][1])+(popt1[3]+pcov1[3][3])),color='green',linestyle='--')
    #lt.plot(t1,(2/np.pi)*np.arctan(t1*(popt1[1]-pcov1[1][1])+(popt1[3]-pcov1[3][3])),color='green',linestyle='--')
    #plt.fill_between(t1,(2/np.pi)*np.arctan(t1*(popt1[1]-pcov1[1][1])+(popt1[3]-pcov1[3][3])),(2/np.pi)*np.arctan(t1*(popt1[1]+pcov1[1][1])+(popt1[3]+pcov1[3][3])),color='black',alpha=0.2,label='error=%.1f'%)

    #plt.plot(t1,popt1[0]*np.arctan(t1*popt1[1]+popt1[3])+popt1[2],color='red',linestyle='--')
    plt.plot(t1,(2/np.pi)*np.arctan(t1*popt1[1]+popt1[3]),color='red',linestyle='--')

    popt2, pcov2 = scipy.optimize.curve_fit(arctan_fit, list_1[int(len(list_1)/2):], list_2[int(len(list_2)/2):])
    for i in range(len(pcov2)):
        pcov2[i][i]=np.sqrt(np.abs(pcov2[i][i]))
    t2 = np.linspace(list_1[int(len(list_1)/2)],list_1[len(list_1)-1],1000)
    #plt.plot(t2,(2/np.pi)*np.arctan(t2*(popt2[1]+pcov2[1][1])+(popt2[3]+pcov2[3][3])),color='green',linestyle='--')
    #plt.plot(t2,(2/np.pi)*np.arctan(t2*(popt2[1]-pcov2[1][1])+(popt2[3]-pcov2[3][3])),color='green',linestyle='--')
    #plt.fill_between(t2,(2/np.pi)*np.arctan(t2*(popt2[1]-pcov2[1][1])+(popt2[3]-pcov2[3][3])),(2/np.pi)*np.arctan(t2*(popt2[1]+pcov2[1][1])+(popt2[3]+pcov2[3][3])),color='black',alpha=0.2,label='error=%.1f'% )

    #plt.plot(t2,popt2[0]*np.arctan(t2*popt2[1]+popt2[3])+popt2[2],color='red',linestyle='--',label='fit')
    plt.plot(t2,(2/np.pi)*np.arctan(t2*popt2[1]+popt2[3]),color='red',linestyle='--',label='fit')

    t3= np.linspace(-500,500,1000)


    def arctan_fit1(x):
        #return np.arctan(popt1[1]*x+popt1[3])*popt1[0]+popt1[2]
        return np.arctan(popt1[1]*x+popt1[3])*(2/np.pi)

    def arctan_fit1min(x):
        return (2/np.pi)*np.arctan(x*(popt1[1]-pcov1[1][1])+(popt1[3]-pcov1[3][3]))

    def arctan_fit1max(x):
        return (2/np.pi)*np.arctan(x*(popt1[1]+pcov1[1][1])+(popt1[3]+pcov1[3][3]))

    def arctan_fit2(x):
        #return np.arctan(popt2[1]*x+popt2[3])*popt2[0]+popt2[2]
        return np.arctan(popt2[1]*x+popt2[3])*(2/np.pi)

    def arctan_fit2min(x):
        return (2/np.pi)*np.arctan(x*(popt2[1]-pcov2[1][1])+(popt2[3]-pcov2[3][3]))

    def arctan_fit2max(x):
        return (2/np.pi)*np.arctan(x*(popt2[1]+pcov2[1][1])+(popt2[3]+pcov2[3][3]))


    integral_value1= sintgr.quad(arctan_fit1, -500,+500, epsrel=1.0e-9)[0]
    integral_value2= sintgr.quad(arctan_fit2, -500,+500, epsrel=1.0e-9)[0]

    integral_value1min= sintgr.quad(arctan_fit1min, -500,+500, epsrel=1.0e-9)[0]
    integral_value2min= sintgr.quad(arctan_fit2min, -500,+500, epsrel=1.0e-9)[0]
    integral_value1max= sintgr.quad(arctan_fit1max, -500,+500, epsrel=1.0e-9)[0]
    integral_value2max= sintgr.quad(arctan_fit2max, -500,+500, epsrel=1.0e-9)[0]

    string_to_show = str('The total area is : ')+str(round(np.abs(integral_value2-integral_value1),1))
    #plt.text(-490,0,string_to_show)

    #plt.fill_between(t3,popt2[0]*np.arctan(t3*popt2[1]+popt2[3])+popt2[2],popt1[0]*np.arctan(t3*popt1[1]+popt1[3])+popt1[2],color='red',alpha=0.2,label='area = '+str(round(np.abs(integral_value2-integral_value1),1)))
    plt.fill_between(t3,(2/np.pi)*np.arctan(t3*popt2[1]+popt2[3]),(2/np.pi)*np.arctan(t3*popt1[1]+popt1[3]),color='red',alpha=0.2,label='area = '+str(round(np.abs(integral_value2-integral_value1),1)))
    plt.fill_between(t1,(2/np.pi)*np.arctan(t1*(popt1[1]-pcov1[1][1])+(popt1[3]-pcov1[3][3])),(2/np.pi)*np.arctan(t1*(popt1[1]+pcov1[1][1])+(popt1[3]+pcov1[3][3])),color='pink',alpha=0.2,label='error='+str(round(np.abs(integral_value2min-integral_value1min),1)))
    plt.fill_between(t2,(2/np.pi)*np.arctan(t2*(popt2[1]-pcov2[1][1])+(popt2[3]-pcov2[3][3])),(2/np.pi)*np.arctan(t2*(popt2[1]+pcov2[1][1])+(popt2[3]+pcov2[3][3])),color='orange',alpha=0.2,label='error='+str(round(np.abs(integral_value2max-integral_value1max),1)))

    theta=kk*15
    angles.append(theta)
    area_calculation.append(round(np.abs(integral_value2-integral_value1),1))
    area_calculation_min.append(round(np.abs(integral_value2min-integral_value1min),1))
    area_calculation_max.append(round(np.abs(integral_value2max-integral_value1max),1))
    plt.legend()
    plt.title('Magnetization in function of the external magnetic field $H$ for Co, theta=%.1f'%theta)
    plt.savefig('output_fig/hysteresis'+name+'.png',dpi=300)
    plt.clf()
#plt.show()
'''
for i in range(len(angles)):
    angles.append(angles[i]+180)
    area_calculation.append(area_calculation[i])
'''

#error_bar = np.array([[], []], np.int32)
error_bar=np.arange(2*len(area_calculation_max)).reshape(2,len(area_calculation_max))

for i in range(len(area_calculation_max)):
    error_bar[0][i]=np.abs(area_calculation[i]-area_calculation_max[i])
    error_bar[1][i]=np.abs(area_calculation_min[i]-area_calculation[i])

plt.scatter(angles,area_calculation,label='Data points',marker='x',color='red')
plt.scatter(angles,area_calculation_min,label='Data points max',marker='x',color='pink')
plt.scatter(angles,area_calculation_max,label='Data points min',marker='x',color='orange')
plt.errorbar(angles,area_calculation,yerr=error_bar,label='error bar')
plt.legend()
plt.xlabel('Angle theta in deg')
plt.ylabel('Area')
plt.title('Value of the area of the hysteric cycle in function of the angle theta')
plt.savefig('output_fig/Co-Value-Area')


#print(list_1[100])
