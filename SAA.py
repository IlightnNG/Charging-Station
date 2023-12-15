import math
from random import random
import matplotlib.pyplot as plt
import numpy as np




#   n1[i]        某i时段快充电桩数量
#   n2[i]        某i时段慢充电桩数量
#   Alpha        某时段快充电桩使用量占一天比例     n1[i]=a*Alpha[i]
#   Beta         某时段慢充电桩使用量占一天比例     n2[i]=b*Beta[i]
#   x[,i]        某i时段快慢充电桩数量比 x[0]=n1[0]/n2[0]

a1=0.8                  #   a1           单个快充电桩的输出损耗率 p出=a*p入
a2=1                    #   a2           单个慢充电桩的输出损耗率 p出=a*p入
p1=50                   #   p1           单个快充电桩的输入功率(kw)
p2=7                    #   p2           单个慢充电桩的输入功率(kw)
n_time=24               #                24个时段
#a=[200]*iter           #                n1[0]=Alpha[0]*a
#b=[1400]*iter
a0=100
b0=700
n1=[0]*n_time 
n2=[0]*n_time 
S_Alpha=[0]*n_time      #                标准Alpha
S_Beta =[0]*n_time 
Pin=[0]*n_time          #   Pin[i]       某i时段充电桩的输入功率
Pout=[0] *n_time        #   Pout[i]      某i时段充电桩的输出功率 （需求量）Pout = a1*p1*alpha*a+a2*p2*n2*beta*b
c1=10                   #   c1           单个快充电桩的建设及运营成本(krmb)
c2=3                    #   c2           单个快慢电桩的建设及运营成本(krmb)
ds_n1=[0]*n_time        #                快充电桩使用时间分布
ds_n2=[0]*n_time        #                慢充电桩使用时间分布






def func(Alpha,Beta,a,b,k):
    for i in range(n_time):                  #求Pin 0~23 的总和

        Pin[i]=p1*a*Alpha[k][i]/100+p2*b*Beta[k][i]/100

    Pin_Avg = np.average(Pin)                #求平均

    Pin_Var=np.var(Pin)                     #求方差
    Pin_Std=np.std(Pin)

    n1=np.max(Alpha[k])/100
    n2=np.max(Beta[k])/100
    W = c1*a*n1+c2*a*n2             #求总成本

    #print(str(Pin_Avg) + "|"+str(Pin_Var/100) +"|"+str(W*2) )
    #print(Pin_Avg + Pin_Var/100 +W*2)
    return Pin_Avg + Pin_Var/1000 +W*5 

def func_new(Alpha,Beta,a,b):
    for i in range(n_time):                  #求Pin 0~23 的总和
        Pin[i]=p1*a*Alpha[i]/100+p2*b*Beta[i]/100

    Pin_Avg = np.average(Pin)                #求平均

    Pin_Var=np.var(Pin)                     #求方差
    Pin_Std=np.std(Pin)

    n1=np.max(Alpha)/100
    n2=np.max(Beta)/100
    W = c1*a*n1+c2*a*n2              #求总成本

    #print(str(Pin_Avg) + "|"+str(Pin_Var/100) +"|"+str(W*2) )
    return Pin_Avg + Pin_Var/1000 +W*5




class SA:
    def __init__(self, func,func_new, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.func_new = func_new
        self.iter = iter         #内循环迭代次数,即为L =100
        self.alpha = alpha       #降温系数，alpha=0.99
        self.T0 = T0             #初始温度T0为100
        self.Tf = Tf             #温度终值Tf为0.01
        self.T = T0              #当前温度


        self.read()     #读取Pout S_Alpha S_Beta
        #初始化Alpha Beta
        self.Alpha = [[(self.S_Alpha[i]+0.5*(random()-random()))  for i in range(n_time)] for _ in range(iter)] #随机生成100个Alpha的值 Alpha=S_Alpha +-2
        self.Beta = [[(self.S_Beta[i]+ 0.5*(random()-random()))  for i in range(n_time)]for _ in range(iter)] #随机生成100个Beta的值
        #print(self.Alpha[0][0])
        self.a=[a0]*iter
        self.b=[b0]*iter
        #初始 a b 
        k=0
        flag=False
        while k < iter:
                self.a[k] = 100 + 20* (random() - random())
                self.b[k] = 300 + 60* (random() - random())
                #print(self.a[k])
                #print(self.b[k])
                for i in range(n_time):
                    if (a1*p1*self.S_Alpha[i]*self.a[k]/100 + a2*p2*self.S_Beta[i]*self.b[k]/100 >= 1.1*self.Pout[i]):#系数1.5
                        flag=True
                if(flag == True): 
                    flag = False
                    k +=1
        #self.a=[(a[i]+(random()-random())*a0/2) for i in range(iter)]    
        #self.b=[(b[i]+(random()-random())*b0/2) for i in range(iter)]

        self.most_best =[]
        self.history = {'f': [], 'T': [],'Alpha':[],'Beta':[],'a':[],'b':[]}

    def read(self):
        self.Pout=np.loadtxt("Pout.txt")
        self.S_Alpha=np.loadtxt("S_Alpha.txt")
        self.S_Beta=np.loadtxt("S_Beta.txt")
        #print(self.Pout)
        #print(self.S_Alpha)
        #print(self.S_Beta)

    def generate_new(self, Alpha,Beta,a,b,k):   #扰动产生新解的过程
        Alpha_new=[0]*n_time
        Beta_new=[0]*n_time
        flag = False

        while True:
            while True:
                a_new=a+self.T * (random() - random())/2
                b_new=b+self.T * (random() - random())*3/2
                for i in range(n_time):
                    if (a1*p1*self.S_Alpha[i]*a_new/100 + a2*p2*self.S_Beta[i]*b_new/100 >= 1.1*self.Pout[i]):#系数
                            flag=True
                if(flag == True): 
                    flag =False
                    break
            i = 0
            count = 0
            while (i < n_time):
                if(count>500):
                    break
                count+=1
                #print(count)
                Alpha_new[i] = Alpha[k][i] + 0.03*self.T * (random() - random())
                Beta_new[i] = Beta[k][i] + 0.03*self.T * (random() - random())
                if (self.S_Alpha[i]-2 <= Alpha_new[i] <= self.S_Alpha[i]+2)&(self.S_Beta[i]-2 <= Beta_new[i] <= self.S_Beta[i]+2)&(a1*p1*a_new*Alpha_new[i]/100 + a2*p2*b_new*Beta_new[i]/100 >= self.Pout[i]):
                    i+=1
                    #print(i)
            #print("||||")
            #print(np.sum(Alpha_new))
            #print(np.sum(Beta_new))        
            if(90<=np.sum(Alpha_new)<=110)&(90<=np.sum(Beta_new)<=110)&(count<500):
                break

            #重复得到新解，直到产生的新解满足约束条件
        return Alpha_new , Beta_new,a_new,b_new

    def Metrospolis(self, f, f_new):   #Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new)/10 / self.T) #接受几率范围
            if random() < p:
                return 1
            else:
                return 0

    def best(self):    #获取最优目标函数值
        f_list = []    #f_list数组保存每次迭代之后的值
        for k in range(self.iter):
            f = self.func(self.Alpha,self.Beta,self.a[k],self.b[k],k)        
            f_list.append(f)
        f_best = min(f_list)
        
        idx = f_list.index(f_best)
        return f_best, idx    #f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标
    
    def draw(self,Alpha,Beta,a,b,idx):

        x = np.arange(n_time)
        total_width, n = 0.8, 2
        # 每种类型的柱状图宽度
        width = total_width / n

        # 重新设置x轴的坐标
        x = x - (total_width - width) 
        # 画柱状图
        plt.bar(x, np.multiply(Alpha[idx],a), width=width, label="Fast")
        plt.bar(x +width, np.multiply(Beta[idx],b), width=width, label="Slow")
        # 显示图例
        plt.legend()
        # 显示柱状图
        plt.show()

    def run(self):
        count = 0
        #外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:       
           
            #内循环迭代100次
            for k in range(self.iter): 

                Alpha_new=[0]*n_time
                Beta_new=[0]*n_time

                #print("start new")
                Alpha_new, Beta_new,a_new,b_new = self.generate_new(self.Alpha, self.Beta,self.a[k],self.b[k],k) #产生新解
                #print("new")
                f=self.func(self.Alpha,self.Beta,self.a[k],self.b[k],k)         #将旧解和新解带入 当前k行的i 0~23
                f_new=self.func_new(Alpha_new,Beta_new,a_new,b_new)

                if self.Metrospolis(f, f_new):                         #判断是否接受新值
                    self.a[k]=a_new
                    self.b[k]=b_new
                    for i in range(n_time):
                        self.Alpha[k][i] = Alpha_new[i]             #如果接受新值，则把新值的x,y存入x数组和y数组
                        self.Beta[k][i] = Beta_new[i]


            # 迭代L次记录在该温度下最优解
            ft, idx = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            #self.history['a'].append(self.a[idx])
            #self.history['b'].append(self.b[idx])
            #温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha        
            count += 1
            print(self.T)
            
            # 得到最优解
        f_best, idx = self.best()
        print(f"F={f_best}, Alpha={self.Alpha[idx]}, Beta={self.Beta[idx]},a={self.a[idx]},b={self.b[idx]}")
        self.draw(self.Alpha,self.Beta,self.a[idx],self.b[idx],idx)
        print(self.a[idx])
        print(self.b[idx])

        n1_max =np.max(self.Alpha[idx])/100* self.a[idx]      #显示最大值
        n2_max =np.max(self.Beta[idx])/100*self.b[idx]
        print("max =")
        print(n1_max)
        print(n2_max)

sa = SA(func,func_new)
sa.run()

plt.plot(sa.history['T'], sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()



