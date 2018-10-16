cell_outage_code

detect outage cell by knn surpervised algorithm

signal power参数说明

macro cell半径设为5km 这部分我们还没改，但是应该没有改的必要 1km也可以
#FSPL自由空间损耗

FSPL = 20lg(d) + 20lg(f) + 32.45
d的单位为km，f的单位为MHz，FSPL的单位为db
#Receive_power

（这里认为Receive_power为RSRP，实际上应该不是）
Pr = Pt + Gt - L + Gr
式中Pt是发射功率，Gt是发射天线增益，L是自由空间损耗，Gr是接收天线增益。这里把天线增益都设为0
发射能量为46dbm 所以 Receive_Power = 46 - FSPL 单位为dbm
#SINR

db = 10lg(P1/P2)
SINR = receive_power/(interference+noise_power)
这里的单位都是mW，所以把得到以dbm为单位的Receive_power和noise_power换成以mW为单位
#maximum neighboring RSRP

就是计算UE相邻小区中最邻近的小区发射到UE，UE接收的能量
#maximum neighboring SINR

我理解的计算这个就是计算UE相邻小区中最邻近的小区发射到UE，UE接收的能量，并且除以干扰和噪声功率
#干扰（interference）:

·每个cell内的UE受到相邻6个cell的干扰
·按距离算出干扰信号的能量
#噪声功率（noise_power）:

 P_noise = 10*log(k*Tsyst)+10*log(B)+NF_px
        = -174+10*lg(2.6*10^9)+9
        = -37.65 dBm
#dbm转换成mW的公式

math.pow(10,dbm/10) = mW
1W = 1000mW
