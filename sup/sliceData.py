import scipy.io as si
import os
import math
from tqdm import tqdm


dataRoot = '../../../data'
mat = si.loadmat('/home/alvaro/Fred/data/train_data_lateral_EKF.mat')
lp = mat['lp'][0]   # lat pos
lsp = mat['lsp'][0] # lat spd
eH = mat['eH'][0]   # error heading
Q = mat['Q'][0]     # quality
T = mat['T'][0]
size = len(eH)

dataRate = 10
pastTime = 2
futurTime = 4

datasetPath = os.path.join(dataRoot,'EKF'+str(pastTime) )
xmlRaw = open(os.path.join(dataRoot,'EKF'+str(pastTime) + '.txt'), 'w')

minLp = 100
maxLp = -100
minLsp = 100
maxLsp = -100
minEH = 100
maxEH = -100

for i in range(len(lp)):
	if lp[i] < -1.8 or lp[i] > 9 or lsp[i] > 2 or eH[i] > 0.8 :
		Q[i] = 0
	if i > 0 and (T[i]-T[i-1] > 0.11 or T[i]-T[i-1] <0.09):
		Q[i] = 0
	if i <len(lp)-1 and (T[i+1]-T[i] > 0.11 or T[i+1]-T[i] <0.09):
		Q[i] = 0

num = 0
for i in tqdm(range(size - futurTime*dataRate - pastTime*dataRate - 1)):
	pastLp = lp[i:i+dataRate*pastTime+1]
	pastLsp = lsp[i:i+dataRate*pastTime+1]
	pastEH = eH[i:i+dataRate*pastTime+1]
	futureLp = [lp[i+dataRate*pastTime+1+10], lp[i+dataRate*pastTime+1+20], lp[i+dataRate*pastTime+1+30], lp[i+dataRate*pastTime+1+40]]
	futureLsp = [lsp[i+dataRate*pastTime+1+10], lsp[i+dataRate*pastTime+1+20], lsp[i+dataRate*pastTime+1+30], lsp[i+dataRate*pastTime+1+40]]
	futureEH = [eH[i+dataRate*pastTime+1+10], eH[i+dataRate*pastTime+1+20], eH[i+dataRate*pastTime+1+30], eH[i+dataRate*pastTime+1+40]]
	pastQ = Q[i:i+dataRate*pastTime+1]
	futureQ = [Q[i+dataRate*pastTime+1+10], Q[i+dataRate*pastTime+1+20], Q[i+dataRate*pastTime+1+30], Q[i+dataRate*pastTime+1+40]]

	# ! important
	# startLp = pastLp[0]
	# for i in range(len(futureLp)):
	# 	futureLp[i] -= startLp
	# for i in range(len(pastLp)):
	# 	pastLp[i] -= startLp

	if min(pastQ)==4 and min(futureQ)==4 and max(pastQ)==4 and max(futureQ)==4:
		minLp = min([minLp, min(pastLp), min(futureLp)])
		minLsp = min([minLsp, min(pastLsp), min(futureLsp)])
		minEH = min([minEH, min(pastEH), min(futureEH)])
		maxLp = max([maxLp, max(pastLp), max(futureLp)])
		maxLsp = max([maxLsp, max(pastLsp), max(futureLsp)])
		maxEH = max([maxEH, max(pastEH), max(futureEH)])

		if num>0:
			xmlRaw.write('\n%d %f %f %f %f %f %f %f %f %f %f %f %f' %(num, futureLp[0], futureLp[1], futureLp[2], futureLp[3], futureLsp[0], futureLsp[1], futureLsp[2], futureLsp[3], futureEH[0], futureEH[1], futureEH[2], futureEH[3] ))
		else:
			xmlRaw.write('%d %f %f %f %f %f %f %f %f %f %f %f %f' %(num, futureLp[0], futureLp[1], futureLp[2], futureLp[3], futureLsp[0], futureLsp[1], futureLsp[2], futureLsp[3], futureEH[0], futureEH[1], futureEH[2], futureEH[3] ))

		outDir = math.floor(num/100)
		dataPath = os.path.join(datasetPath,str(outDir),str(num))
		if not os.path.exists(dataPath):
			os.makedirs(dataPath)
		rawPath = os.path.join(dataPath,'data.raw')
		rawF = open(rawPath,'w')
		for itm in pastLp:
			rawF.write(str(itm)+' ')
		rawF.write('\n')
		for itm in pastLsp:
			rawF.write(str(itm)+' ')
		rawF.write('\n')
		for itm in pastEH:
			rawF.write(str(itm)+' ')
		rawF.close()
		num += 1
		
print('total num data: %d'%num)
print('lp range: (%f, %f)' %(minLp, maxLp))
print('lsp range: (%f, %f)' %(minLsp, maxLsp))
print('EH range: (%f, %f)' %(minEH, maxEH))
