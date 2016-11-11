# Loop that goes throught the real data pairs
# INSTRUCTIONS: Put all Tuebingen data files into the same directory
import numpy as np
from causalLabels import XCausesY
import entropicCausalPair
total_counter = 0
decision_counter = 0

#setoffiles = range(1,52)+range(56,71)+range(72,93)+range(94,105)
success_counter = 0
for i in range(len(XCausesY)):
	if XCausesY[i]!=0:
		file_name = 'pair'+str(i).rjust(4, '0')+'.txt'
		print "Filename: %s"%(file_name)
		r = entropicCausalPair.main(file_name)
		decision_counter+= abs(r)
		print "Decision: %f,Actual: %f \n"%(r,XCausesY[i])
		if r == XCausesY[i]:
			success_counter+=1
		total_counter+=1

print 'Decision Rate = %f'%(float(decision_counter)/total_counter)
print 'Success Rate = %f'%(float(success_counter)/decision_counter)