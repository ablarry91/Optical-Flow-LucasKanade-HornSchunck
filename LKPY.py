import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def compareGraphs(imgOld, imgNew, POI, V):
	plt.imshow(imgNew,cmap = 'gray')
	# plt.scatter(POI[:,0,1],POI[:,0,0])
	for i in range(len(POI)):
		plt.arrow(POI[i,0,1],POI[i,0,0],V[i,1]*1,V[i,0]*1, color = 'red')
	# plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
	plt.show()

def buildA(img, centerX, centerY, kernelSize):
	#build a kernel containing pixel intensities
	mean = kernelSize//2
	count = 0
	home = img[centerY, centerX] #storing the intensity of the center pixel
	A = np.zeros([kernelSize**2, 2])
	for j in range(-mean,mean+1): #advance the y
		for i in range(-mean,mean+1): #advance the x
			if i == 0:
				Ax = 0
			else:
				Ax = (home - img[centerY+j, centerX+i])/i
			if j == 0:
				Ay = 0
			else:
				Ay = (home - img[centerY+j, centerX+i])/j
			#write to A
			A[count] = np.array([Ay, Ax])
			count += 1
	# print np.linalg.norm(A)
	return A

def buildB(imgNew, imgOld, centerX, centerY, kernelSize):
	mean = kernelSize//2
	count = 0
	home = imgNew[centerY, centerX]

	B = np.zeros([kernelSize**2])
	for j in range(-mean,mean+1):
		for i in range(-mean,mean+1):
			Bt = imgNew[centerY+j,centerX+i] - imgOld[centerY+j,centerX+i]
			B[count] = Bt
			count += 1
		# print np.linalg.norm(B)
	return B

def gaussianWeight(kernelSize, even=False):
	if even == True:
		weight = np.ones([kernelSize,kernelSize])
		weight = weight.reshape((1,kernelSize**2))
		weight = np.array(weight)[0]
		weight = np.diag(weight)
		return weight
	SIGMA = 1 #the standard deviation of your normal curve
	CORRELATION = 0 #see wiki for multivariate normal distributions
	weight = np.zeros([kernelSize,kernelSize])
	cpt = kernelSize%2+kernelSize//2 #gets the center point
	for i in range(len(weight)):
		for j in range(len(weight)):
			ptx = i + 1
			pty = j + 1
			weight[i,j] = 1/(2*np.pi*SIGMA**2)/(1-CORRELATION**2)**.5*np.exp(-1/(2*(1-CORRELATION**2))*((ptx-cpt)**2+(pty-cpt)**2)/(SIGMA**2))
			# weight[i,j] = 1/SIGMA/(2*np.pi)**.5*np.exp(-(pt-cpt)**2/(2*SIGMA**2))
	weight = weight.reshape((1,kernelSize**2))
	weight = np.array(weight)[0] #convert to a 1D array
	weight = np.diag(weight) #convert to n**2xn**2 diagonal matrix
	return weight
	# return np.diag(weight)

def getPOI(xSize, ySize, kernelSize):
	mean = kernelSize//2
	xPos = mean
	yPos = mean
	xStep = (xSize-mean)//kernelSize
	yStep = (ySize-mean)//kernelSize
	length = xStep*yStep
	POI = np.zeros([length,1,2])
	count = 0
	for i in range(yStep):
		for j in range(xStep):
			POI[count,0,1] = xPos
			POI[count,0,0] = yPos
			xPos += kernelSize
			count += 1
		xPos = mean
		yPos += kernelSize
	return POI

def LK():
	KERNEL = 5 #must be odd/
	FILTER = 7

	#get your first image
	count = 0
	# directory = 'box/box.'
	# directory = 'office/office.'
	# directory = 'rubic/rubic.'
	directory = 'sphere/sphere.'
	fileName = directory + str(count) + '.bmp'
	imgOld = cv2.imread(fileName,0)
	imgOld = cv2.GaussianBlur(imgOld,(FILTER,FILTER),1)

	#evaluate the first frame's POI
	POI = getPOI(200,200,KERNEL)

	#get the weights 
	W = gaussianWeight(KERNEL)

	#loop until no pictures are available
	while True:
		#load the next image
		count += 1
		imgNew = cv2.imread(directory + str(count) + '.bmp',0)
		imgNew = cv2.GaussianBlur(imgNew,(FILTER,FILTER),1)	
		try:
			if imgNew.any():
				# print 'it exists'
				pass
		except:
			# print 'it doesnt exist'
			print 'count is',count
			break

		#evaluate every POI
		V = np.zeros([(POI.shape)[0],2])
		for i in range(len(POI)):	
			A = buildA(imgNew, POI[i][0][1], POI[i][0][0], KERNEL)
			B = buildB(imgNew, imgOld, POI[i][0][1], POI[i][0][0], KERNEL)

			#solve for v		
			try:
				Vpt = np.matrix((A.T).dot(W**2).dot(A)).I.dot(A.T).dot(W**2).dot(B)
				# print Vpt
				V[i,0] = Vpt[0,0]
				V[i,1] = Vpt[0,1]
			except:
				pass

		#all done, evaluate the image
		compareGraphs(imgOld,imgNew, POI, V)

		#if we don't want to loop several times
		if count == 1:
			break

		#update lists
		imgOld = imgNew
		POI = getPOI(200,200,KERNEL)

LK()