import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def showImage(image,frameName):
	cv2.imshow(frameName,img)
	cv2.waitKey(0) #if a keystroke is detected in this time, do not continue
	cv2.destroyAllWindows() #destroys all windows that we created
	cv2.imshow(frameName,img)

def getGradient(gradX,gradY):
	return np.power(gradX**2 + gradY**2,.5)

def compareGraphs():
	# plt.subplot(2,2,1),plt.imshow(grad,cmap = 'gray')
	# plt.title('Austin grad test'), plt.xticks([]), plt.yticks([])
	# plt.subplot(2,2,2),plt.imshow(img,cmap = 'gray')
	# plt.scatter(test[:,0,0],test[:,0,1])
	# plt.subplot(2,2,3),plt.imshow(sobelX,cmap = 'gray')
	# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	# plt.subplot(2,2,4),plt.imshow(sobelY,cmap = 'gray')
	# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	plt.ion() #makes it so plots don't block code execution
	plt.imshow(imgNew,cmap = 'gray')
	# plt.scatter(POI[:,0,1],POI[:,0,0])
	for i in range(len(POI)):
		plt.arrow(POI[i,0,1],POI[i,0,0],V[i,1]*1,V[i,0]*1, color = 'red')
	# plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
	plt.show()

def buildKernel(gradientFrame, centerX, centerY, kernelSize):
	kernel = np.zeros([kernelSize,kernelSize])
	mean = kernelSize//2 #kernelSize MUST be odd
	for i in range(kernelSize):
		for j in range(kernelSize):
			try:
				kernel[j,i] = gradientFrame[j - mean + centerY, i - mean + centerX]
			except:
				pass #we're on the edge of the image and there's no data to grab
	return kernel


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

def buildFlowMap(vX, vY, ptX, ptY):
	flow = np.zeros([len(vX),len(vY)])
	count = 0
	for j in range(len(vY)):
		for i in range(len(vX)):
			flow[j,i] = (vX[count]**2+vY[count]**2)**.5
			count += 1
	return flow

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
		#build A
		A = buildA(imgNew, POI[i][0][1], POI[i][0][0], KERNEL)

		#build b
		B = buildB(imgNew, imgOld, POI[i][0][1], POI[i][0][0], KERNEL)

		#solve for v		
		try:
			Vpt = np.matrix((A.T).dot(W**2).dot(A)).I.dot(A.T).dot(W**2).dot(B)
			# print Vpt
			V[i,0] = Vpt[0,0]
			V[i,1] = Vpt[0,1]
		except:
			# Vpt = np.matrix([0,0])
			# print 'skip called'
			pass
		# if np.linalg.norm(Vpt) !=0:
			# print 'woop',Vpt
			# break

	compareGraphs()
	# time.sleep(1)
	# time.sleep(2)
	# plt.close("all")
	print count
	if count == 1:
		break

	#update lists
	imgOld = imgNew
	# POI = cv2.goodFeaturesToTrack(img,20,.01,20)
	POI = getPOI(200,200,KERNEL)

# img = cv2.imread('sphere/sphere.0.bmp',0)
# img = cv2.imread('rubic/rubic.0.bmp',0)
# img = cv2.imread('office/office.0.bmp',0)

# showImage(img,'original')
blur = cv2.GaussianBlur(imgNew,(5,5),0)