import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

a = 0
b = 0
w = 0

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
	# plt.ion() #makes it so plots don't block code execution
	plt.imshow(img,cmap = 'gray')
	plt.scatter(POI[:,0,0],POI[:,0,1])
	for i in range(len(POI)):
		plt.arrow(POI[i,0,0],POI[i,0,1],V[i,0]*5,V[i,1]*5)
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

def buildA(gradientX, gradientY, centerX, centerY, kernelSize):
	gradX = buildKernel(gradientX, centerX, centerY, kernelSize)
	gradY = buildKernel(gradientY, centerX, centerY, kernelSize)
	AX = gradX.reshape((1,kernelSize**2))
	AY = gradY.reshape((1,kernelSize**2))
	A = np.column_stack([AX.transpose(), AY.transpose()])
	return A

def gaussianWeight(kernelSize):
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
	
def buildB(gradTotal, centerX, centerY, kernelSize):
	B = buildKernel(gradTotal, centerX, centerY, kernelSize)
	B = B.reshape((1,kernelSize**2))
	return -B.transpose()

def totalGrad(gradX,gradY):
	shape = gradX.shape
	grad = np.zeros([shape[0],shape[1]])
	for i in range(shape[0]): #gradient in the y direction
		for j in range(shape[1]): #gradient in the x direction
			try: #try to calculate and normalize
				grad[i,j] = getGradient(gradX[i,j],gradY[i,j])
			except: 
				pass
	return grad

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
			POI[count,0,0] = xPos
			POI[count,0,1] = yPos
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
# def LK2():
KERNEL = 7 #must be odd/-
#get your first image
count = 0
directory = 'box/box.'
# directory = 'office/office.'
fileName = directory + str(count) + '.bmp'
img = cv2.imread(fileName,0)
img = cv2.GaussianBlur(img,(7,7),0)

#evaluate the first frame's POI
# POI = cv2.goodFeaturesToTrack(img,20,.01,20)
POI = getPOI(200,200,KERNEL)

#evaluate the first frame's gradients
gradXOld = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
gradYOld = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
gradOld = totalGrad(gradXOld,gradYOld)

#get the weights 
W = gaussianWeight(KERNEL)

#loop until no pictures are available
while True:
	#load the next image
	count += 1
	img = cv2.imread(directory + str(count) + '.bmp',0)
	img = cv2.GaussianBlur(img,(7,7),0)	
	try:
		if img.any():
			# print 'it exists'
			pass
	except:
		# print 'it doesnt exist'
		print 'count is',count
		break

	#evaluate the new image's gradients
	gradXNew = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	gradYNew = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	gradNew = totalGrad(gradXNew,gradYNew)


	#evaluate every POI
	V = np.zeros([(POI.shape)[0],2])
	for i in range(len(POI)):	
		#build A
		A = buildA(gradXNew, gradYNew, POI[i][0][0], POI[i][0][1], KERNEL)

		#build b
		B = buildB(gradNew-gradOld, POI[i][0][0], POI[i][0][1], KERNEL)

		#solve for v		
		try:
			Vpt = np.matrix((A.T).dot(W**2).dot(A)).I.dot(A.T).dot(W**2).dot(B)
			# print Vpt
			V[i,0] = Vpt[0]
			V[i,1] = Vpt[1]
		except:
			pass

	compareGraphs()
	# time.sleep(1)
	# time.sleep(2)
	# plt.close("all")
	print count
	break

	#update lists
	gradXOld = gradXNew
	gradYOld = gradYNew
	gradOld = gradNew
	# POI = cv2.goodFeaturesToTrack(img,20,.01,20)
	POI = getPOI(200,200,KERNEL)

	#visualize


	# break


		#estimate the new point
	#estimate the new POI

	#determine edge within the POI



# img = cv2.imread('sphere/sphere.0.bmp',0)
# img = cv2.imread('rubic/rubic.0.bmp',0)
# img = cv2.imread('office/office.0.bmp',0)

# showImage(img,'original')
blur = cv2.GaussianBlur(img,(5,5),0)

def lucas_kanade_np(im1, im2, win=2):
	assert im1.shape == im2.shape
	I_x = np.zeros(im1.shape)
	I_y = np.zeros(im1.shape)
	I_t = np.zeros(im1.shape)
	I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
	I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
	I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
	params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
	params[..., 0] = I_x * I_x # I_x2
	params[..., 1] = I_y * I_y # I_y2
	params[..., 2] = I_x * I_y # I_xy
	params[..., 3] = I_x * I_t # I_xt
	params[..., 4] = I_y * I_t # I_yt
	del I_x, I_y, I_t
	cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
	del params
	win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
				  cum_params[2 * win + 1:, :-1 - 2 * win] -
				  cum_params[:-1 - 2 * win, 2 * win + 1:] +
				  cum_params[:-1 - 2 * win, :-1 - 2 * win])
	del cum_params
	op_flow = np.zeros(im1.shape + (2,))
	det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
	op_flow_x = np.where(det != 0,
						 (win_params[..., 1] * win_params[..., 3] -
						  win_params[..., 2] * win_params[..., 4]) / det,
						 0)
	op_flow_y = np.where(det != 0,
						 (win_params[..., 0] * win_params[..., 4] -
						  win_params[..., 2] * win_params[..., 3]) / det,
						 0)
	op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
	op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
	return op_flow


# compareGraphs()
