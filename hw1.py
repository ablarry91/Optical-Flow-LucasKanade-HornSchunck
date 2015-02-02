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
	plt.ion() #makes it so plots don't block code execution
	plt.imshow(img,cmap = 'gray')
	plt.scatter(POI[:,0,0],POI[:,0,1])
	for i in range(len(POI)):
		plt.arrow(POI[i,0,0],POI[i,0,1],V[i,0]*20,V[i,1]*20)
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

# def LK2():
KERNEL = 3 #must be odd/-
#get your first image
count = 0
fileName = 'office/office.' + str(count) + '.bmp'
img = cv2.imread(fileName,0)
img = cv2.GaussianBlur(img,(7,7),0)

#evaluate the first frame's POI
POI = cv2.goodFeaturesToTrack(img,20,.01,20)

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
	img = cv2.imread('office/office.' + str(count) + '.bmp',0)
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
	V = np.zeros([20,2])
	for i in range(len(POI)):	
		#build A
		A = buildA(gradXNew-gradXOld, gradYNew-gradYOld, POI[i][0][0], POI[i][0][1], KERNEL)

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

	#update lists
	gradXOld = gradXNew
	gradYOld = gradYNew
	gradOld = gradNew
	POI = cv2.goodFeaturesToTrack(img,20,.01,20)

	#visualize

	compareGraphs()
	# time.sleep(1)
	# time.sleep(2)
	# plt.close("all")
	print count
	# break


		#estimate the new point
	#estimate the new POI

	#determine edge within the POI



# img = cv2.imread('sphere/sphere.0.bmp',0)
# img = cv2.imread('rubic/rubic.0.bmp',0)
# img = cv2.imread('office/office.0.bmp',0)

# showImage(img,'original')
blur = cv2.GaussianBlur(img,(5,5),0)

# compareGraphs()
