import cv2
import numpy as np
from matplotlib import pyplot as plt

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
	plt.subplot(2,2,1),plt.imshow(grad,cmap = 'gray')
	plt.title('Austin grad test'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,2),plt.imshow(img,cmap = 'gray')
	plt.scatter(test[:,0,0],test[:,0,1])
	plt.subplot(2,2,3),plt.imshow(sobelX,cmap = 'gray')
	plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,4),plt.imshow(sobelY,cmap = 'gray')
	plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
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

def lucasKanade(POI, oldFrame, newFrame):
	#evaluate the first frame's POI


	#for every





	W = gaussianWeight()
	for i in range(len(POI)):
		omegaNew = buildKernel(newFrame, POI[i][0][0], POI[i][0][1], KERNEL)
		omegaOld = buildKernel(oldFrame, POI[i][0][0], POI[i][0][1], KERNEL)
		for j in range(KERNEL):
			for k in range(KERNEL):
				pass
	pass

# def LK2():
KERNEL = 7 #must be odd/-
#get your first image
count = 0
fileName = 'sphere/sphere.' + str(count) + '.bmp'
img = cv2.imread(fileName,0)

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
	img = cv2.imread('sphere/sphere.' + str(count) + '.bmp',0)
	try:
		if img.any():
			print 'it exists'
	except:
		print 'it doesnt exist'
		print 'count is',count
		break

	#evaluate the new image's gradients
	gradXNew = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	gradYNew = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	gradNew = totalGrad(gradXNew,gradYNew)


	#evaluate every POI
	for i in range(len(POI)):	
		#build A
		A = buildA(gradXNew-gradXOld, gradYNew-gradYOld, POI[i][0][0], POI[i][0][1], KERNEL)

		#build b
		B = buildB(gradNew-gradOld, POI[i][0][0], POI[i][0][1], KERNEL)

		#solve for v		
		try:
			V = np.matrix((A.T).dot(W**2).dot(A)).I.dot(A.T).dot(W**2).dot(B)
		except:
			pass
	#estimate the new POI

	#determine edge within the POI



img = cv2.imread('sphere/sphere.0.bmp',0)
# img = cv2.imread('rubic/rubic.0.bmp',0)
# img = cv2.imread('office/office.0.bmp',0)

# showImage(img,'original')
blur = cv2.GaussianBlur(img,(5,5),0)
# showImage(img,'blurred')

# showImage(grad,'gradient')

# determine the gradient


# showImage(grad,"gradient")
# showImage(img,"original")

test = cv2.goodFeaturesToTrack(img,20,.01,20)

# lucasKanade()
# compareGraphs()
