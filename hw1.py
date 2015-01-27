import cv2
import numpy as np
from matplotlib import pyplot as plt

KERNEL = 7 #must be odd

def showImage(image,name):
	cv2.imshow(name,img)
	cv2.waitKey(0) #if a keystroke is detected in this time, do not continue
	cv2.destroyAllWindows() #destroys all windows that we created
	cv2.imshow(name,img)

def getGradient(gradX,gradY):
	return np.power(gradX**2 + gradY**2,.5)
def compareGraphs():
	plt.subplot(2,2,1),plt.imshow(grad,cmap = 'gray')
	plt.title('Austin grad test'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,2),plt.imshow(img,cmap = 'gray')
	plt.scatter(test[:,0,0],test[:,0,1])
	plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
	plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
	plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	plt.show()

def lucasKanade(POI, oldFrame, newFrame):
	W = gaussianWeight()
	for i in range(len(POI)):
		omegaNew = buildKernel(newFrame, POI[i][0][0], POI[i][0][1], KERNEL)
		omegaOld = buildKernel(oldFrame, POI[i][0][0], POI[i][0][1], KERNEL)
		for j in range(KERNEL):
			for k in range(KERNEL):
				pass
	pass

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

def gaussianWeight():
	SIGMA = 1 #the standard deviation of your normal curve
	CORRELATION = 0 #see wiki for multivariate normal distributions
	weight = np.zeros([KERNEL,KERNEL])
	cpt = KERNEL%2+KERNEL//2 #gets the center point
	for i in range(len(weight)):
		for j in range(len(weight)):
			ptx = i + 1
			pty = j + 1
			weight[i,j] = 1/(2*np.pi*SIGMA**2)/(1-CORRELATION**2)**.5*np.exp(-1/(2*(1-CORRELATION**2))*((ptx-cpt)**2+(pty-cpt)**2)/(SIGMA**2))
			# weight[i,j] = 1/SIGMA/(2*np.pi)**.5*np.exp(-(pt-cpt)**2/(2*SIGMA**2))
	print weight
	return weight
	# return np.diag(weight)
	

img = cv2.imread('sphere/sphere.0.bmp',0)
# img = cv2.imread('rubic/rubic.0.bmp',0)
# img = cv2.imread('office/office.0.bmp',0)

# showImage(img,'original')
blur = cv2.GaussianBlur(img,(5,5),0)
# showImage(img,'blurred')

shape = img.shape
grad = np.zeros([shape[0],shape[1]])

# grad = np.gradient(img)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

# showImage(grad,'gradient')

# determine the gradient
for i in range(shape[0]): #gradient in the y direction
	for j in range(shape[1]): #gradient in the x direction
		try: #try to calculate and normalize
			grad[i,j] = getGradient(sobelx[i,j],sobely[i,j])
		except: 
			pass

# showImage(grad,"gradient")
# showImage(img,"original")

test = cv2.goodFeaturesToTrack(img,20,.01,20)
compareGraphs()

gaussianWeight()

