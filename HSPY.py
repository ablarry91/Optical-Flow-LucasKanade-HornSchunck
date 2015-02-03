import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

FILTER = 7
count = 0

#upload images# 
# directory = 'box/box.'
# directory = 'office/office.'
# directory = 'rubic/rubic.'
directory = 'sphere/sphere.'
fileName = directory + str(count) + '.bmp'
imgOld = cv2.imread(fileName,0)
imgOld = cv2.GaussianBlur(imgOld,(FILTER,FILTER),1)

count += 1
imgNew = cv2.imread(fileName,0)
imgNew = cv2.GaussianBlur(imgNew,(FILTER,FILTER),1)

def HS(im1, im2, alpha, ite,):

	#set up initial velocities
	uInitial = np.zeros([im1.shape[0],im1.shape[1]])
	vInitial = np.zeros([im1.shape[0],im1.shape[1]])

	# Set initial value for the flow vectors
	u = uInitial
	v = vInitial

	# Estimate derivatives
	[fx, fy, ft] = computeDerivatives(im1, im2)

	# Averaging kernel
	kernel=np.matrix([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]])

	print fx[100,100],fy[100,100],ft[100,100]

	# Iteration to reduce error
	for i in range(ite):
		# Compute local averages of the flow vectors
		uAvg = cv2.filter2D(u,-1,kernel)
		vAvg = cv2.filter2D(v,-1,kernel)

		uNumer = (fx.dot(uAvg) + fy.dot(vAvg) + ft).dot(ft)
		uDenom = alpha + fx**2 + fy**2
		u = uAvg - np.divide(uNumer,uDenom)

		# print np.linalg.norm(u)

		vNumer = (fx.dot(uAvg) + fy.dot(vAvg) + ft).dot(ft)
		vDenom = alpha + fx**2 + fy**2
		v = vAvg - np.divide(vNumer,vDenom)
	return (u,v)

def computeDerivatives(im1, im2):
	# build kernels for calculating derivatives
	kernelX = np.matrix([[-1,1],[-1,1]])*.25 #kernel for computing dx
	kernelY = np.matrix([[-1,-1],[1,1]])*.25 #kernel for computing dy
	kernelT = np.ones([2,2])*.25

	#apply the filter to every pixel using OpenCV's convolution function
	fx = cv2.filter2D(im1,-1,kernelX) + cv2.filter2D(im2,-1,kernelX)
	fy = cv2.filter2D(im1,-1,kernelY) + cv2.filter2D(im2,-1,kernelY)
	# ft = im2 - im1
	ft = cv2.filter2D(im2,-1,kernelT) + cv2.filter2D(im1,-1,-kernelT)
	return (fx,fy,ft)

def smoothImage(img,kernel):
	G = gaussFilter(kernel)
	smoothedImage=cv2.filter2D(img,-1,G)
	smoothedImage=cv2.filter2D(smoothedImage,-1,G.T)
	return smoothedImage

def gaussFilter(segma):
	kSize = 2*(segma*3)
	x = range(-kSize/2,kSize/2,1+1/kSize)
	x = np.array(x)
	G = (1/(2*np.pi)**.5*segma) * np.exp(-x**2/(2*segma**2))
	return G

def compareGraphs():
	plt.ion() #makes it so plots don't block code execution
	plt.imshow(imgNew,cmap = 'gray')
	# plt.scatter(POI[:,0,1],POI[:,0,0])
	for i in range(len(u)):
		if i%5 ==0:
			for j in range(len(u)):
				if j%5 == 0:
					plt.arrow(j,i,v[i,j]*.00001,u[i,j]*.00001, color = 'red')
				pass
		# print i
	# plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
	plt.show()
[u,v] = HS(imgOld, imgNew, 1, 100)
compareGraphs()