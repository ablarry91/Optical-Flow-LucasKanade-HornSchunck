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

# HS(imgOld, imgNew, 1, 100)

def HS(im1, im2, alpha, ite,):

    #set up initial velocities
    uInitial = np.zeros([im1.shape])
    yInitial = np.zeros([im1.shape])

    # Set initial value for the flow vectors
    u = uInitial
    v = vInitial

    # Estimate spatiotemporal derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Averaging kernel
    kernel=np.matrix([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]])

    # Iterations
    for i in range(ite):
        # Compute local averages of the flow vectors
        uAvg=conv2(u,kernel,'same')
        vAvg=conv2(v,kernel,'same')
        # Compute flow vectors constrained by its local average and the optical flow constraints
        u = uAvg - fx.dot(((fx.dot(uAvg)) + (fy.dot(vAvg)) + ft))
        v = u
        # u= uAvg - ( fx .* ( ( fx .* uAvg ) + ( fy .* vAvg ) + ft ) ) ./ ( alpha^2 + fx.^2 + fy.^2) 
        # v= vAvg - ( fy .* ( ( fx .* uAvg ) + ( fy .* vAvg ) + ft ) ) ./ ( alpha^2 + fx.^2 + fy.^2)

    # u(isnan(u))=0
    # v(isnan(v))=0

    # %% Plotting
    # if displayFlow==1
    #     plotFlow(u, v, displayImg, 5, 5); 
    # end

def smoothImg(img,segma):
    # % Convolving an image with a Gaussian kernel.

    # % The degree of smoothing is determined by the Gaussian's standard
    # % deviation 'segma'. The Gaussian outputs a 'weighted average' of each
    # % pixel's neighborhood, with the average weighted more towards the value of
    # % the central pixels. The larger the standard deviation, the less weight
    # % towards the central pixels and more weight towards the pixels away, hence
    # % heavier blurring effect and less details in the output image.
    # %
    # % Author: Mohd Kharbat at Cranfield Defence and Security
    # % mkharbat(at)ieee(dot)org , http://mohd.kharbat.com
    # % Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    # % 3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    # %
    # % October 2008

    G=gaussFilter(segma);
    smoothedImg=conv2(img,G,'same');
    # smoothedImg=conv2(smoothedImg,G','same')

def gaussFilter(segma,kSize):
    # % Creates a 1-D Gaussian kernel of a standard deviation 'segma' and a size
    # % of 'kSize'. 
    # %
    # % In theory, the Gaussian distribution is non-zero everywhere. In practice,
    # % it's effectively zero at places further away from about three standard
    # % deviations. Hence the reason why the kernel is suggested to be truncated
    # % at that point.
    # %
    # % The 2D Gaussian filter is a complete circular symmetric operator. It can be
    # % seperated into x and y components. The 2D convolution can be performed by
    # % first convolving with 1D Gaussian in the x direction and the same in the
    # % y direction.
    # %
    # % Author: Mohd Kharbat at Cranfield Defence and Security
    # % mkharbat(at)ieee(dot)org , http://mohd.kharbat.com
    # % Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    # % 3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
    # %
    # % October 2008

    # if nargin<1
    #     segma=1;
    # end
    # if nargin<2
    #     kSize=2*(segma*3);
    # end

    import pylab
    x = pylab.frange(kSize/2,kSize/2,(1+1/kSize))
    # x=-(kSize/2):(1+1/kSize):(kSize/2);
    G=(1/(sqrt(2*pi)*segma)) * exp (-(x**2)/(2*segma**2));

def computeDerivatives(im1, im2):

    if im2.shape[0]==0:
        im2=zeros(size(im1))

    # Horn-Schunck original method
    kernelX = np.matrix([[-1,1],[-1,1]])*.25 #kernel for computing dx
    kernelY = np.matrix([[-1,-1],[1,1]])*.25 #kernel for computing dy
    kernelT = np.ones([2,2])*.25
    fx = cv2.filter2D(im1,-1,kernelX) + cv2.filter2D(im2,-1,kernelX)
    fy = cv2.filter2D(im1,-1,kernelY) + cv2.filter2D(im2,-1,kernelY)
    # ft = cv2.filter2D(im1,-1,kernelT) + cv2.filter2d(im2,-1,-kernelT)
    ft = im2 - im1
    return (fx,fy,ft)
