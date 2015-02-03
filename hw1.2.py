




FILTER = 7

#upload images# 
# directory = 'box/box.'
# directory = 'office/office.'
# directory = 'rubic/rubic.'
directory = 'sphere/sphere.'
fileName = directory + str(count) + '.bmp'
imgOld = cv2.imread(fileName,0)
imgOld = cv2.GaussianBlur(imgOld,(FILTER,FILTER),1)

HS(imgOld, imgNew, 1, 100, 0, 0)

def HS(imgOld, imgNew, alpha, iter, uStart, vStart):

if nargin<5 || nargin<6
    uInitial = zeros(size(im1(:,:,1)));
    vInitial = zeros(size(im2(:,:,1)));
elseif size(uInitial,1) ==0 || size(vInitial,1)==0
    uInitial = zeros(size(im1(:,:,1)));
    vInitial = zeros(size(im2(:,:,1)));
end

# Set initial value for the flow vectors
u = uStart;
v = vStart;

# Estimate spatiotemporal derivatives
[fx, fy, ft] = computeDerivatives(im1, im2);

# Averaging kernel
kernel_1=[1/12 1/6 1/12;1/6 0 1/6;1/12 1/6 1/12];

# Iterations
for i=1:ite
    # Compute local averages of the flow vectors
    uAvg=conv2(u,kernel_1,'same');
    vAvg=conv2(v,kernel_1,'same');
    # Compute flow vectors constrained by its local average and the optical flow constraints
    u= uAvg - ( fx .* ( ( fx .* uAvg ) + ( fy .* vAvg ) + ft ) ) ./ ( alpha^2 + fx.^2 + fy.^2); 
    v= vAvg - ( fy .* ( ( fx .* uAvg ) + ( fy .* vAvg ) + ft ) ) ./ ( alpha^2 + fx.^2 + fy.^2);

u(isnan(u))=0;
v(isnan(v))=0;

# Plotting
if displayFlow==1
    plotFlow(u, v, displayImg, 5, 5); 

