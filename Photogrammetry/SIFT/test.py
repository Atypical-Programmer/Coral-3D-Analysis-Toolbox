import cv2
import random
import numpy as np
import os
import matchers

def SiftDetector(image):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures = 20000)
    kpt, desc = sift.detectAndCompute(image, None)
    kpts = np.zeros((len(kpt),2),dtype='float')
    for i in range(len(kpt)):
        kpts[i][0] = kpt[i].pt[0]
        kpts[i][1] = kpt[i].pt[1]
    desc /= desc.sum(axis=1, keepdims=True)
    desc = np.sqrt(desc)
    return kpts, desc

# import images
img_path1 = "./data/PL510938_acr.JPG"
img_path2 = "./data/PL510939_acr.JPG"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

# feature extraction
kp1, desc1 = SiftDetector(img1)
kp2, desc2 = SiftDetector(img2)

# feature visualization
imgShow1 = cv2.imread(img_path1)
for i in range(kp1.shape[0]):
    cv2.circle(imgShow1,
               (int(kp1[i][0]), int(kp1[i][1])),
               int(img1.shape[0]/300), (0, 255, 0), 2)
cv2.imwrite(img_path1+"-SIFT.jpg", imgShow1)
imgShow2 = cv2.imread(img_path2)
for i in range(kp2.shape[0]):
    cv2.circle(imgShow2,
               (int(kp2[i][0]), int(kp2[i][1])),
               int(img1.shape[0]/300), (0, 255, 0), 2)
cv2.imwrite(img_path2+"-SIFT.jpg", imgShow2)

# feature matching
matchID = 1
if matchID == 1:
    matches = matchers.mutual_nn_ratio_matcher(desc1, desc2, 0.8).astype(np.uint32)
if matchID == 2:
    from adalam import AdalamFilter
    AdalamMatcher = AdalamFilter()
    AdalamMatcher.config['scale_rate_threshold'] = None
    AdalamMatcher.config['orientation_difference_threshold'] = None
    #AdalamMatcher.config['device'] = torch.device('cpu') 
    matches = AdalamMatcher.match_and_filter(kp1, kp2, desc1, desc2).data.cpu().numpy().astype(np.uint32)
    ### - start
    matches2 = AdalamMatcher.match_and_filter(kp2, kp1, desc2, desc1).data.cpu().numpy().astype(np.uint32)
    matchL = np.ones((len(kp1[:,0]), 1),dtype='int64') * (-1)
    matchR = np.ones((len(kp2[:,0]), 1),dtype='int64') * (-1)
    IDL = np.ones((len(kp1[:,0]), 1),dtype='int64') * (-1)
    for i1, i2 in matches:
        matchL[i1] = i2
    for i1, i2 in matches2:
        matchR[i1] = i2
    for i in range(matchL.shape[0]):
        IDL[i] = i
        if not matchR[matchL[i]] == i:
            matchL[i] = -1
    matches = np.delete(np.hstack((IDL, matchL)), np.where(matchL<0)[0], axis=0)
    
# match visualization
margin = 10
H1, W1 = img1.shape[0:2]
H2, W2 = img2.shape[0:2]
imageMatchShow = 255*np.ones((max(H1, H2), W1 + W2 + margin, 3), np.uint8)
imageMatchShow[:H1, :W1, :] = img1
imageMatchShow[:H2:, W1+margin:, :] = img2
print("matches", matches.shape[0])

for i1, i2 in matches:
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    cv2.line(imageMatchShow,
             (int(kp1[i1][0]), int(kp1[i1][1])),
             (int(kp2[i2][0]) + margin + W1, int(kp2[i2][1])),
             (r, g, b),
             thickness=int(H1/600),
             lineType=cv2.LINE_AA)
    cv2.circle(imageMatchShow,
               (int(kp1[i1][0]), int(kp1[i1][1])),
               int(H1/300),
               (r, g, b),
               2)
    cv2.circle(imageMatchShow,
               (int(kp2[i2][0]) + margin + W1, int(kp2[i2][1])),
               int(H1/300),
               (r, g, b),
               2)
cv2.imwrite(os.path.join(
    os.path.dirname(img_path1),
    os.path.basename(img_path1) + "-" +
    os.path.basename(img_path2) + "-SIFT.png"
), imageMatchShow)


