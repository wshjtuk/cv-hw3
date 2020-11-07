import cv2
import numpy as np
import random
RANSAC_TIME = 1500
MIN_DIS = 5
compression = 3
index = 0
def stitch(img1, img2):
    global index
    index += 1
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    height2 = img2.shape[0]
    width2 = img2.shape[1]
    height = max(height1, height2)
    width = max(width1, width2)
    img1 = cv2.copyMakeBorder(img1, (height - height1)//2, height - height1 - (height - height1)//2, (width - width1)//2, width - width1 - (width - width1)//2, cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(img2, (height - height2)//2, height - height2 - (height - height2)//2, (width - width2)//2, width - width2 - (width - width2)//2, cv2.BORDER_CONSTANT)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    match = flann.knnMatch(des1, des2, k=2)
    goodMatch = []
    for i,(m,n) in enumerate(match):
            if m.distance < 0.75 * n.distance:
                    goodMatch.append(m)

    max_num = 0
    list_max = []
    for i in range(RANSAC_TIME):
        list = []
        ranIdx = []
        for j in range(4):
            ranIdx.append(random.randint(0, len(goodMatch) - 1))
        x1, y1 = kp1[goodMatch[ranIdx[0]].queryIdx].pt
        x1p, y1p = kp2[goodMatch[ranIdx[0]].trainIdx].pt
        x2, y2 = kp1[goodMatch[ranIdx[1]].queryIdx].pt
        x2p, y2p = kp2[goodMatch[ranIdx[1]].trainIdx].pt
        x3, y3 = kp1[goodMatch[ranIdx[2]].queryIdx].pt
        x3p, y3p = kp2[goodMatch[ranIdx[2]].trainIdx].pt
        x4, y4 = kp1[goodMatch[ranIdx[3]].queryIdx].pt
        x4p, y4p = kp2[goodMatch[ranIdx[3]].trainIdx].pt
        A = np.matrix([[x1, y1, 1, 0, 0, 0, -x1 * x1p, -x1p * y1],
                   [0, 0, 0, x1, y1, 1, -x1 * y1p, -y1 * y1p],
                   [x2, y2, 1, 0, 0, 0, -x2 * x2p, -x2p * y2],
                   [0, 0, 0, x2, y2, 1, -x2 * y2p, -y2 * y2p],
                   [x3, y3, 1, 0, 0, 0, -x3 * x3p, -x3p * y3],
                   [0, 0, 0, x3, y3, 1, -x3 * y3p, -y3 * y3p],
                   [x4, y4, 1, 0, 0, 0, -x4 * x4p, -x4p * y4],
                   [0, 0, 0, x4, y4, 1, -x4 * y4p, -y4 * y4p]])
        b = np.matrix([[x1p],
                       [y1p],
                       [x2p],
                       [y2p],
                       [x3p],
                       [y3p],
                       [x4p],
                       [y4p]])
        num = 0
        try:
            x = np.linalg.inv(A) * b
            H = np.matrix([[x[0, 0], x[1, 0], x[2, 0]],
                           [x[3, 0], x[4, 0], x[5, 0]],
                           [x[6, 0], x[7, 0], 1]])
            num = 0
            for j in range(len(goodMatch)):
                x, y = kp1[goodMatch[j].queryIdx].pt
                ptr = H * np.matrix([[x], [y], [1]])
                xp, yp = kp2[goodMatch[j].trainIdx].pt
                ptrp = np.matrix([[xp], [yp], [1]])
                dist = np.linalg.norm(ptrp - ptr)
                if dist <= MIN_DIS:
                    num += 1
                    list.append(goodMatch[j])
            if num >= max_num:
                maxH = H
                max_num = num
                list_max = list

        except:
            pass
    img_out = cv2.drawMatches(img1, kp1, img2, kp2, list_max, None, flags=2)
    cv2.imwrite(str(index) + 'match.jpg', img_out)
    print(max_num)
    warpImg = cv2.warpPerspective(img2, np.array(maxH), (img1.shape[1] + img2.shape[1], img2.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    direct = warpImg.copy()
    direct[0:img1.shape[0], 0:img1.shape[1]] = img1
    rows, cols = img1.shape[:2]

    for col in range(0, cols):
        if img1[:, col].any() and warpImg[:, col].any():
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if img1[:, col].any() and warpImg[:, col].any():
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows - 1):
        for col in range(0, cols):
            if not img1[row, col].any():
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = img1[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
    p = 0
    for i in range(warpImg.shape[1]):
        pixel = warpImg.shape[1] - i - 1
        for j in range(warpImg.shape[0]):
            if (warpImg[j][pixel][0] != 0 or warpImg[j][pixel][1] != 0 or warpImg[j][pixel][2] != 0):
                if p == 0:
                    p = pixel

    warpImg = warpImg[:, 0:p]
    return warpImg

img1 = cv2.imread('test1/1.jpg')
img2 = cv2.imread('test1/2.jpg')
img3 = cv2.imread('test1/3.jpg')
img4 = cv2.imread('test1/4.jpg')
img1 = cv2.resize(img1, (img1.shape[1]//compression,img1.shape[0]//compression), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (img2.shape[1]//compression,img2.shape[0]//compression), interpolation=cv2.INTER_AREA)
img3 = cv2.resize(img3, (img3.shape[1]//compression,img3.shape[0]//compression), interpolation=cv2.INTER_AREA)
img4 = cv2.resize(img4, (img4.shape[1]//compression,img4.shape[0]//compression), interpolation=cv2.INTER_AREA)
test = stitch(stitch(stitch(img1, img2),img3),img4)

cv2.imwrite('test.jpg',test)
cv2.imshow('test',test)
cv2.waitKey(0)

