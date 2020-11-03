import numpy as np
import cv2
import random
import math

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15



def alignImages(im1, im2):

    im1Gray = im1
    im2Gray = im2
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort Matches by Score
    matches = sorted(matches, key = lambda x: x.distance, reverse=False)

    # Remove bad matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)

    matches = matches[:numGoodMatches]

    # Draw Top Matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite('matches.png', imMatches)

    # Extract Location of Good Matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i,:] = keypoints1[match.queryIdx].pt
        points2[i,:] = keypoints2[match.trainIdx].pt
    
    # TODO: Find Homography
    H, b = findBestHomography(points1, points2, 2500, 25)
    np.save('OutputP5H.npy', H)
    
    # TODO: Warp Perspective
    height, width, channels = im2.shape
    
    im1Reg = homography_warp(im1, H, width, height)
    

    
    return im1Reg


def findHomography(src, dst):
    # returns H matrix of homography paramters
    # 3x3 ndarray
    x1p, y1p = dst[0][0], dst[0][1]
    x2p, y2p = dst[1][0], dst[1][1]
    x3p, y3p = dst[2][0], dst[2][1]
    x4p, y4p = dst[3][0], dst[3][1]

    x1, y1 = src[0][0], src[0][1]
    x2, y2 = src[1][0], src[1][1]
    x3, y3 = src[2][0], src[2][1]
    x4, y4 = src[3][0], src[3][1]
    
    B = np.asarray([
        [-x1, -y1, -1, 0, 0, 0, x1p*x1, x1p*y1, x1p], \
        [0, 0, 0, -x1, -y1, -1, y1p*x1, y1p*y1, y1p], \
        [-x2, -y2, -1, 0, 0, 0, x2p*x2, x2p*y2, x2p], \
        [0, 0, 0, -x2, -y2, -1, y2p*x2, y2p*y2, y2p], \
        [-x3, -y3, -1, 0, 0, 0, x3p*x3, x3p*y3, x3p], \
        [0, 0, 0, -x3, -y3, -1, y3p*x3, y3p*y3, y3p], \
        [-x4, -y4, -1, 0, 0, 0, x4p*x4, x4p*y4, x4p], \
        [0, 0, 0, -x4, -y4, -1, y4p*x4, y4p*y4, y4p] \
    ])
    
    try:
        '''
        B_invert = np.linalg.inv(B)
        transformedPoints2 = np.array([[x1p], [y1p], [x2p], [y2p], [x3p], [y3p], [x4p], [y4p]])

        homography_parameters = np.dot(B_invert, transformedPoints2)

        appendH = np.append(homography_parameters,[1])

        H = appendH.reshape((3,3))
        '''
        u, s, vh = np.linalg.svd(B)
        L = vh[8].reshape(3,3)
        H = L/ L[2][2]
        
        return H
    except:
        return []


def applyHomography(H, src):
    # returns the transformed destination points
    # Inputs: 
    # H -> 3x3
    # dst -> 3x1

    nd_d = np.array([[src[0]], [src[1]], [1.0]])
    out = np.dot(H, nd_d)
    out = out.flatten()
    out = out.tolist()
    out = [out[0]/out[2], out[1]/out[2]]
    return out


def findBestHomography(xs, xd, maxIterations, eps):
    # TODO: Apply RANSAC to find inlier points
    bestH = None
    bestInlierCount = 0
    
    for _ in range(maxIterations):
        # Choose Four Random Points
        allChoices = [a for a in range(len(xs))]
        rand_idx = []
        rand_src = []
        rand_dst = []
        # choose four random indicies
        while len(rand_idx) < 4:
            r = random.choice(allChoices)
            if r not in rand_idx and r in allChoices:
                allChoices.remove(r)
                rand_idx.append(r)
        
        for j in rand_idx:
            rand_src.append(xs[j])
            rand_dst.append(xd[j])
        
        # TODO: Obtain Random Homography Parameter Matrix

        H = findHomography(rand_src, rand_dst)
        if len(H) > 0:
            # TODO: Apply Homography to compare transformed value to see if it is within the epsilon value
            curInlierCount = 0
            for c,s in enumerate(xs):
                t_xd = applyHomography(H, s)
                distance = findDistance(t_xd, xd[c])
                if distance < eps:
                    curInlierCount+=1
            
            # if the current H produces the most inliers than use it!
            if curInlierCount > bestInlierCount:
                bestH = H
                bestInlierCount = curInlierCount

    print(bestH)
    return bestH, bestInlierCount

def findDistance(p1, p2):
    xp1, yp1 = p1[0], p1[1]
    xp2, yp2 = p2[0], p2[1]

    return math.sqrt((xp1 - xp2)**2 + (yp1-yp2)**2)

def homography_warp(im1, H, width, height):
    
    imH, imW, channels = im1.shape
    out = np.zeros((height, width, 3), dtype='uint8')

    out_h, out_w, channels_out = out.shape

    '''
    H -> 3x3
    Obtain each coordinate in original im1
    Apply H
    Obtain new coordinate, go to new coordinate in out, copy the BGR color channels to there
    '''

    for y in range(out_h):
        for x in range(out_w):
            if (y >= 0 and x >= 0 and y < imH and x < imW):
                og_color = im1[y][x]
                new_coord = applyHomography(H, [x, y])
                new_y = int(new_coord[1])
                new_x = int(new_coord[0])
                if (new_y >= 0 and new_x >=0 and new_y < out_h and new_x < out_w):
                    out[new_y, new_x] = og_color

                    # To Handle Peppering across the image and from rounding erros from the float coordinates
                    if (new_y+1 < out_h):
                        out[new_y+1][new_x] = og_color
                    if (new_y -1 >= 0):
                        out[new_y-1][new_x] = og_color
                    if (new_x+1 < out_w):
                        out[new_y][new_x+1] = og_color
                    if (new_x-1 >= 0):
                        out[new_y][new_x-1] = og_color
                    
                    if (new_y+1 < out_h and new_x+1 < out_w):
                        out[new_y+1][new_x+1] = og_color
                    if (new_y -1 >= 0 and new_x -1 >=0):
                        out[new_y-1][new_x-1] = og_color
                    if (new_x+1 < out_w and new_y-1 >=0):
                        out[new_y-1][new_x+1] = og_color
                    if (new_x-1 >= 0 and new_y +1 < out_h):
                        out[new_y+1][new_x-1] = og_color

    return out


def stitchedImages(im1Warped, im2):
    h1, w1, c1 = im1Warped.shape
    h2, w2, c2 = im2.shape

    out = np.zeros((h2, w2, 3), dtype='uint8')

    hOut, wOut, cOut = out.shape

    for y in range(hOut):
        for x in range(wOut):
            if (y >=0 and y < h1 and x>=0 and x < w1):
                if (sum(im1Warped[y][x]) != 0):
                    avgCoord = [int(0.5*sum(x)) for x in zip(im1Warped[y][x], im2[y][x])]
                    out[y][x] = avgCoord
                else:
                    out[y][x] = im2[y][x]
    
    return out

    

if __name__ == '__main__':
    refImage = 'wall1.png'
    imReference = cv2.imread(refImage, cv2.IMREAD_COLOR)
    alignImage = 'wall2.png'
    im = cv2.imread(alignImage, cv2.IMREAD_COLOR)
    res = alignImages(imReference, im)

    stitched = stitchedImages(res, im)
    # Output
    warpedFileName = 'warpedWall1.png'
    cv2.imwrite(warpedFileName, res)

    outFileName = 'outputP5Wall.png'
    cv2.imwrite(outFileName, stitched)
