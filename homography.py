## Be able to upload two images and get the homography matrix and stitch them together

import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def get_keypoints(image, nfeatures=4000):
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp = orb.detect(greyscale_image, None)
    print(f"Keypoints found: {len(kp)}")
    kp, des = orb.compute(greyscale_image, kp)
    kp = np.array([kp.pt for kp in kp])
    return kp, des


def match_keypoints(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m.queryIdx,n.trainIdx])
    good = np.array(good, dtype=np.int32)
    return good

def reprojection_h(src, dst,H, threshold):
    src_h = np.hstack((src.astype(np.float64), np.ones((len(src), 1))))
    proj  = (H @ src_h.T).T                  # (N,3)
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj_xy - dst.astype(np.float64), axis=1)
    inlier_mask = (err <= threshold)
    return inlier_mask, err

def findHomography(src, dst, iterations, threshold):
    print(f"Estimating Homography with {iterations} iterations and {threshold} threshold")
    best_H = None
    max_inliers = 0
    best_mask = None
    for i in range(iterations):
        sample_indices = np.random.choice(src.shape[0], size=4, replace=False)
        src_samples = src[sample_indices]
        dst_samples = dst[sample_indices]
        ## Normalize Samples
        eps = 1e-6
        src_mean = np.mean(src_samples, axis=0)
        src_delta = src_samples - src_mean
        src_var = np.mean(np.sqrt(src_delta[:, 0]**2 + src_delta[:,1]**2), axis=0)
        src_s = np.sqrt(2)/max(src_var,eps) 
        dst_mean = np.mean(dst_samples, axis=0)
        dst_delta = dst_samples - dst_mean
        dst_var = np.mean(np.sqrt(dst_delta[:, 0]**2 +dst_delta[:, 1]**2), axis=0)
        dst_s = np.sqrt(2)/max(dst_var, eps)
        T_dst = np.array([[dst_s, 0, -1*dst_s*dst_mean[0]],
                        [0, dst_s, -1*dst_s*dst_mean[1]],
                        [0,0,1]])  
        T_src = np.array([[src_s, 0, -1*src_s*src_mean[0]],
                        [0, src_s, -1*src_s*src_mean[1]],
                        [0,0,1]])  
        src_h = np.hstack((src_samples, np.ones((len(src_samples),1))))
        dst_h = np.hstack((dst_samples, np.ones((len(dst_samples),1))))

        A = np.zeros((8, 9), dtype=np.float64)
        src_samples = (T_src @ src_h.T).T
        dst_samples = (T_dst@ dst_h.T).T
        for k in range(len(src_samples)):
            src_x, src_y, w = src_samples[k]
            dst_x, dst_y, w_ = dst_samples[k]
            vec_1 = np.array((-src_x, -src_y, -1, 0, 0, 0, src_x*dst_x, dst_x*src_y, dst_x))
            vec_2 = np.array((0,0,0, -src_x, -src_y, -1,src_x*dst_y, src_y*dst_y, dst_y))
            A[2*k, :] = vec_1
            A[2*k +1, :] = vec_2
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1]            # (9,)
        Hn = h.reshape(3, 3)  # normalized H
        H  = np.linalg.inv(T_dst) @ Hn @ T_src
        H /= H[2, 2]
        inliers_mask, inliers = reprojection_h(src, dst, H, threshold)
        n_in = int(inliers_mask.sum())

        if n_in > max_inliers:
            max_inliers = n_in
            best_H = H
            best_mask = inliers_mask
            print(f"Updated best: {max_inliers} inliers")

    if best_mask is None or max_inliers < 4:
        raise RuntimeError("Failed to estimate a homography")

    eps = 1e-6
    src_samples = src[best_mask]
    dst_samples = dst[best_mask]
    src_mean = np.mean(src_samples, axis=0)
    src_delta = src_samples - src_mean
    src_var = np.mean(np.sqrt(src_delta[:, 0]**2 + src_delta[:,1]**2), axis=0)
    src_s = np.sqrt(2)/max(src_var,eps) 
    dst_mean = np.mean(dst_samples, axis=0)
    dst_delta = dst_samples - dst_mean
    dst_var = np.mean(np.sqrt(dst_delta[:, 0]**2 +dst_delta[:, 1]**2), axis=0)
    dst_s = np.sqrt(2)/max(dst_var, eps)
    T_dst = np.array([[dst_s, 0, -1*dst_s*dst_mean[0]],
                    [0, dst_s, -1*dst_s*dst_mean[1]],
                    [0,0,1]])  
    T_src = np.array([[src_s, 0, -1*src_s*src_mean[0]],
                    [0, src_s, -1*src_s*src_mean[1]],
                    [0,0,1]])  
    src_h = np.hstack((src_samples, np.ones((len(src_samples),1))))
    dst_h = np.hstack((dst_samples, np.ones((len(dst_samples),1))))

    A = np.zeros((2*len(src_samples), 9), dtype=np.float64)
    src_samples = (T_src @ src_h.T).T
    dst_samples = (T_dst@ dst_h.T).T
    for k in range(len(src_samples)):
        src_x, src_y, w = src_samples[k]
        dst_x, dst_y, w_ = dst_samples[k]
        vec_1 = np.array((-src_x, -src_y, -1, 0, 0, 0, src_x*dst_x, dst_x*src_y, dst_x))
        vec_2 = np.array((0,0,0, -src_x, -src_y, -1,src_x*dst_y, src_y*dst_y, dst_y))
        A[2*k, :] = vec_1
        A[2*k +1, :] = vec_2
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]            # (9,)
    Hn = h.reshape(3, 3)  # normalized H
    H  = np.linalg.inv(T_dst) @ Hn @ T_src
    H /= H[2, 2]
    return H


def bounds(img1_shape, img2_shape, H):
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    corners = np.array([[0,0], [w1-1,0], [w1-1, h1-1], [0, h1-1]], dtype=np.float64)
    corners = np.hstack([corners, np.ones((4,1))])
    warped = (H @ corners.T).T # Map corners to points in new plane
    warped_xy = warped[:, :2]/warped[:, 2:3] ## Normalize by last dimension such that everything is on the same plane again
    all_x = np.hstack([warped_xy[:,0], [0, w2-1]])
    all_y = np.hstack([warped_xy[:,1], [0, h2-1]])

    min_x = int(np.floor(all_x.min()))
    min_y = int(np.floor(all_y.min()))
    max_x = int(np.ceil(all_x.max()))
    max_y = int(np.ceil(all_y.max()))

    W = max_x - min_x + 1
    Hh = max_y - min_y + 1

    # translation to keep everything positive on canvas
    T = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]], dtype=np.float64)
    return (W, Hh), T, (min_x, min_y)




def warp_to_canvas(img_src, H, canvas_size, T):
    H_canvas = T @ H                  # include translation
    warped = cv2.warpPerspective(img_src, H_canvas, canvas_size)
    mask = cv2.warpPerspective(np.ones(img_src.shape[:2], np.uint8)*255, H_canvas, canvas_size)
    return warped, mask
def paste_reference(img_ref, canvas_size, offset):
    W, Hh = canvas_size
    tx, ty = offset  # can be negative
    h_r, w_r = img_ref.shape[:2]

    # 1) intersection on the canvas
    x_start = max(0, tx)
    x_end   = min(W, tx + w_r)
    y_start = max(0, ty)
    y_end   = min(Hh, ty + h_r)

    out  = np.zeros((Hh, W, 3), dtype=np.uint8)
    mask = np.zeros((Hh, W), dtype=np.uint8)

    # 2) empty intersection? nothing to paste
    if x_end <= x_start or y_end <= y_start:
        return out, mask

    # 3) map back to ref-image coordinates
    ref_x0 = x_start - tx
    ref_x1 = x_end   - tx
    ref_y0 = y_start - ty
    ref_y1 = y_end   - ty

    # 4) paste
    out[y_start:y_end, x_start:x_end] = img_ref[ref_y0:ref_y1, ref_x0:ref_x1]
    mask[y_start:y_end, x_start:x_end] = 255
    return out, mask

def simple_blend(warped_img, warped_mask, ref_img_on_canvas, ref_mask):
    # where both valid → average, else take valid
    out = warped_img.copy()
    both = (warped_mask>0) & (ref_mask>0)
    only_ref = (warped_mask==0) & (ref_mask>0)
    out[only_ref] = ref_img_on_canvas[only_ref]
    out[both] = ((out[both].astype(np.float32) + ref_img_on_canvas[both].astype(np.float32)) * 0.5).astype(np.uint8)
    return out


def combine(images):
    kp1, des1 = get_keypoints(images[0])
    kp2, des2 = get_keypoints(images[1])
    matches = match_keypoints(des1, des2)
    print(f"Keypoints Source: {kp1.shape}")
    print(f"Keypoints Destination: {kp2.shape}")
    print(f"Matches found: {matches.shape}")
    if len(matches) < 4:
        raise ValueError("Not enough matches found")
    print(f"First 4 matches: {matches[:4]}")
    src = kp1[matches[:, 0]]
    dst = kp2[matches[:, 1]]
    
    H = findHomography(src, dst, 20000, 3)

    canvas_size, T, offset = bounds(images[0].shape, images[1].shape, H)
    print(f"Computed Canvassize {canvas_size} with offset {offset}")

    W, Hh = canvas_size

    # warp img1 into canvas
    H_canvas = T @ H
    warped1, mask1 = warp_to_canvas(images[0], H, (W, Hh), T)   # or inverse_warp_nn(img1, H_canvas, (W,Hh))

    # paste img2 into canvas
    ref2_on_canvas, mask2 = paste_reference(images[1], (W, Hh), offset)

    # blend
    stitched = simple_blend(warped1, mask1, ref2_on_canvas, mask2)
    plt.imshow(stitched); plt.axis('off'); plt.show()

    return 


def stitch_images(img1_bgr, img2_bgr, iterations: int = 8000, threshold: float = 3.0):
    """Stitch two BGR images and return the stitched BGR image."""
    kp1, des1 = get_keypoints(img1_bgr)
    kp2, des2 = get_keypoints(img2_bgr)
    matches = match_keypoints(des1, des2)
    if len(matches) < 4:
        raise ValueError("Not enough matches found")
    src = kp1[matches[:, 0]]
    dst = kp2[matches[:, 1]]

    H = findHomography(src, dst, iterations, threshold)
    canvas_size, T, offset = bounds(img1_bgr.shape, img2_bgr.shape, H)
    W, Hh = canvas_size

    warped1, mask1 = warp_to_canvas(img1_bgr, H, (W, Hh), T)
    ref2_on_canvas, mask2 = paste_reference(img2_bgr, (W, Hh), offset)
    stitched = simple_blend(warped1, mask1, ref2_on_canvas, mask2)
    return stitched



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, nargs="+", help="Path to the images to stitch")
    args = ap.parse_args()
    images = args.images
    if len(images) < 2:
        raise ValueError("At least two images are required to stitch")
    images_bgr = [cv2.imread(image, cv2.IMREAD_COLOR) for image in images]
    stitched = stitch_images(images_bgr[0], images_bgr[1])
    out_path = "stitched_output.png"
    cv2.imwrite(out_path, stitched)
    print(f"Wrote {out_path}")
if __name__ == "__main__":
    main()