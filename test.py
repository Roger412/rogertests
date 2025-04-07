import cv2
import numpy as np

K = np.eye(3)
P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K @ np.hstack((np.eye(3), np.array([[1],[0],[0]])))

pts1 = np.array([[100, 150], [200, 250], [300, 350], [400, 450]], dtype=np.float32)
pts2 = np.array([[102, 152], [202, 252], [298, 348], [398, 448]], dtype=np.float32)

pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts4D /= pts4D[3]
print("Triangulated points:\n", pts4D[:3].T)
