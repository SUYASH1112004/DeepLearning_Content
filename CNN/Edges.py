#Detect Edges in an Image
import cv2

#Read Image
img=cv2.imread("photo.jpg",0)

#Canny Edge Detection
edges=cv2.Canny(img,100,200)

cv2.imshow("Original",img)
cv2.imshow("Edges",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()