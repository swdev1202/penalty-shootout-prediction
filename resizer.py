import cv2
import os

input_path = 'right'
image_names = os.listdir(input_path)

output_path = 'right_resized'

output_image_num = 0

for image in image_names:
    print(image)
    # first, read the image
    img = cv2.imread(os.path.join(input_path,image), cv2.IMREAD_COLOR)

    #resize
    newimg = cv2.resize(img,(1280,720))
    #write
    cv2.imwrite(os.path.join(output_path, str(output_image_num))+".jpg", newimg)


    output_image_num += 1

print("FINISHED")