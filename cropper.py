import cv2
import os

input_path = 'right_resized'
image_names = os.listdir(input_path)

output_path = 'right_cropped'

y_max = 564
y_min = 308
x_max = 706
x_min = 450

output_image_num = 0

for image in image_names:
    print(image)
    # first, read the image
    img = cv2.imread(os.path.join(input_path,image), cv2.IMREAD_COLOR)

    #crop
    crop = img[y_min:y_max, x_min:x_max]

    #outImage = Image.fromarray(crop)
    #outImage.save(os.path.join(output_path, str(output_image_num))+".jpg")
    cv2.imwrite(os.path.join(output_path, str(output_image_num))+".jpg", crop)
    output_image_num += 1

print("FINISHED")