from PIL import Image
from pandas import read_csv
import os

labels = read_csv('Large Files/test.rotfaces/test.preds.csv').values

for item in labels:
    colorImage  = Image.open(os.path.join('Large Files/test.rotfaces/test/', item[0]))
    if item[1] == 'rotated_right':
        rotated = colorImage.rotate(90)
    if item[1] == 'rotated_left':
        rotated = colorImage.rotate(-90)
    if item[1] == 'upside_down':
        rotated = colorImage.rotate(180)
    rotated.save(os.path.join('rotated_test_images/', item[0]+'.png'))

print('Rotation complete')