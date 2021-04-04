import argparse
import json
import os
import cv2

from pycocotools import mask
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--conf', dest='conf', default='coco_annotations.json', help='coco annotation json file')
parser.add_argument('-i', '--image_index', dest='image_index', default=0, help='image over which to annotate, uses the rgb rendering', type=int)
parser.add_argument('-b', '--base_path', dest='base_path', default='examples/coco_annotations/output/coco_data', help='path to folder with coco_annotation.json and images', type=str)
parser.add_argument('--save', '-s', action='store_true', help='saves visualization of coco annotations under base_path/coco_annotated_x.png ')

args = parser.parse_args()

conf = args.conf
image_idx = args.image_index
base_path = args.base_path
#save = args.save
save = True

# Read coco_annotations config
with open(os.path.join(base_path, conf)) as f:
    annotations = json.load(f)
    categories = annotations['categories']
    annotations = annotations['annotations']

im_path = os.path.join(base_path, "rgb_{:04d}.png".format(image_idx))
if os.path.exists(im_path):
    im = Image.open(im_path)
else:
    im = Image.open(im_path.replace('png', 'jpg'))

def get_category(_id):
    category = [category["name"] for category in categories if category["id"] == _id]
    if len(category) != 0:
        return category[0]
    else:
        raise Exception("Category {} is not defined in {}".format(_id, os.path.join(base_path, conf)))
im.show()






coords = []
im_arr = np.array(im)
height, width = len(im_arr), len(im_arr[0])
im8bit = im.convert('L')
img8bit = np.array(im8bit)

edges = cv2.Canny( img8bit, 20, 1000 )
#Image.fromarray(edges).show()




font = ImageFont.load_default()
#print(annotations)
# Add bounding boxes and masks
overlay = Image.new('L', im.size)
edges = Image.fromarray(edges)
for idx, annotation in enumerate(annotations):
    #print(annotation)
    if annotation["image_id"] == image_idx:

        

        draw = ImageDraw.Draw(im)
        #im.show()
        bb = annotation['bbox']
        #draw.rectangle(((bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3])), fill=None, outline="red")
        #draw.text((bb[0] + 2, bb[1] + 2), get_category(annotation["category_id"]), font=font)
        if isinstance(annotation["segmentation"], dict):
            im.putalpha(255)
            an_sg = annotation["segmentation"]
            item = mask.decode(mask.frPyObjects(an_sg, im.size[1], im.size[0])).astype(np.uint8) * 255

            negs = item - 255
            negs = np.where(negs==1, 255, negs)
            negs = Image.fromarray(negs, mode='L')
            
            draw_outlines = ImageDraw.Draw(edges)
            draw_outlines.bitmap((0, 0), negs, fill=(0))
            #edges.show()
            
            im_outlines = edges
            im_outlines = Image.fromarray(np.array(im_outlines))
            #im_outlines.show()
            

            #item = Image.fromarray(item, mode='L')
            #item.show()
            item = Image.fromarray(item, mode='L')
            #item.show()
            draw_ov = ImageDraw.Draw(overlay)
            draw_ov.bitmap((0, 0), item, fill=(255))
            


            im = overlay
            #print("IMAGE:", list(im.getdata()))
            #print("IMAGE", overlay)
            im = Image.fromarray(np.array(im))
            #im.show()
            #im = Image.alpha_composite(im, overlay)
        else:
            item = annotation["segmentation"][0]
            poly = Image.new('RGBA', im.size)
            pdraw = ImageDraw.Draw(poly)
            pdraw.polygon(item, fill=(0, 0, 0, 0), outline=(0, 0, 0, 0))
            im.paste(poly, mask=poly)
if save:
    im.save(os.path.join(base_path, 'coco_annotated_{}.png'.format(image_idx)), "PNG")
    
im.show()
im.save('mask.png')
#print(im)


#edges = np.where(edges==255., 0.00392157, edges)
#edges = np.where(edges==255., 0.00392157, edges)


#plt.imshow(edges)
#print(np.unique(edges, return_counts=True))
#plt.show()

#Image.fromarray(edges).show()
