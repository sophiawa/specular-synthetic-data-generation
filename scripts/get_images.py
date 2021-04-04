import os


count = 0
for i in range(2):
    os.system('python run.py examples/coco_annotations/config.yaml m examples/coco_annotations/pan_scene.blend examples/coco_annotations/output resources/cctextures resources/haven')
    str_index = '0' * (6 - len(str(count))) + str(count)
    for j in range(2):
        os.system('python scripts/vis_coco_annotation.py -i ' + str(j) + '-gi ' + str_index)
        os.system('python scripts/visHdf5Files.py examples/coco_annotations/output/' + str(j) + '.hdf5' + str_index)
        count += 1

