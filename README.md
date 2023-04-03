# Segementation Mask to COCO Json Converter
|           mask                      |          original    |      json_visualization |
|:-----------------------------------:|:--------------------:|:-----------------------:|
|<img src="mask/image_001.png" alt= “”>  | <img src="image/image_001.jpg" alt= “”>| <img src="image_viz/image_001.jpg" alt= “”>


### 1. Install dependencies 
```
pip install -r requirements.txt
```

### 2. Convert Segmentation to COCO Json annotation
```
python main.py -i mask
```
option: \
-i : mask directory\

### 3. Visualize COCO json annotation 
To cross check generated segmentation data, visualize it by drawing on original image.
```
python visualize_segmentation.py -i image -a mask.json
```
option: \
-i : original image directory\
-a : json annotation file

### 4. Change class name and color

Modify `config/config.yaml`.

