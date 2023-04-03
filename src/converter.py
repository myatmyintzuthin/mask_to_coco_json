
import os
import cv2
import json
import glob
import yaml
import numpy as np
from PIL import Image 
import src.template as template
from shapely.geometry import Polygon

class SegToJson:
    
    @property
    def imgDir(self) -> str:
        return self.__img_dir
    
    @property
    def catConfig(self) -> dict:
        return self.__cat_config
    
    @property
    def colorConfig(self) -> dict:
        return self.__color_config
    
    def __init__(self, img_dir: str, config: str) -> None:
        super().__init__()
        self.__img_dir = img_dir
        self.__cat_config, self.__color_config = self.__readConfig(config)
             
    def run(self)-> None:
        
        coco_format = template.get_coco_json_format()
        coco_format["categories"] = template.create_category_annotation(self.catConfig)
        coco_format["images"], coco_format["annotations"], annotation_cnt = self.__convert(self.imgDir)
        
        json_name = os.path.basename(self.imgDir)
        with open(f"{json_name}.json","w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Created %d annotations for images in folder: %s" % (annotation_cnt, self.imgDir))
        
    def __convert(self, maskpath: str) -> tuple[list, list, int]:

        annotation_id = 0
        image_id = 0
        annotations = []
        images = []
        
        for mask_image in glob.glob(maskpath + "/*.png"):
            
            original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"

            mask_image_open = Image.open(mask_image).convert("RGB")
            w, h = mask_image_open.size
            # mask_image_open.show()
            
            image = template.create_image_annotation(original_file_name, w, h, image_id)
            images.append(image)

            sub_masks = self.__create_sub_masks(mask_image_open, w, h, self.__color_config.keys())
            
            # remove background - black color
            # color_list = ['(192, 0, 0)']
            # # print(self.__color_config.keys())
            # not_color = []
            # for k in sub_masks.keys():
            #     if k not in color_list:
            #         not_color.append(k)
            # print(not_color)
            # for k in not_color:
            #         del sub_masks[k]
            
            # # print(sub_masks)
            # background = '(0, 0, 0)'
            # if background in sub_masks.keys():
            #     del sub_masks[background]
                
            for color, sub_mask in sub_masks.items():
                category_id = self.colorConfig[color]
                
                polygons = self.__create_sub_mask_annotation(sub_mask)
            
                for i in range(len(polygons)):
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    annotation = template.create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                    annotations.append(annotation)
                    annotation_id += 1
            image_id += 1
            
        return images, annotations, annotation_id
    
    @staticmethod
    def __create_sub_masks(mask_image: Image.Image, width: int, height: int, color_list: list ) -> dict:
        # Initialize a dictionary of sub-masks indexed by RGB colors
        sub_masks = {}
        for x in range(width):
            for y in range(height):
                pixel = mask_image.getpixel((x,y))[:3]
                
                if(str(pixel) in color_list):

                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                        sub_masks[pixel_str] = Image.new("1", (width+2, height+2))
                    sub_masks[pixel_str].putpixel((x+1, y+1), 1)
        return sub_masks
    
    @staticmethod    
    def __create_sub_mask_annotation(sub_mask: Image.Image) -> list:
    
        sub_mask = np.array(sub_mask) 
        sub_mask = sub_mask.astype(np.uint8)
        sub_mask *= 255
    
        contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
        
            for i in range(len(contour)):
                row, col = contour[i][0]
                contour[i] = (row - 1, col - 1)
            
            contour = np.reshape(contour, (-1, 2))
            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.5, preserve_topology=False)
            if(poly.is_empty):
                continue
            polygons.append(poly)
            
        return polygons
    
    @staticmethod
    def __readConfig(config: str) -> tuple[dict, dict]:
        with open(config, "r") as stream:
            try:
                all_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        color_config = all_config["category_colors"]
        color_config = {v:k for k,v in color_config.items()}
        cat_config = all_config["category_ids"]
        return cat_config, color_config