
import os
import cv2
import json
import yaml
import argparse
import numpy as np
import ast
from collections import defaultdict

def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        prog='visualize',
        description='Visualization of coco Json')
    parser.add_argument('-i', '--imgdir')
    parser.add_argument('-a', '--anno')
    parser.add_argument('-c', '--config', default="configs/config.yaml")
    args = parser.parse_args()
    return args

class Visualize:
    
    @property
    def imgDir(self) -> str:
        return self.__img_dir
    
    @property
    def annoPath(self) -> str:
        return self.__anno_path
    
    @property
    def config(self) -> str:
        return self.__config
        
    @property
    def saveDir(self) -> str:
        return self.__save_dir
     
    def __init__(self, img_dir: str, anno_path: str, config: str) -> None:
        super().__init__()
        self.__img_dir = img_dir
        self.__anno_path = anno_path
        self.__config = self.__readConfig(config)
        self.__save_dir = f"{self.__img_dir}_viz"
        os.makedirs(self.__save_dir, exist_ok=True)

    def run(self) -> None:
        print("[INFO] Processing data, please wait ...")
        anno_map, img_map, cat_map = self.__readJson()
        self.__drawAnnotation(anno_map, img_map, cat_map)
        print(f"[INFO] Visualization results saved in {self.saveDir}")

    def __drawAnnotation(self, anno_map: dict, img_map: dict, cat_map: dict) -> None:
        
        for i in range(len(img_map)):
            image_name = img_map[i]
            image = cv2.imread(os.path.join(self.imgDir, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result_img = os.path.join(self.saveDir, image_name)
            anno_item = anno_map[i]
            copy_image = image.copy()
            for j in range(len(anno_item)):
                cat_id, bbox, segmentation = anno_item[j]

                segments = np.array(segmentation)
                segments = np.reshape(segments, (-1, 2))
                segments = segments.astype(int)
                segments = segments[ :,np.newaxis, :]
                
                mask_color = ast.literal_eval(self.config[cat_id])
                copy_image = cv2.drawContours(copy_image, [segments], -1, mask_color, -1)
                
                # copy = cv2.drawContours(copy, [segments], -1, mask_color, -1)
            
            result = cv2.addWeighted(image, 0.5, copy_image, 0.5, 0)
            copy_image = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            # self.__showImage(copy_image, "result image")   
            cv2.imwrite(result_img, copy_image)
    
    def __readJson(self) -> tuple[dict, dict, dict]:
        with open(self.annoPath, 'r') as reader:
            json_data = json.load(reader)

        annotations = json_data['annotations']
        images = json_data['images']
        categories = json_data['categories']

        anno_map = self.__getAnnotationMap(annotations)
        img_map = self.__getImageIdMap(images)
        cat_map = self.__getCatIdMap(categories)

        return anno_map, img_map, cat_map
    
    @staticmethod
    def __showImage(img: cv2.Mat, windowName: str) -> None:
        cv2.imshow(windowName, img)
        key = cv2.waitKey(0)
        if key == 27:  # wait for ESC key to exit
            cv2.destroyWindow('visualize')
            
    @staticmethod
    def __getAnnotationMap(annotations: list) -> dict:
        annotation_map = defaultdict(list)
        for annotation in annotations:
            annotation_map[annotation["image_id"]].append(
                [annotation["category_id"], annotation["bbox"], annotation["segmentation"]])
        return annotation_map

    @staticmethod
    def __getImageIdMap(images: list) -> dict:
        image_map = {}
        for image in images:
            image_map[image['id']] = image['file_name']
        return image_map

    @staticmethod
    def __getCatIdMap(categories: list) -> dict:
        cat_map = {}
        for cat in categories:
            cat_map[cat['id']] = cat['name']
        return cat_map
    
    @staticmethod
    def __readConfig(config: str) -> dict:
        with open(config, "r") as stream:
            try:
                all_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        color_config = all_config["category_colors"]
        return color_config
            

if __name__ == "__main__":

    args = get_args()
    visualizer = Visualize(args.imgdir, args.anno, args.config)
    visualizer.run()
