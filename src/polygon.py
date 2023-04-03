import cv2

class Polygon:
    
    @property
    def cords(self) -> list[list]:
        return self.__point
    
    @property
    def area(self) -> float:
        return self.__area
    
    @property
    def bbox(self) -> float:
        return self.__bbox
        
    def __init__(self, point:list[list]) -> None:
        self.__point = self.__simplify(point)
        self.__area = self.__calculateArea()
        self.__bbox = self.__bboxCoords()
        
    def __calculateArea(self) -> float:
        area = cv2.contourArea(self.cords)
        return area
    
    def __bboxCoords(self) -> list:
        
        (x_center, y_center), (w, h), angle = cv2.minAreaRect(self.cords)
        xmin, ymin, xmax, ymax = int(x_center-(w/2)), int(y_center-(h/2)), int(x_center+(w/2)), int(y_center+(h/2))
        bbox = [xmin, ymin, xmax, ymax]
        return bbox
    
    def __simplify(self, point: list[list]):
        
        epsilon = 0.001*cv2.arcLength(point,True)
        simplified = cv2.approxPolyDP(point,epsilon,True)
        return simplified
        
        
    
    