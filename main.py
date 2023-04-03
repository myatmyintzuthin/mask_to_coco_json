
import argparse
from src.converter import SegToJson

def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
    prog='visualize',
    description='Visualization of coco Json')
    parser.add_argument('-i', '--imgdir')
    parser.add_argument('-c', '--config', default="configs/config.yaml")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    converter = SegToJson(args.imgdir, args.config)
    converter.run()
    
    
