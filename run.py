
import argparse
import os, errno
import sys
import yaml

from program import Program
from models.rec.model import Model
from data.data_loader import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="text_recognition model training session."
    )
    
    parser.add_argument(
        "-pt",
        "--phoneme_type",
        type=bool,
        help="Set target source as {True:phoneme / False:character}",
        default=False
    )
    
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    if args.phoneme_type:
        with open('./conf/phoneme.yml') as f:
            #https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            conf = yaml.load(f,Loader=yaml.FullLoader)
    else:
        with open('./conf/character.yml') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
    
    program = Program(conf['Program'])
    
    dataloader = DataLoader(conf['Basic'])
    
    model = Model(conf['Model'])
            
    

if __name__ == "__main__":
    main()
