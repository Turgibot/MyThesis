from zipfile import Path
from project.simulation.scene import *
from project.simulation import path_maker as pm

def main():
    xml_path = "./project/models/vx300s/vx300s.xml"
    scene = Scene(xml_path)
    # path = pm.Path(scene=scene)
    
    scene.advance()
if __name__== "__main__":
    main()