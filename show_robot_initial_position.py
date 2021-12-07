from project.simulation.robot import *

def main():
    xml_path = "./project/models/vx300s/vx300s.xml"
    robot = VX300s(xml_path)
    robot.show_simulation()

if __name__== "__main__":
    main()