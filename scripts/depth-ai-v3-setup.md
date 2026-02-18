. Install DepthAI v3
Command Line

git clone https://github.com/luxonis/depthai-core.git && cd depthai-core
python3 -m venv venv
source venv/bin/activate
# Installs library and requirements
python3 examples/python/install_requirements.py

or via pip:
Command Line

pip install depthai --force-reinstall


DepthAI Components

    Nodes represent a sensor, accelerated hardware, or some compute function
    Pipeline consists of linked nodes and gets deployed to the device where it runs on accelerated hardware blocks
    Messages are used for communication between nodes. They hold data and metadata
    Device represents Luxonis' device - OAK or OAK4 camera. It handles connectivity and communication
    Bootloader handles logic when booting RVC2 devices and makes them accessible for connection
    Luxonis OS is a custom Linux distro for RVC4 devices (OAK4)

See documentation: https://docs.luxonis.com/software-v3/depthai/api/python/