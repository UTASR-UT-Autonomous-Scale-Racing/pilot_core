source /home/barracuda/Desktop/pilot_core/.venv/bin/activate

export LD_LIBRARY_PATH=$HOME/Desktop/pilot_core/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH}
python3 start.py --enable-slam --debug