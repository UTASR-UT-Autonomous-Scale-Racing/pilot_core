import socket
import time
import numpy as np
import pickle
import struct
from pilot_controller.lap_engine import Mapper


def sender(mapper):

    start_time = time.time()

    depth_line = np.zeros(64, dtype=float)

    with open('/home/zebrafish/Desktop/sensor_data.pkl', 'rb') as file:
        my_list = pickle.load(file)

    for i in range(len(my_list)):
        t, ax, ay, az, w, x, y, z = my_list[i]
        mapper.process_step(t, depth_line, ax, ay, az, w, x, y, z, 0, 0, 0, 0)
        #time.sleep(1)
    
    i = 1
    # Send all frames over a single connection; reconnect only on failure
    while i < len(my_list):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                s.settimeout(None)
                while i < len(my_list):
                    t, ax, ay, az, w, x, y, z = my_list[i]
                    track_points, car_position = mapper.process_step(t, depth_line, ax, ay, az, w, x, y, z, 0, 0, 0, 0)
                    msg = [i, track_points, car_position]
                    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
                    header = struct.pack('!I', len(payload))
                    s.sendall(header + payload)
                    print(f"Sent frame {i}")
                    i += 1
                print("All frames sent.")
                break  # done
        except (ConnectionRefusedError, socket.timeout, socket.error) as e:
            print(f"Connection/send failed at frame {i}: {e}. Retrying in 1s...")
            time.sleep(1)
    

if __name__ == "__main__":
    mapper = Mapper(broadcast_port=9001)
    sender(mapper)

