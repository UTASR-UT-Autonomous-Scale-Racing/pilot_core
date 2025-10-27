import socket
from pilot_controller.lap_engine import Mapper
import pickle
import time
import struct
import matplotlib.pyplot as plt

class ClientDisconectException(Exception):
    pass

def receiver(mapper):
    host = "127.0.0.1"
    port = 9001

    def recv_exact(conn, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)
    while True:
        plt.close()

        mapper.track_points = []
        mapper.car_positions = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            print(f"Listening on {host}:{port}...")
            count = 0
            try:
                while True:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected by {addr}")
                        while True:
                            header = recv_exact(conn, 4)
                            if header is None:
                                print("Client disconnected")
                                raise ClientDisconectException
                            (msg_len,) = struct.unpack('!I', header)
                            payload = recv_exact(conn, msg_len)
                            if payload is None:
                                print("Connection lost during message")
                                raise ClientDisconectException
                            try:
                                msg = pickle.loads(payload)
                                count += 1
                                i, track_points, car_position = msg
                                mapper.track_points.append(track_points)
                                mapper.car_positions.append(car_position)
                                if count % 10 == 0:
                                    mapper.update_live_plot()
                                print(f"Processed frame {i}")
                            except Exception as e:
                                print(f"Error processing message: {e}")
                                continue
            except KeyboardInterrupt:
                print("\nReceiver interrupted.")
            except ClientDisconectException:
                continue


if __name__ == "__main__":
    mapper = Mapper()
    receiver(mapper)
    mapper.finalize_lap()
    print("Lap finalized.")

