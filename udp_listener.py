# udp_listener.py
import socket
import threading

class UDPListener:
    def __init__(self, name: str, port: int, data_handler_callback):
        self.name        = name
        self.port        = port
        self.data_handler = data_handler_callback

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port))
        self.sock.settimeout(1.0)

        self.thread  = threading.Thread(target=self._listen, daemon=True)
        self.running = False

    def _listen(self):
        print(f"Listener UDP '{self.name}' iniciado en el puerto {self.port}.")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                if data:
                    self.data_handler(data, addr)   # ← pasa addr también
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error en listener '{self.name}': {e}")
        self.sock.close()
        print(f"Listener UDP '{self.name}' detenido.")

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()