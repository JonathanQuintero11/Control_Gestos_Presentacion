from pyfirmata2 import Arduino
from time import sleep

tarjeta = Arduino('COM5')  # Cambiar por el puerto de esa computadora
led = tarjeta.get_pin('d:2:o')

while True:
    led.write(1)
    sleep(1)
    led.write(0)
    sleep(1)