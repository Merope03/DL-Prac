from TwoLayerNetPrac import MyNet
import numpy as np

Network = MyNet(28*56, 300, 200, 100)

data = np.load('mynet_params.npz')
params = [data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']]
Network.params = params
# 각 레이어에도 재할당
Network.layers[0].W, Network.layers[0].b = params[0], params[1]
Network.layers[2].W, Network.layers[2].b = params[2], params[3]
Network.layers[4].W, Network.layers[4].b = params[4], params[5]

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# 네가 만든 신경망 import
# from Net import MyNet
# Network = ... # 파라미터 불러오기/학습한 네트워크 세팅

WIDTH, HEIGHT = 560, 280  # 그릴 창 크기 (MNIST 10배)
DRAW_RADIUS = 8

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("손글씨 숫자 인식기 (MNIST)")

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack()
        self.image = Image.new("L", (WIDTH, HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.label = tk.Label(root, text="예측 결과: ", font=("Arial", 24))
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.root.bind("<Return>", self.predict)
        self.root.bind("<BackSpace>", self.clear)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-DRAW_RADIUS, y-DRAW_RADIUS, x+DRAW_RADIUS, y+DRAW_RADIUS, fill='black')
        self.draw.ellipse([x-DRAW_RADIUS, y-DRAW_RADIUS, x+DRAW_RADIUS, y+DRAW_RADIUS], fill='black')

    def predict(self, event=None):
        img = self.image.resize((28, 56), Image.LANCZOS)
        img = ImageOps.invert(img)  # MNIST: 흑배경/백글씨라면 주석처리
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.reshape(1, 28*56)

        # 네트워크 예측
        pred = Network.predict(arr)
        digit = np.argmax(pred, axis=1)[0]
        self.label.config(text=f"예측 결과: {digit}")

    def clear(self, event=None):
        self.canvas.delete("all")
        self.image = Image.new("L", (WIDTH, HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="예측 결과: ")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

