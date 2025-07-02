import torch
import torch.nn as nn

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, num_bboxes=2):
        super(SimpleYOLO, self).__init__()
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        # 每个网格预测 num_bboxes * (x, y, w, h, confidence) + 类别概率
        self.output_dim = grid_size * grid_size * (num_bboxes * 5 + num_classes)

        # 骨干网络：简单的 CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        # 重塑输出为 (batch_size, grid_size, grid_size, num_bboxes * 5 + num_classes)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_bboxes * 5 + self.num_classes)
        return x

def main():
    # 输入图像尺寸：(batch_size, channels, height, width)
    batch_size = 1
    img = torch.randn(batch_size, 3, 448, 448)  # YOLO 通常使用 448x448 输入
    model = SimpleYOLO(num_classes=20, grid_size=7, num_bboxes=2)
    output = model(img)
    print(f"Output shape: {output.shape}")  # 应为 (batch_size, 7, 7, 30)

if __name__ == "__main__":
    main()

