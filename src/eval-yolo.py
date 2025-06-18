import torch
from ultralytics import YOLO

def main():
    # 继承 YOLO 类
    class ModifiedYOLO(YOLO):
        def __init__(self, model_path):
            super().__init__(model_path)

        @torch.no_grad()
        def postprocess(self, preds, img, orig_imgs):
            # 调用原始的 postprocess 方法获取预测结果
            results = super().postprocess(preds, img, orig_imgs)

            # 修改类别索引，将大于 5 的类别索引改为 5
            for result in results:
                if result.boxes.cls is not None:
                    # 将类别索引转换为 numpy 数组以便修改
                    cls = result.boxes.cls.cpu().numpy()
                    # 将大于 5 的类别索引改为 5
                    cls[cls > 5] = 5
                    # 将修改后的类别索引放回结果中
                    result.boxes.cls = torch.from_numpy(cls).to(result.boxes.cls.device)

            return results


    # 加载自定义的 YOLO 模型
    # model = ModifiedYOLO('YoloV11m_traffic_object_detection_final.pt')
    model = YOLO("yolov8s.pt")
    # 验证模型
    results = model.val(
        data="./yolo11dataset.yaml",  # 数据集配置文件路径
        imgsz=640,  # 图像尺寸
        device="cuda"  # 验证设备，如 "cpu" 或 "cuda"
    )

    print(results)

if __name__ == '__main__':
    main()