# import os
# import json
# from PIL import Image, ImageDraw
#
# def draw_lanemarks(json_folder, image_folder, output_folder):
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历JSON文件夹中的所有文件
#     for json_file in os.listdir(json_folder):
#         if json_file.endswith(".json"):
#             json_path = os.path.join(json_folder, json_file)
#             image_file = json_file.replace(".json", ".jpg")
#             image_path = os.path.join(image_folder, image_file)
#             image_file = image_file.replace(".jpg", ".png")
#             output_image_path = os.path.join(output_folder, image_file)
#
#             # 检查对应的图像文件是否存在
#             if not os.path.exists(image_path):
#                 print(f"Warning: Image file {image_path} not found. Skipping...")
#                 continue
#
#             # 加载JSON数据
#             with open(json_path, "r") as f:
#                 data = json.load(f)
#
#             # 使用图像文件获取图像大小
#             with Image.open(image_path) as img:
#                 image_width, image_height = img.size
#
#             # 创建一个空白的灰度图像，背景为0
#             image = Image.new("L", (image_width, image_height), 0)
#             draw = ImageDraw.Draw(image)
#
#             # 提取车道线数据并绘制
#             for frame in data["frames"]:
#                 for obj in frame["objects"]:
#                     if obj["category"].startswith("lane/"):
#                         points = obj["poly2d"]
#                         x_coords = [point[0] for point in points]
#                         y_coords = [point[1] for point in points]
#                         types = [point[2] for point in points]
#
#                         # 绘制曲线或直线
#                         for i in range(len(points) - 1):
#                             if types[i] == "C" and types[i + 1] == "C":
#                                 # 绘制贝塞尔曲线（简化为直线）
#                                 draw.line((x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1]), fill=255,
#                                           width=5)
#                             else:
#                                 # 绘制直线
#                                 draw.line((x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1]), fill=255,
#                                           width=5)
#
#             # 保存为PNG
#             image.save(output_image_path)
#             print(f"Processed {json_file} and saved to {output_image_path}")
#
# # 示例用法
# json_folder = rf".\bdd100k_labels\val"  # 替换为JSON文件夹路径
# image_folder = rf".\bdd100k_images_100k\val"  # 替换为图像文件夹路径
# output_folder = rf".\mask\val"  # 替换为输出文件夹路径
#
# draw_lanemarks(json_folder, image_folder, output_folder)
import json
import os


def bdd100k_json_to_yolo_txt(categorys, json_file, write_path):
    """
    将BDD100K的JSON标签转换为YOLO格式的TXT文件。
    :param categorys: 需要保留的类别列表
    :param json_file: JSON文件路径
    :param write_path: 输出TXT文件的路径
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_name = data['name']
    txt_content = []

    for frame in data['frames']:
        for obj in frame['objects']:
            if obj['category'] in categorys:
                category_index = categorys.index(obj['category'])
                x1, y1, x2, y2 = obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']
                dw, dh = 1.0 / 1280, 1.0 / 720  # 假设图像宽度为1280，高度为720
                x_center = ((x1 + x2) / 2.0) * dw
                y_center = ((y1 + y2) / 2.0) * dh
                width = (x2 - x1) * dw
                height = (y2 - y1) * dh

                txt_content.append(f"{category_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    if txt_content:
        with open(os.path.join(write_path, f"{image_name}.txt"), 'w') as txt_file:
            txt_file.writelines(txt_content)
        print(f"{image_name}.txt has been created.")
    else:
        print(f"No objects found in {image_name}. No TXT file created.")


if __name__ == "__main__":
    # categories = ['person', 'car', 'truck', 'bike', 'traffic light', 'traffic sign']
    categories = [
        'person', 'bike', 'car', 'motorcycle', 'airplane', 'aa', 'bb', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'traffic sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]
    json_path = rf"D:\BaiduNetdiskDownload\bdd100k_labels\test"  # 替换为JSON文件夹路径
    output_path = rf"D:\BaiduNetdiskDownload\data\labels\test"  # 替换为图像文件夹路径

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for json_file in os.listdir(json_path):
        if json_file.endswith('.json'):
            bdd100k_json_to_yolo_txt(categories, os.path.join(json_path, json_file), output_path)
