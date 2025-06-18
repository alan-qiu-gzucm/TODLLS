import pyttsx3
import time
import os
import argparse


def text_to_speech(text, rate=200, volume=1.0, voice_id=None):
    """将文本转换为语音并播放"""
    engine = pyttsx3.init()

    # 设置语速
    engine.setProperty('rate', rate)

    # 设置音量 (0.0 到 1.0)
    engine.setProperty('volume', volume)

    # 设置语音 (如果提供了voice_id)
    if voice_id:
        engine.setProperty('voice', voice_id)

    engine.say(text)
    engine.runAndWait()


def get_last_line(file_path):
    """获取文件的最后一行内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if lines:
                return lines[-1].strip()
            return ""
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return ""


def monitor_file(file_path, check_interval=1.0, rate=200, volume=1.0, voice_id=None):
    """监控文件变化并转换最后一行为语音"""
    last_modified = 0
    last_line = ""

    print(f"开始监控文件: {file_path}")
    print(f"检查间隔: {check_interval}秒")
    print(f"语速: {rate}")
    print(f"音量: {volume}")
    if voice_id:
        print(f"语音ID: {voice_id}")

    while True:
        try:
            # 获取文件的最后修改时间
            current_modified = os.path.getmtime(file_path)

            # 检查文件是否被修改
            if current_modified > last_modified:
                last_modified = current_modified
                new_last_line = get_last_line(file_path)

                # 如果最后一行与之前不同，则转换为语音
                if new_last_line != last_line:
                    last_line = new_last_line
                    print(f"检测到文件更新，转换最后一行为语音: {last_line}")
                    text_to_speech(new_last_line, rate, volume, voice_id)

            # 等待指定时间后再次检查
            time.sleep(check_interval)

        except Exception as e:
            print(f"监控文件时出错: {e}")
            time.sleep(check_interval)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='监控文本文件并将其内容转换为语音')

    # 添加命令行参数
    parser.add_argument('file', help='要监控的文本文件路径')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='检查文件更新的时间间隔(秒)，默认值: 1.0')
    parser.add_argument('-r', '--rate', type=int, default=200, help='语音语速，默认值: 200')
    parser.add_argument('-v', '--volume', type=float, default=1.0, help='语音音量(0.0-1.0)，默认值: 1.0')
    parser.add_argument('--voice', help='语音ID，使用 pyttsx3.engine.getProperty("voices") 查看可用语音')

    # 解析命令行参数
    args = parser.parse_args()

    try:
        monitor_file(args.file, args.interval, args.rate, args.volume, args.voice)
    except KeyboardInterrupt:
        print("\n程序已停止")


if __name__ == "__main__":
    main()