
import signal

# 定义超时异常
class TimeoutException(Exception):
    pass

# 处理超时信号
def handler(signum, frame):
    raise TimeoutException()

# 设置超时时间（秒）
timeout = 30

# 使用装饰器设置超时
def timeout_decorator(func):
    def wrapper(*args, **kwargs):
        # 设置信号处理器
        signal.signal(signal.SIGALRM, handler)
        # 启动闹钟
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
        except TimeoutException:
            print("Function timed out!")
            result = None
        finally:
            # 关闭闹钟
            signal.alarm(0)
        return result
    return wrapper