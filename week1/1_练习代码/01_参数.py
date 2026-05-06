# 1. *args 和 **kwargs 练习
def order_info(*dishes, **customer_info):
    print(f"顾客信息: {customer_info}")
    print(f"菜品：{','.join(dishes)}")

order_info("宫保鸡丁", "麻婆豆腐", name="张三", total=5)

# 2. 基础装饰器
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'函数 {func.__name__} 执行耗时：{end - start:.2f}秒')
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.5)
    print('任务完成')

slow_function()