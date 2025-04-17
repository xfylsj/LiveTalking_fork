
from time import sleep


def test_simple_yield():
    """测试基本的yield生成器功能"""
    def number_generator():
        for i in range(5):
            yield i

    gen = number_generator()
    assert list(gen) == [0, 1, 2, 3, 4]

def test_yield_send():
    """测试yield的send功能"""
    def echo_generator():
        value = yield "准备接收"
        while True:
            value = yield f"收到: {value}"
    
    gen = echo_generator()
    assert next(gen) == "准备接收"
    assert gen.send("你好") == "收到: 你好"
    assert gen.send("世界") == "收到: 世界"

def test_yield_from():
    """测试yield from功能"""
    def sub_generator():
        yield 1
        yield 2
        yield 3

    def main_generator():
        yield "开始"
        yield from sub_generator()
        yield "结束"

    gen = main_generator()
    assert list(gen) == ["开始", 1, 2, 3, "结束"]

def test_yield_exception():
    """测试yield异常处理"""
    def generator_with_exception():
        try:
            yield "正常"
            raise ValueError("发生错误")
        except ValueError as e:
            yield f"捕获异常: {str(e)}"

    gen = generator_with_exception()
    assert next(gen) == "正常"
    assert next(gen) == "捕获异常: 发生错误"

def test_print_yield():
    """测试基本的yield生成器功能"""
    def number_generator():
        for i in range(5):
            print(f"yield {i}")
            yield i
            sleep(1)

    gen = number_generator()
    print(f"gen: {gen}")
    print(f"gen list: {list(gen)}")
    # assert list(gen) == [0, 1, 2, 3, 4]
    print(f"gen: {gen}")
    for i in gen:
        print(f"i: {i}")

if __name__ == "__main__":
    test_simple_yield()
    test_yield_send()
    test_yield_from()
    test_yield_exception()
    # print("所有测试通过!")
    test_print_yield()
