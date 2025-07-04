# 出题工具

由于需要日常帮忙出题，但是也不需要出一些偏的例子，只需要随机，于是写一个小工具

# 安装

将项目克隆到本地后，需要先安装 [uv](https://docs.astral.sh/uv/)

然后在项目文件夹运行命令：

```bash
uv sync
```

# 出题

在 `src` 文件夹中手动创建文件，或使用 `make init` 来初始化一个文件

可以参考 `src/example.py` 查看使用方法

## 题面

在 `Solution` 的子类中写上文档注释，这个注释使用 `markdown` 格式进行书写

## 题解

在 `Solution` 的子类中实现方法 `solve` 与读取数据的函数 `read_input`，其中：

1. `read_input` 用于读取数据到 `self.input_data` 列表中，列表的每一项都是一份完整的数据，用于后续求解

2. `solve` 会求解 `self.input_data` 中所有的数据，将答案合并为一个符合输出格式的字符串并返回

> [!NOTE]
> 你可以添加任意的辅助函数（注意需要将其写为私有函数的形式）

## 生成器

在 `Generator` 子类中，实现生成的功能 `generate_data` ，注意，题面生成的方法在内部已实现，只需要专注于生成数据即可

此方法综合了所有需要生成的问题

## 测试

我们可以使用单元测试来测试样例是否正确

> [!IMPORTANT]
> 注意，题解的正确性请在 OJ 上或使用更可信的方法进行检查，这里的单元测试只为检查数据的初步正确）


## 运行

只需要激活虚拟环境后，运行创建的文件即可，可以在 `data/` 文件夹下找到生成的题面以及数据