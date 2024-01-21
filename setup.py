# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


# 定义一个名为软件包，基本信息包括名称、版本号、依赖关系、在安装和分发时使用
setup(
    name="llama",
    version="0.0.1",
    # 自动查找并包括所有子包
    packages=find_packages(),
    # 依赖关系，将在安装时自动解决
    install_requires=get_requirements("requirements.txt"),
)
