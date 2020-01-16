#-*-coding:utf-8-*-
import logging

def getlog(filepath):
    #1.创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #2.创建一个Handler, 写入文件
    fh = logging.FileHandler(filepath, mode='w')
    fh.setLevel(logging.DEBUG)
    #3.创建一个Handler, 输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    #4.定义Handler的格式
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    #5.添加到logger的Handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
