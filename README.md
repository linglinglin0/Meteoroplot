# MeteroPlot
## Overview
MeteroPlot is a Python package for plotting meteorological data. It provides a simple and intuitive interface for creating various types of meteorological plots.
大三上学期天气学课程设计，基于Python中cartopy库，nc4def库，使用面向对象方法所编写的气象数据可视化脚本。

## 数据解释
| 文件名   | 物理量    |
| ----- | ------ |
| air   | 气温     |
| hgt   | 位势高度   |
| omega | 垂直速度   |
| slp   | 水平面气压  |
| uwnd  | 纬向速度分量 |
| vwnd  | 经向速度分量 |
- 文件名中的mon即monthly-average 代表着数据为每月的平均值
- 文件名中的ltm即long-term mean 代表着数据是一段时间内的平均值，而不是某个特定月份的数据
- 以air.mon.ltm.1991-2020.nc为例，其中储存着1月-12月的月平均气温数据，每一个月平均气温数据代表着1991-2020年，30年气温数据的平均态。
