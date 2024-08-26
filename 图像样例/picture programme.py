# 霖澪霖
# 开发时间：
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
# from mpl_toolkits.basemap import Basemap
import os
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from PIL import Image
from decimal import Decimal
from tqdm import trange
from time import sleep
import imageio
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cmaps
from scipy.interpolate import griddata
from matplotlib import axes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# 准备设置
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams["font.family"] = 'Arial'  #默认字体类型
plt.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
plt.rcParams["axes.linewidth"] = 1
'''
存在的问题
1.公用模块中的参数需要反复修改，不能完美契合，具体而言
    ①colorbar需要反复修改，包括选取的色盘和截取的颜色，其实这个没啥办法，因为是要具体数据具体分析，但是可以整理出常用变量比较合适的色盘
    ②标题文字，这个其实可以放到axes里面，在初始化的时候传进去，比如月份和高度，但是月份和高度都是可有可无的，没传的时候就是''
2.流线断裂
3.等高线标签垃圾
'''

class Axes :
    '''
    底图类
    拥有extent属性为绘图的经纬度范围,extent = [leftlon,rightlon,lowerlat,upperlat]
    月份属性
    '''

    def __init__(self,level=None,extent = [-180, 180, -90, 90]):
        self.extent = extent
        # self.month = month
        self.level = level
    def normal(self,layout,fig):#fig是画布
        '''
        使用等距圆柱投影，绘制正常经纬度底图
        layout为绘图布局，可以选择单独绘制一张图也可以多张图片布局在一个figure上，具体布局规则参考subplot的使用
        '''
        ax = fig.add_subplot(layout, projection=ccrs.PlateCarree(central_longitude=(self.extent[1]-self.extent[0])/2)) #layout是布局变量按照111,211的格式输入，projection是投影方式，在此选择的是等距圆柱投影
        ax.set_extent(self.extent, crs=ccrs.PlateCarree()) # 设置地图范围
        ax.coastlines()  # 画海岸线
        # ax.add_feature(cfeature.LAND) # 填充陆地颜色
        #gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle=':',
                      #    x_inline=False)#画出网格
        ax.set_xticks(np.arange(self.extent[0], self.extent[1]+3, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.extent[2], self.extent[3]+1, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        return ax

    def lambert(self,layout,fig):
        proj = ccrs.LambertConformal(central_longitude = 125, central_latitude = 20,  standard_parallels = (0, 40))
        ax = fig.add_subplot(layout, projection=proj)
        gl1 = ax.gridlines(draw_labels = True, x_inline = False, y_inline = False)  # 添加栅格线，绘制刻度标签，但禁止在栅格线内绘制标签
        ax.coastlines(resolution='110m')
        ax.add_feature(cfeature.LAND,color = [0.5,0.5,0.5])#要调整的地方
        gl1.rotate_labels = False  # 禁止刻度标签旋转
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        return ax

    def lat_level(self,layout,fig):
        '''
        纬高图
        :param layout:
        :return:
        '''
        # fig = plt.figure(figsize=(11, 12))
        ax = fig.add_subplot(layout)#layout是布局变量按照111,211的格式输入，projection是投影方式，在此选择的是等距圆柱投影
        ax.set_yscale('symlog')
        ax.invert_yaxis()
        ax.set_xticks([90, 60, 30, 0, -30, -60, -90])
        ax.set_xticklabels([r'90$^\degree$', r'60$^\degree$', r'30$^\degree$N', r'EQ',
                                r'30$^\degree$S', r'60$^\degree$', r'90$^\degree$'])
        ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 60])
        ax.set_yticklabels([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, "hPa"])
        return ax

    def northpolar(self,layout,fig):
        '''
        北半球投影图
        :param layout:
        :return:
        '''
        ax = fig.add_subplot(layout, projection=ccrs.NorthPolarStereo(central_longitude=180))
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=1, color='grey', linestyle='--')
        ax.set_extent([-180,180,0,90], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        return ax

    def plt_val(self,fig,ax,X,Y,val,level = 1000,hgt = 0,feature = None):#给level设置一个默认值，其实我是想直接根据输入的val值计算一个数组，然后以他为默认值，但是我也保留了输入等值线根数和等值线范围的接口。
        '''
        :param ax:
        :param X:
        :param Y:
        :param val:
        :param level: 如果不输入，默认根据val的值以5或10为间隔画等值线图,也可以输入条数和等值线范围
        :return:
        '''
        flag = 0 #无输入还是输入数值（1），还是数组（2）
        # if level == 1000: flag = 1#无输入时自动设置等高线的范围
        if hgt == 1 : val = val/10
        if flag: level,level_clable = Data.cal_contourlevels(val)
        if feature == "hgt_588":
            c3 = ax.contour(X, Y, val , levels=[588], colors='k',
                                transform=ccrs.PlateCarree(), linewidths=1,)  # linestyles='--'
            ax.clabel(c3,  inline=True, fontsize=10)  # 显示等值线上数值
            return
        c2 = ax.contour(X, Y, val , levels= level,extent = "both", colors='k',
                            transform=ccrs.PlateCarree(), linewidths=1,
                            )  # linestyles='--'
        if flag:
            ax.clabel(c2, levels =level_clable , inline=True, fontsize=10)  # 显示等值线上数值
        else:
            ax.clabel(c2, inline=True, fontsize=10)
        return ax

    def add_title(self,value,path = "",month = '',level= ''):
        if value == 'slp':
            plt.title(str(month) + "月平均海平面气压场(hPa)", fontsize=18)  # 添加地图标题
            plt.savefig(path+"海平面气压分布(hPa)" + ".png", dpi=600)  # 导出图片
        elif value == 'hgt':
            plt.title(str(month) + "月平均" + str(self.level) + "hpa高度场（单位：dagpm）", fontsize=18)  # 添加地图标题
            plt.savefig(path+str(month) + "月平均" + str(self.level) + "hpa高度场（单位：dagpm）", dpi=600)
        elif value == "air":
            plt.title(str(month) + "月平均" + str(self.level) + "hpa温度场", fontsize=18)  # 添加地图标题
            # plt.savefig(path + str(month) + "月平均" + str(self.level) + "hpa温度场")
        elif value == "hgt588":
            plt.title(str(month) + "月平均" + str(500) + "hpa高度场588位势什米线（单位：dagpm）", fontsize=18)  # 添加地图标题
            plt.savefig(path + str(month) + "月平均" + str(500) + "hpa高度场588位势什米线（单位：dagpm）", dpi=600)
        elif value =='hgt_northpolar':
            plt.title('北半球'+str(month)+'月平均'+str(200)+'hpa高度场')
        elif value == "hgt588continue":
            plt.title("500hpa西太平洋副高脊的月平均位置", fontsize = 18)
            # plt.savefig("500hpa西太平洋副高脊的月平均位置",dpi = 600)
        elif value == 'wind_global':
            plt.title(str(month)  + '月'+str(level) + "hp平均风场", fontsize=18)  # 添加地图标题
            # plt.savefig(path + str(self.level) + "hp平均风场", dpi=600)
        elif value == 'wind_northpolar':
            plt.title(str(month)+'月'+ str(level)+"hp北半球平均西风分布（单位m/s）")
        elif value == 'wind_latitude':
            plt.title(str(month) +'月沿纬圈平均的平均纬向风速的经向剖面图',fontsize = 18)
        elif value =='meridional circulation':
            plt.title(str(month) + '月北半球平均经圈环流', fontsize=18)
        elif value == "demo":
            plt.show()

    @classmethod
    def colorbar(self,fig,ax,X,Y,val,level=30):
        # cmap = plt.cm.RdYlBu
        cmap = cmaps.ncl_default
        newcolors = cmap(np.linspace(0, 1, 256))
        newcmp = ListedColormap(newcolors[1:256])
        c1 = ax.contourf(X, Y, val, zorder=0, levels=level,
                         extend='both', transform=ccrs.PlateCarree(), cmap=newcmp)
        cbar = fig.colorbar(c1, orientation='horizontal', shrink=0.9,pad = 0.08,extend = False)
        cbar.set_label("位势高度（dagpm）", fontsize=13)

class Data:
    '''
    数据处理类
    '''
    def __init__(self,address):
        '''
        拥有地址属性
        经度属性
        维度属性
        值属性
        高度属性
        '''
        self.address = address
        self.val = nc.Dataset(self.address)
        self.lons =[]
        self.lats =[]
        self.data_value =[]
        self.levels =[]
        pass

    def print_data(self):
        '''
        打印数据的相关信息
        :param val: 需要读取的变量的名称
        :return: 打印输出变量的相关信息，包括名称，维度，数值等
                 返回变量数组,以待后续的打印查看
        '''
        print(self.val.variables.keys())
        for var in self.val.variables.keys():
            print(self.val.variables[var])
            data = self.val.variables[var][:].data
            print(var, data.shape)
            print(data[:17])
        return data

    def read_data(self, value):
        '''
        读取数据
        :param value:
        :return:
        '''
        self.lons = self.val["lon"][:]
        self.lats = self.val["lat"][:]
        self.data_value =self.val[value][:]
        try:#如果没有高度的维度直接pass
            self.levels = self.val["level"][:]
        except:
            pass

    def lons_average(self):
        self.data_value = np.array(self.data_value).mean(3)

    def time_average(self):
        self.data_value = np.array(self.data_value).mean(0)
    def time_levelaverage(self,level):
        levels = [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100.,
                  70., 50., 30., 20., 10.]
        level_index = levels.index(level)
        self.data_value = self.data_value[5:7].mean(0)[level_index]
        # ls = []
        # for i in (11,0,1):
        #     ls.append(self.data_value[i])
        # self.data_value = np.array(ls).mean(0)[level_index]
    def clear_centralline(self,flag):#消除中心白线
        if flag == "weigao":
            self.data_value, cycle_lats = add_cyclic_point(np.array(self.data_value), coord=self.lats)
            self.lats, self.levels = np.meshgrid(cycle_lats, self.levels)
        else:
            self.data_value, cycle_lons = add_cyclic_point(np.array(self.data_value), coord=self.lons)
            self.lons,self.lats  = np.meshgrid(cycle_lons, self.lats)

    def select_suitable(self,month = 0,level=0,style=''):
        if style == 'weigao':
            self.data_value = self.data_value[month-1]
            return
        levels = [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100.,
                  70., 50., 30., 20., 10.]
        level_index = levels.index(level)
        if style == 'monthly priority':
            self.data_value = self.data_value[5:7].mean(0)[level_index]
        #     ls = []
        #     for i in (11,0,1):
        #         ls.append(self.data_value[i])
        #     self.data_value = np.array(ls).mean(0)[level_index]
        #     print(self.data_value.shape)
        else:
            self.data_value = self.data_value[month -1][level_index]#数组的index是从0开始的

    @staticmethod
    def cal_contourlevels(val):
        val_max = Data.rounding(np.max(val))
        val_min = Data.rounding(np.min(val))
        levels = np.arange(val_min,val_max,5)
        levels_clabel = np.arange(val_min,val_max,10)
        if(len(levels)>26):#我数了一下书上的图，大概每张图上的等值线数量不超过这个值，如果超过了则是太过稠密
            levels = np.arange(val_min,val_max,10)
            levels_clabel = np.arange(val_min,val_max,20)
        return levels,levels_clabel

    @staticmethod
    def rounding(x):
        x = str(x)
        x = Decimal(x).quantize(Decimal("1."),rounding = "ROUND_HALF_UP")
        return x


class Pic(Axes,Data):
    def __init__(self,address):
        Data.__init__(self,address)
        Axes.__init__(self)
    def interpolation(self):
        plan_level = np.arange(100, 1000.1, 50)
        X, Y = np.meshgrid(-self.lats, plan_level)
        data = self.data_value.flatten()
        x_star = np.hstack((-self.lats.flatten()[:, None], self.levels.flatten()[:, None]))
        res = griddata(x_star, data, (X, Y), method='nearest')
        return res,X,Y

    def colorbar(self,fig,ax,X,Y,val,level=10):
        cmap = cmaps.GMT_haxby  #cmap = cmaps.ncl_default CBR_coldhot CBR_coldhot GMT_haxby
        newcolors = cmap(np.linspace(0,1,23))#cmap(np.linspace(0,1,256)) cmap(np.linspace(0, 1, 8))
        newcmp = ListedColormap(newcolors[3:])
        c1 = ax.contourf(X, Y, val, zorder=0, levels=np.arange(-80,30,5),
                         extend='both',  cmap=newcmp)#当是北极投影时，transform=ccrs.PlateCarree()
        cbar = fig.colorbar(c1, orientation='horizontal', shrink=0.5, pad=0.1, extend=False)#ticks = [-3,-2,-1,0,]
        cbar.set_label("温度（℃）", fontsize=15)

    def plt_temp(self,fig,layout,month,level):
        self.read_data('air')
        self.lons_average()
        self.select_suitable(month,style = 'weigao')
        ax = self.lat_level(fig,layout)
        self.colorbar(fig, ax, -self.lats, self.levels, self.data_value, level=level)
        c = ax.contour(-self.lats, self.levels, self.data_value, levels=level, colors='k',
                       linewidths=1)
        ax.clabel(c, inline=True, fontsize=10)

    def plt_westwind_northpolar(self,fig,layout,month,level):
        self.read_data('uwnd')
        self.select_suitable(month,level=500)
        self.clear_centralline(0)
        ax = self.northpolar(fig,layout)
        self.colorbar(fig,ax,self.lons,self.lats,self.data_value,level = level)
        c1 = ax.contour(self.lons, self.lats, self.data_value, zorder=0, levels=np.arange(-20, 40, 5), linewidths=0.9,
                            extend='both', transform=ccrs.PlateCarree(), colors='k')
        # c2 = ax.contourf(self.lons, self.lats, self.data_value, zorder=0, levels=np.arange(-20, 40, 5),
        #                extend='both', transform=ccrs.PlateCarree())
        ax.clabel(c1,inline = True,fontsize=10)

    def plt_weigaowind(self,fig,layout,month,level):
        self.read_data('uwnd')
        self.lons_average()
        self.select_suitable(month,style = 'weigao')
        self.clear_centralline('weigao')
        ax = self.lat_level(layout,fig)
        c= ax.contour(self.lats,self.levels,self.data_value,levels = level, colors='k',
                    linewidths=1 )
        ax.clabel(c,inline = True,fontsize=10)
        self.colorbar(fig,ax,self.lats,self.levels,self.data_value,level = level)

    def plt_hgt_northpolar(self,fig,layout,month,level):
        self.read_data('hgt')
        self.select_suitable(month,level = 200)
        self.clear_centralline(0)
        ax = self.northpolar(layout,fig)
        self.colorbar(fig,ax,self.lons,self.lats,self.data_value,level = level)
        c1 = ax.contour(self.lons, self.lats, self.data_value/10, zorder=0, levels=level, linewidths=0.9,
                        extend='both', transform=ccrs.PlateCarree(), colors='k')
        ax.clabel(c1, inline=True, fontsize=10,levels = np.arange(1100,1300,20))

    def lat_level(self,fig,layout):
        ax = fig.add_subplot(layout)  # layout是布局变量按照111,211的格式输入，projection是投影方式，在此选择的是等距圆柱投影
        ax.set_yscale('log')
        ax.invert_yaxis()
        # ax.set_xticks([90, 60, 30, 0, -30, -60, -90])
        ax.set_xticks(np.arange(-90,90.1,30))
        ax.xaxis.set_major_formatter(LatitudeFormatter())
        ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200,100,0])
        ax.set_yticklabels([1000, 900, 800, 700, 600, 500, 400, 300, 200,100,''])
        # ax.set_aspect(55)#调整长款比例，高/宽
        return ax

    def plt_longitude_circulation(self,month,value,style = 'Global'):
        plan_level = np.arange(100, 1000.1, 100)#原来高度之间不等距，利用插值处理，这是从100开始，到1000结束，以50为间隔的等距纵坐标
        self.read_data(value)#读入数据
        X, Y = np.meshgrid(-self.lats[24:61], plan_level)#生成插值后的二维网格备用
        if style == 'Global':
            self.lons_average()#关于经度平均
        elif style == 'Partial Area':
            self.partial_longitude_average()
        self.select_suitable(month,style='weigao')#根据月份找到纬高图合适的数据
        self.clear_centralline('weigao')
        # self.lats,self.levels = np.meshgrid(self.lats,self.levels)#生成插值前的二维网格，作用为与传入的data对应
        coordinate = np.hstack((self.lats.flatten()[:,None],self.levels.flatten()[:,None]))#hstack是将两个数组水平拼接，比如[[1,3][2,4]]和[[5,7][6,8]]
        data = self.data_value.flatten()
        #拼接起来即为[[1,3,5,7][2,4,6,8]]
        #flatten的作用为将数组降维，在降维之后在此变为一维数组，然后[:None]的作用是可以将[1,2,3,4,5]变为[[1][2][3][4]]变为一个(N,2)的二维数组，N是原来一维数组中的元素个数
        #最终的效果就是生成了一个储存了对应data值坐标位置的二维数组[[x1,y1][x2,y2]]
        res = griddata(coordinate,data,(X,Y),method = 'nearest')
        return X,Y,res

    def partial_longitude_average(self):
        self.data_value = self.data_value[:,:,:,30:47]
        self.data_value = self.data_value.mean(3)

    def plt_slp(self):
        fig = plt.figure(figsize=(12,11))
        for i in (1,7):
            self.read_data('slp')
            self.select_suitable(i,style='weigao')
            self.clear_centralline(0)
            g = 211
            if i == 7: g =212
            ax = self.normal(g,fig)

            c1 = ax.contour(self.lons, self.lats, self.data_value, levels=np.arange(980,1070,5), colors='k',
                            transform=ccrs.PlateCarree(), linewidths=0.8, )
            c2 = ax.clabel(c1, inline=False, fontsize=6,levels=np.arange(980,1060,10))  # 显示等值线上数值
            self.colorbar(fig,ax,self.lons,self.lats,self.data_value,level = np.arange(980,1070,10))
            self.add_title("hgt588", month=i)
        plt.savefig('4-1-5 平均海平面气压场（单位：hpa）')


def plt_hgt(month,level,data_address,save_address='',style = None,hgt = 1):

    if isinstance(month,list) == False:
        fig = plt.figure(figsize=(12,11))
        hgt = Data(data_address)
        ax1 = Axes(level = 500)
        hgt.read_data(data_address[0:3])
        # hgt.select_suitable(month, level)#单独一个月
        hgt.time_levelaverage(500)#几个月平均
        hgt.clear_centralline(0)
        c = ax1.plt_val(fig,ax1.normal(111,fig), hgt.lons, hgt.lats, hgt.data_value/10, hgt=hgt, level=30)
        ax1.colorbar(fig,c, hgt.lons, hgt.lats, hgt.data_value/10,level=30)
        if style == "demo":
            ax1.add_title('demo')
        else:
            ax1.add_title(data_address[0:3],save_address,month = month)
    elif isinstance(month,list) ==True:
        start = month[0]
        end = month[1]+1
        for i in trange(start, end):#trange是循环中的计时模块
            fig = plt.figure(figsize=(12, 11))
            hgt = Data(data_address)
            ax1 = Axes(month, level)
            hgt.read_data(data_address[0:3])
            hgt.select_suitable(i, level)
            hgt.clear_centralline(0)
            c = ax1.plt_val(fig,ax1.normal(111,fig), hgt.lons, hgt.lats, hgt.data_value, hgt=hgt, level=30)
            ax1.colorbar(c, fig, hgt.lons, hgt.lats, hgt.data_value, level=30)
            if style == 'demo':
                ax1.add_title('demo')
            else:
                ax1.add_title(data_address[0:3], save_address,month = i)

def plt_hgt588_continuous(data_address,save_address="",style = None):#一张图上会同时画很多个月份的588等高线
    hgt = Data(data_address)
    ax1 = Axes(1, 500, extent=[100, 150, 0, 40])
    ax2 = Axes(1, 500, extent=[100, 150, 0, 40])
    fig = plt.figure(figsize = (11,12))
    ax = ax1.lambert(211,fig)
    for i in trange(5, 9):
        hgt.read_data("hgt")
        hgt.select_suitable(i, 500)
        hgt.clear_centralline(0)
        hgt.data_value /= 10
        c3 = ax.contour(hgt.lons, hgt.lats, hgt.data_value, levels=[588], colors='k',
                        transform=ccrs.PlateCarree(), linewidths=2, )  # linestyles='--'
        c4 = ax.clabel(c3, inline=True, fontsize=10)  # 显示等值线上数值
    # ax1.add_title("hgt588",path =save_address,month = i)
    ax1.add_title("hgt588continue")
    bx  = ax2.lambert(212,fig)
    for i in range(8,11):
        hgt.read_data("hgt")
        hgt.select_suitable(i, 500)
        hgt.clear_centralline(0)
        hgt.data_value /= 10
        c3 = bx.contour(hgt.lons, hgt.lats, hgt.data_value, levels=[588], colors='k',
                        transform=ccrs.PlateCarree(), linewidths=2, )  # linestyles='--'
        c4 = bx.clabel(c3, inline=True, fontsize=10)  # 显示等值线上数值
    ax1.add_title("hgt588continue")
    plt.savefig("500hpa西太平洋副高脊的月平均位置", dpi=600)

def plt_hgt588_discontinuity(data_address,save_address):
    hgt = Data(data_address)
    ax1 = Axes(1, 500, extent=[100, 150, 0, 40])

    for i in trange(1, 13):
        fig = plt.figure(figsize=(11, 12))
        hgt.read_data("hgt")
        hgt.select_suitable(i, 500)
        #hgt.clear_centralline(0)
        hgt.data_value /= 10
        ax = ax1.lambert(111,fig)
        c3 = ax.contour(hgt.lons, hgt.lats, hgt.data_value, levels =np.arange(588,598,10) ,colors='k',
                        transform=ccrs.PlateCarree(), linewidths=2, )  # linestyles='--'
       # c4 = ax.clabel(c3, inline=True, fontsize=10)  # 显示等值线上数值
        ax1.add_title("hgt588", path=save_address, month=i)

def plt_globalwind(data_address1,data_address2):
    wind1 = Data(data_address1)
    wind2 = Data(data_address2)
    ax1 = Axes(extent=[-180,180,-60,60])
    fig = plt.figure(figsize=(10, 8))
    for i in (1,7):
        wind1.read_data("uwnd")
        wind2.read_data("vwnd")
        wind1.select_suitable(i,1000)
        wind2.select_suitable(i, 1000)
        wind1.clear_centralline(0)
        wind2.clear_centralline(0)
        if i == 1: g =211
        else: g= 212
        ax = ax1.normal(g,fig)
        c1 = ax.streamplot( wind1.lons,wind1.lats, wind1.data_value, wind2.data_value, density=[2, 1], transform=ccrs.PlateCarree(),linewidth = 0.8,color = 'k')
        #c2= ax.contourf(wind1.lons,wind1.lats, wind1.data_value, levels=[ -100,0], extent = 'both',colors = 'xkcd:light brown', zorder=0
                   #    , transform=ccrs.PlateCarree())
        ax1.colorbar(fig,ax,wind1.lons,wind1.lats, wind1.data_value)
        ax1.add_title('wind_global',month = i,level = 1000)
    plt.savefig("1000hp平均风场", dpi=600)
def plt_globalwind_monthaverage(data_address1,data_address2):
    wind1 = Data(data_address1)
    wind2 = Data(data_address2)
    ax1 = Axes(extent=[-180, 180, -60, 60])
    fig = plt.figure(figsize=(10, 8))
    wind1.read_data("uwnd")
    wind2.read_data("vwnd")
    wind1.select_suitable(level = 925,style = 'monthly priority')
    wind2.select_suitable(level = 925,style = 'monthly priority')
    wind1.clear_centralline(0)
    wind2.clear_centralline(0)
    ax = ax1.normal(111,fig)
    c1 = ax.streamplot(wind1.lons, wind1.lats, wind1.data_value, wind2.data_value, density=[2, 1],
                       transform=ccrs.PlateCarree(), linewidth=0.8, color='k')
    c2= ax.contourf(wind1.lons,wind1.lats, wind1.data_value, levels=[ -100,0], extent = 'both',colors = 'xkcd:light brown', zorder=0
       , transform=ccrs.PlateCarree())
    ax1.colorbar(fig, ax, wind1.lons, wind1.lats, wind1.data_value)
    ax1.add_title('wind_global', month='6-8', level=925)
    plt.savefig('6-8月925hpa平均流畅')
    # plt.show()
def create_gif(image_list, gif_name, fps = 1):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name,frames,'GIF',  fps = fps)
    return

def print_gif(path,gif_name,fps = 0.7):#  path = "D:\\pythonProject1\\data visualization\\meterological plot\\多年平均的月平均100hpa高度场逐月变化\\"
    img_list = []
    for i in trange(1, 13):
        img_list.append(path + str(i) + "月平均" + str(300) + "hpa温度场" + ".png")
    #gif_name = 'D:\\pythonProject1\\data visualization\\meterological plot\\多年平均的月平均100hpa高度场逐月变化\\多年平均的月平均100hpa高度场逐月变化.gif'
    create_gif(img_list, gif_name, fps)


def plt_meridional_circulation(pic1,pic2,fig,layout,month,style = 'Global'):
    pic1.read_data('omega')
    X, Y, res_omega = pic1.plt_longitude_circulation(month, 'omega',style=style)

    res_vwnd = pic2.plt_longitude_circulation(month, 'vwnd',style=style)[2]
    ax = pic1.lat_level(fig, layout)
    plt.ylim(1000, 100)
    c1 = ax.streamplot(X, Y, res_vwnd / 10, res_omega * 100, density=[1.3, 1], linewidth=1.8, color='k', arrowstyle='->',
                       minlength=0.15,broken_streamlines=True)  # density第一个参数是经度方向流线密度，第二个参数为纬度方向流线密度
    c2 = ax.quiver(X, Y, res_vwnd / 10, res_omega * -100, scale=150, pivot='mid', headwidth=2, headlength=4,
                   color=[0.5, 0.5, 0.5])
    ax.quiverkey(c2, 0.98, -0.2, U=10, label='10m/s', color='k')
    pic1.colorbar(fig, ax, X, Y, res_omega*100)
    pic1.add_title('meridional circulation',month = month)

def plt_411():
    pic1 = Pic('air.mon.ltm.1991-2020.nc')
    fig = plt.figure(figsize=(10, 20))
    g = 211
    for i in (1,7):
        if i == 7: g = 212
        pic1.plt_temp(fig,g,i,np.arange(-80,30,5))
        pic1.add_title('air',month = i)
    plt.show()

def plt_413():
    pic1 = Pic('hgt.mon.ltm.1991-2020.nc')
    fig = plt.figure(figsize=(12,11))
    g = 121
    for i in (1,7):
        if i == 7: g = 122
        pic1.plt_hgt_northpolar(fig,g,i,np.arange(1100,1300,10))
        pic1.add_title('hgt_northpolar',month = i)
    plt.savefig('4-1-3北半球平均200hpa高度长（填色）')

def plt_414():
    pic1 = Pic('uwnd.mon.ltm.1991-2020.nc')
    fig = plt.figure(figsize = (12,11))
    g = 121
    for i in (1,7):
        if i == 7: g = 122
        pic1.plt_westwind_northpolar(fig,g,i,20)
        pic1.add_title('wind_northpolar',month = i)
    plt.savefig('4-1-4北半球平均500hpa西风风速分布（填色）')

def plt_415():
    pic1 = Pic('slp.mon.ltm.1991-2020.nc')
    pic1.plt_slp()

def plt_416():
    plt_globalwind("uwnd.mon.ltm.1991-2020.nc","vwnd.mon.ltm.1991-2020.nc")

def plt_417():
    pic1 = Pic("uwnd.mon.ltm.1991-2020.nc")
    fig = plt.figure(figsize=(11, 15))
    g = 211
    for i in (1,7):
        if i == 7:
            g = 212
        pic1.plt_weigaowind(fig,g,i,20)
        pic1.add_title('wind_latitude',month = i)
    plt.savefig( "4-1-7沿纬圈平均的平均纬向风速的经向剖面图", dpi=600)

def plt_419():
    pic1 = Pic('omega.mon.ltm.1991-2020.nc')
    pic2 = Pic('vwnd.mon.ltm.1991-2020.nc')
    fig = plt.figure(figsize=(12, 14))
    plt_meridional_circulation(pic1, pic2, fig, 211, 1, style='Global')
    plt_meridional_circulation(pic1, pic2, fig, 212, 7, style='Global')
    plt.savefig('北半球平均经圈环流')

def plt_4110():
    pic1 = Pic('omega.mon.ltm.1991-2020.nc')
    pic2 = Pic('vwnd.mon.ltm.1991-2020.nc')
    fig = plt.figure(figsize=(12,14))
    plt_meridional_circulation(pic1,pic2,fig,211,1,style='Partial Area')
    plt_meridional_circulation(pic1,pic2,fig,212,7,style='Partial Area')
    plt.savefig('75°-110°E的平均经圈环流')
if __name__ == "__main__":
    # plt_globalwind_monthaverage("uwnd.mon.ltm.1991-2020.nc","vwnd.mon.ltm.1991-2020.nc")
    plt_hgt('6-8',500,'hgt.mon.ltm.1991-2020.nc',style='dem')





