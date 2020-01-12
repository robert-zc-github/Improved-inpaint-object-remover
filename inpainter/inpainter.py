import numpy as np
import matplotlib.pyplot as plt
import time
import os
from skimage.io import imread, imsave
from skimage.color import rgb2grey, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from math import pi


class Inpainter():
    def __init__(self, image, mask, patch_size=9, plot_progress=False):
        self.image = image.astype('uint8')  # 原图像480*360*3
        self.mask = mask.round().astype('uint8')  # 原掩膜480*360*1
        self.patch_size = patch_size  # 默认补丁大小
        self.plot_progress = plot_progress  # 是否显示中间过程

        # Non initialized attributes 未初始化参数
        self.plot_image_path = '../resources/plot_process/test001/image8/'  # 中间文件保存路径
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.H_p = None      #增加的改进项
        self.priority = None

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()  # 确认图像和掩膜大小是否一致
        self._initialize_attributes()  # 初始化所有参数

        start_time = time.time()  # 开始计时
        keep_going = True  # 继续下一步填充标志位
        while keep_going:
            self._find_front()  # 寻找当前边界，边界标位1，其余位置为0　当前边界为self.front
            if self.plot_progress:  # 若显示中间过程选项打开，则输出中间过程图片
                self._plot_image()  # 输出中间过程图片

            self._update_priority()  # 更新优先权函数

            target_pixel = self._find_highest_priority_pixel()  # 寻找边界上优先权最大的像素点
            self.update_patchsize(target_pixel)    # 设置合适的补丁大小
            find_start_time = time.time()  # 开始查找补丁的计时
            source_patch = self._find_source_patch(target_pixel)  # 寻找最佳匹配的补丁
            print('Time to find best: %f seconds'
                  % (time.time() - find_start_time))  # 查找用时

            self._update_image(target_pixel, source_patch)  # 更新原图片，将补丁复制到待修复区域

            keep_going = not self._finished()  # 循环结束条件判断
            if not keep_going:
                if self.plot_progress:  # 若显示中间过程选项打开，则输出中间过程图片
                    self._plot_image()  # 输出中间过程图片

        print("now generate result gif")
        os.system('convert -delay 30 -loop 0 ' + self.plot_image_path + '*.jpg ' + self.plot_image_path + 'result.gif')
        print('Took %f seconds to complete' % (time.time() - start_time))  # 程序运行总时间
        return self.working_image  # 返回修复完的图像

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:  # 验证原图像与掩膜大小是否一致
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.working_mask.shape  # 480*360

        # Remove the target region from the image 去除原图像的掩膜区域
        inverse_mask = 1 - self.working_mask  # 保留区域为１，掩膜区域为０
        rgb_inverse_mask = self._to_rgb(inverse_mask)  # 重建掩膜区域，全填充0
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red 边界填充红色
        image[:, :, 0] += self.front * 255  # 只在Red上填充

        # Fill the inside of the target region with white  当前待修复图像的掩膜区域内为黑色，将其填充为白色
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plot_image_path = self.plot_image_path  # 中间文件保存路径
        remaining = self.working_mask.sum()
        plot_image_path = plot_image_path + str(int(height * width - remaining)) + '.jpg'

        # 文件保存与显示细节
        # plt.clf()
        # plt.imshow(image)
        # plt.savefig(plot_image_path)
        # plt.draw()
        # plt.pause(0.001)  # TODO: check if this is necessary
        imsave(plot_image_path, image, quality=100)

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]  # image.shape[:2] = 【480, 360】

        self.confidence = (1 - self.mask).astype(float)  # 置信度初始化，掩膜区域全填充0，保留区域全填充1
        self.data = np.zeros([height, width])  # 数据项初始化，全区域填充0
        self.H_p = np.zeros([height, width])   #结构张量项初始化
        self.lamda = np.zeros([height, width, 2])
        self.working_image = np.copy(self.image)  # 复制原图像，准备开始填充
        self.working_mask = np.copy(self.mask)  # 复制掩膜，准备开始填充

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.

        使用拉普拉斯算子寻找边界
        拉普拉斯算子会返回掩膜图像的边界，在数据上表现为掩膜区域为正（白），保留区域为负（黑），
        我们只想保留白色区域，所以我们会过滤负数值

        laplace(self.working_mask)返回图像的边界，边界上靠近掩膜区域大于０，靠近保留区域小于０
        >0　则将整幅图片以０划分，边界上靠近掩膜区域为true，靠近保留区域为false，其余全为false
        .astype('int8')将所有的true变为１,本质上减小了边界宽度
        """

        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # TODO: check if scipy's laplace filter is faster than scikit's

    def update_patchsize(self,target_pixel):       # 根据张量结构信息设置填充块的大小
        target_patch = self._get_patch(target_pixel)
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        target_lamda = self._patch_data(self.lamda, target_patch)
        lamda1_avg = sum(sum(target_lamda[:, :, 0] * mask)) / sum(sum((mask == 1)))
        lamda2_avg = sum(sum(target_lamda[:, :, 1] * mask)) / sum(sum((mask == 1)))
        ave = ((lamda1_avg - lamda2_avg)/(lamda1_avg + lamda2_avg))**2
        if ave >= 0.9:
            self.patch_size = 5
        elif ave >= 0.8:
            self.patch_size = 7
        elif ave >= 0.7:
            self.patch_size = 9
        else:
            self.patch_size = 11

    def _update_priority(self):
        self._update_confidence()  # 更新置信度
        gradient = self._update_data()  # 更新数据项
        self._structure_tensor(gradient)
        # self.priority = self.confidence * self.data * self.front
        self.priority = (3/5 * self.confidence + 1/5 * self.data + 1/5 * self.H_p) * self.front
        # self.priority = (2 / 5 * self.confidence + 3 / 5 * self.H_p) * self.front
        # 优先权矩阵，置信度*数据项*边界，且只有边界上才会取值，减少计算量

    def _update_confidence(self):
        # 置信度含义：当前像素对应的补丁内，所有包含在原图像中的像素与其对应的置信度积之和与补丁面积的比值，反应了当前补丁内的有效信息
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)  # 返回边界的位置，使用边界各像素在图像中的位置表示
        for point in front_positions:
            patch = self._get_patch(point)
            # k = self._patch_data(self.confidence, patch)
            # kk = sum(self._patch_data(self.confidence, patch))
            # kkk = sum(sum(self._patch_data(self.confidence, patch)))
            new_confidence[point[0], point[1]] = sum(           # 单个sum只能合并一维的数据
                sum(self._patch_data(self.confidence, patch))   # 返回patch内的信息
            ) / self._patch_area(patch)

        self.confidence = new_confidence  # 更新置信度矩阵

    def _update_data(self):
        # 数据项含义：当前像素对边界的法向量与当前像素位置的等照度线的内积与归一化因子之比，反映当前位置的结构信息
        normal = self._calc_normal_matrix()  # 法向量与归一化因子比      实际计算出来这个是梯度旋转90°，即等照度线
        gradient, max_gradient = self._calc_gradient_matrix()  # 等照度线的垂线       这个才是法向量
        normal_gradient = normal * max_gradient  # 数据项计算公式

        # 原作者代码，这个应该有问题
        # self.data = np.sqrt(normal_gradient[:, :, 0] ** 2 + normal_gradient[:, :, 1] ** 2) + 0.0001
        # To be sure to have a greater than 0 data

        # 根据论文修改的试验代码
        self.data = np.abs(normal_gradient[:, :, 0] + normal_gradient[:, :, 1]) + 0.0001
        # To be sure to have a greater than 0 data
        return gradient
        # TODO：此处的公式是否最优？

    def _calc_normal_matrix(self):  # 计算当前边界上各点对于边界的法向量除以归一化系数
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])  # X方向的边缘算子
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])  # Y方向的边缘算子

        x_normal = convolve(self.working_mask.astype(float), x_kernel)  # 水平梯度卷积
        y_normal = convolve(self.working_mask.astype(float), y_kernel)  # 垂直梯度卷积
        normal = np.dstack((x_normal, y_normal))  # 合并梯度结果

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal ** 2 + x_normal ** 2).reshape(height, width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1  # 防止除以0

        unit_normal = normal / norm  # 数据项计算公式
        return unit_normal
        # TODO：边缘算子取值会对数据项计算有影响吗
        # 会，会改变计算出的方向，若不改变算子的形式则不会

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient

        height, width = self.working_image.shape[:2]  # 480*360
        # 新的梯度计算方法，分别计算RGB的梯度
        # image = np.copy(self.working_image)
        # gradient_r = np.nan_to_num(np.array(np.gradient(image[:, :, 0])))
        # gradient_g = np.nan_to_num(np.array(np.gradient(image[:, :, 1])))
        # gradient_b = np.nan_to_num(np.array(np.gradient(image[:, :, 2])))
        # gradient = np.zeros([2, height, width])
        # gradient[0] = (gradient_r[0] + gradient_g[0] + gradient_b[0])/3
        # gradient[1] = (gradient_r[1] + gradient_g[1] + gradient_b[1])/3

        # 原计算梯度方法
        grey_image = rgb2grey(self.working_image)
        grey_image[self.working_mask == 1] = None  # 计算等 照度线前去除掩膜区域(working_mask==1)的影响

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))  # 获取全图每一个像素点的梯度，掩膜区域用0(或无穷大的任意数字)填充，尺寸为2*height*width（x，y方向）
        gradient_val = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)  # 获取全图每一个像素点的梯度大小
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)  # 返回边界的位置，使用边界各像素在图像中的位置表示
        for point in front_positions:  # 遍历图像边界
            patch = self._get_patch(point)  # 取边界上的各像素对应补丁范围
            patch_y_gradient = self._patch_data(gradient[0], patch)  # 梯度交换  gradient[0] = X轴梯度
            patch_x_gradient = self._patch_data(gradient[1], patch)  # 梯度交换  gradient[1] = Y轴梯度
            patch_gradient_val = self._patch_data(gradient_val, patch)  # 边界上这个像素点对应的补丁内，所有像素点的梯度大小
            patch_max_pos = np.unravel_index(patch_gradient_val.argmax(), patch_gradient_val.shape)  # 查找这个区域内最大梯度
            # 原作者代码，这个应该有问题
            # x和y方向上的最大梯度值也由最大梯度值决定
            # max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]  #
            # max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]  #

            # 根据论文修改的试验版本
            # x和y方向上的最大梯度值也由最大梯度值决定
            max_gradient[point[0], point[1], 0] = patch_x_gradient[patch_max_pos]  #
            # max_gradient[point[0], point[1], 1] = patch_y_gradient[patch_max_pos] * -1  #
            max_gradient[point[0], point[1], 1] = patch_y_gradient[patch_max_pos]

        return gradient, max_gradient

    def _structure_tensor(self, gradient):
        height, width = gradient.shape[1:]

        # 需要单独用到特征值时
        h = np.zeros([height, width])
        ro = 0.2
        for i in range(height):
            for j in range(width):
                str_tensor = np.array([[gradient[0, i, j]*gradient[0, i, j], gradient[0, i, j]*gradient[1, i, j]],
                                       [gradient[0, i, j]*gradient[1, i, j], gradient[1, i, j]*gradient[1, i, j]]])\
                                       / (2*pi*ro**2)*np.exp(-(i**2+j**2)/(2*ro**2))
                eigenvalue, featurevector = np.linalg.eig(str_tensor)
                h[i, j] = (eigenvalue[0] - eigenvalue[1])**2
                self.lamda[i, j, :] = eigenvalue
        self.H_p = 0.8 * h + np.exp(-h)

        # 不需要用到特征值时
        # front_positions = np.argwhere(self.front == 1)
        # for point in front_positions:
        #     i = point[0]
        #     j = point[1]
        #     str_tensor = np.array([[gradient[0, i, j] * gradient[0, i, j], gradient[0, i, j] * gradient[1, i, j]],
        #                           [gradient[0, i, j]*gradient[1, i, j], gradient[1, i, j]*gradient[1, i, j]]])\
        #                           / (2 * pi * ro ** 2) * np.exp(-(i ** 2 + j ** 2) / (2 * ro ** 2))
        #     eigenvalue, featurevector = np.linalg.eig(str_tensor)
        #     h[i, j] = (eigenvalue[0] - eigenvalue[1])**2
        #     self.H_p = 0.8 * h + np.exp(-h)

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        # 返回边界上优先权最大的位置。多维边界使用unravel_index
        return point

    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel)  # 获取待修复区域范围
        height, width = self.working_image.shape[:2]  # 480*360，全图范围查找
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image)
        # 将RGB空间转换为CIELAB色彩空间，由L，a，b三通道表示颜色，为了方便计算颜色差异

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height - 1],
                    [x, x + patch_width - 1]
                ]
                if self._patch_data(self.working_mask, source_patch).sum() != 0:  # 确定此区域中包含边界
                    continue

                # 这样处理的好处：保证待修复区域和寻找的补丁大小一致，而且避免了图像边界的错误
                difference = self._calc_patch_difference(
                    lab_image,  # 经过颜色变换的原图像
                    target_patch,  # 待修复区域
                    source_patch  # 当前寻找的补丁
                )

                if best_match is None or difference < best_match_difference:  # 第一次的问题由best_match is None条件解决
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)  # 获得待修复区域位置
        pixels_positions = np.argwhere(self._patch_data(self.working_mask, target_patch) == 1) \
                           + [target_patch[0][0], target_patch[1][0]]  # 获得待修复区域内边界上的所有像素位置
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]  # 待修复区域中心点置信度
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence  # 待修复区域内边界上所有像素点的置信度统一为区域中心的置信度

        mask = self._patch_data(self.working_mask, target_patch)  # 掩膜图像在待修复区域的局部图像
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.working_image, source_patch)  # 补丁区域图像信息
        target_data = self._patch_data(self.working_image, target_patch)  # 待修复区域图像信息

        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)
        # 新填充区域=     (待修复区域)           +      (原图区域)  完整补丁

        self._copy_to_patch(self.working_image, target_patch, new_data)  # 复制完整补丁到原图的待修复区域
        self._copy_to_patch(self.working_mask, target_patch, 0)  # 复制0到掩膜上的待修复区域，即缩小边界

    def _get_patch(self, point):  # 返回补丁范围，并没有针对规定图像
        half_patch_size = (self.patch_size - 1) // 2  # 地板除，返回比真正的商小的最接近的数字
        height, width = self.working_image.shape[:2]  # 480*360  point[0] = 480,为图像高度，360为图像宽度
        patch = [
            [
                max(0, point[0] - half_patch_size),  # 补丁上边界，与0比较防止越界
                min(point[0] + half_patch_size, height - 1)  # 补丁下边界，与图像的高比较防止越界
            ],
            [
                max(0, point[1] - half_patch_size),  # 补丁左边界，与0比较防止越界
                min(point[1] + half_patch_size, width - 1)  # 补丁右边界，与图像的宽比较防止越界
            ]
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data) ** 2).sum()  # 对应元素差的平方之和
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0]) ** 2 +
            (target_patch[1][0] - source_patch[1][0]) ** 2
        )  # tie-breaker factor 待修复区域和当前补丁原点距离    偏向于选用附近的块修复图像
        height, width = mask.shape
        two_dim_mask = mask.reshape(height, width, 1).repeat(2, axis=2)
        target_lamda = self._patch_data(self.lamda, target_patch) * two_dim_mask
        source_lamda = self._patch_data(self.lamda, source_patch) * two_dim_mask
        lamda_distance = sum(sum(10*(self.moa(target_lamda) - self.moa(source_lamda) ** 2)))
        # lamda_distance = sum(sum((target_lamda[:, :, 0]-source_lamda[:, :, 0])**2 +
        #                          (target_lamda[:, :, 1]-source_lamda[:, :, 1])**2))
        return squared_distance + euclidean_distance + lamda_distance
        # return squared_distance + euclidean_distance
        # return squared_distance  # TODO:针对代码运行结果与论文结果不一致的问题，尝试删去附加项

    def moa(self, lamda):
        moa = (np.nan_to_num((lamda[:, :, 0] - lamda[:, :, 1])/(lamda[:, :, 0] + lamda[:, :, 1])))**2
        return moa

    def _finished(self):
        height, width = self.working_image.shape[:2]  # 480*360
        remaining = self.working_mask.sum()  # 待修复区域总像素数
        total = height * width  # 图像像素总数
        print('%d of %d completed' % (total - remaining, total))
        #return remaining == 0  # 循环结束条件，当remaining==0时，返回true，结束循环
        return remaining <= 10  # 循环结束条件，当remaining<=10时，返回true，结束循环

    @staticmethod
    def _patch_area(patch):  # 返回补丁面积
        return (1 + patch[0][1] - patch[0][0]) * (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_shape(patch):  # 返回补丁大小
        return (1 + patch[0][1] - patch[0][0]), (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_data(source, patch):  # 根据patch位置返回所给定的图像（原图或置信度）的补丁数据
        return source[
               patch[0][0]:patch[0][1] + 1,
               patch[1][0]:patch[1][1] + 1
               ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):  # 根据data，填充dest图像中dest_patch处的值
        dest[
        dest_patch[0][0]:dest_patch[0][1] + 1,
        dest_patch[1][0]:dest_patch[1][1] + 1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape  # Height480 * Width360
        return image.reshape(height, width, 1).repeat(3, axis=2)
