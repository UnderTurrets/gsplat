# examples/simple_trainer.py
## Config
加载各种参数


## create_splats_with_optimizers
初始化高斯点云和需要优化的参数，包含`means3d`,`scales`,`quats`,`opacities`。此外，可以选择球谐函数或者特征向量之一的方式表示颜色。若使用球谐函数，则添加优化参数`sh0`,`shN`；若使用特征向量，则添加优化参数`features`,`colors`。

返回值：
`splats`：torch.nn.ParameterDict，包含所有优化参数的一个字典
`optimizers`：优化器列表，每个优化参数对应一个优化器。

##  Runner
### rasterize_splats

先调整颜色表示方式(SH or appearance optimization)，再调整渲染模式(antialiased抗锯齿 or classic)，进rasterization，返回

### train
1. 设置schedule，加载数据集
2. 进入迭代，从data中取出各种变量，如cameratoword、相机内参、图片等等
3. 调整球谐阶数
4. 调用rasterize_splats，获得render_colors、render_alphas等等信息，然后，根据获得render_colors最后一维的值，分离出颜色信息和深度信息
5. 计算loss，可以选择是否计算深度上的loss，然后反向传播获得梯度
6. 若当前迭代轮次小于refine_stop_iter，则进行剪枝和分裂操作
7. 如果选择sparse_grad，则进行转化
8. 优化器步进，保存checkpoint，在指定的迭代轮次进行评价

# gsplat/rendering.py

## rasterizaiton

1. 检查各个变量维度
2. 送入cuda实现的函数fully_fused_projection进行投影
3. 送入cuda实现的isect_tiles函数计算相交的瓦片用于光栅化。
4. 计算高斯点在世界坐标下的方向，将结果送入cuda实现的函数spherical_harmonics中计算颜色，并保证其不小于0。
5. 调用cuda实现的rasterize_to_pixels函数，渲染成图

# gsplat/cuda/_wrapper.py
## fully_fused_projection
根据packed参数，调用_FullyFusedProjectionPacked或_FullyFusedProjection

# residual对参数的jacobian推导

## 待优化参数

`means`: [N, 3]

`scales`: [N, 3]

`quats`: [N, 4]

`opacities`: [N, 1]

`sh0`: [N, 1, 3], `shN`: [N, 15, 3] -> `color`:[C, N, 3]

## fully_fused_projection

means -> mean_c

mean_c -> J

scales, quats -> covar -> covar_c

covar_c, J -> covar2d

- output1: mean_c -> means2d

- output2: mean_c -> depth

- output3: covar2d -> conics, radius

- output4: eps2d, covar2d -> compensation

## rasterize_to_pixels

conic, means2d, opac -> alpha

alpha -> T

alpha, T -> vis

color, vis -> pix_out

- output1: background, pixout -> render_colors

- output2: T -> render_alphas

- output3: last_ids



 