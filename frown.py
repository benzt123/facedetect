from follow_emoji import ExpressionAnimator
input_image = "c.jpg"  # 输入图像路径
animator = ExpressionAnimator()
# 加载预置的皱眉表情参数
frown_params = animator.load_expression("frown") 
# 执行表情迁移
result = animator.apply_expression(input_image, frown_params)
