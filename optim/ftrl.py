import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _use_grad_for_differentiable


class FTRL(Optimizer):
    """
    FTRL (follow the regularized leader) 优化算法
    """

    def __init__(self, params, alpha=0.1, beta=1.0, l1=1e-5, l2=1e-5):
        """

        :param params: 模型参数
        :param alpha: 学习率的参数
        :param beta: 学习率的参数
        :param l1: l1 norm 系数
        :param l2: l2 norm 系数
        """
        defaults = {
            "alpha": alpha,
            "beta": beta,
            "l1": l1,
            "l2": l2,
            "differentiable": False
        }
        super().__init__(params, defaults)

    def _init_group(self, group, param_list, param_grad_list, z_list, n_list):
        """
        获取参数组中每个参数的权重, 梯度, 历史梯度综合, 累积梯度平方和

        :param group: dict, 参数组
        :param param_list: list of tensor, 权重
        :param param_grad_list: list of tensor, 权重梯度
        :param z_list: list of tensor, 历史梯度综合
        :param n_list: list of tensor, 累积梯度平方和
        """

        for param in group["params"]:
            # 添加权重, 权重梯度
            if param.grad is not None:
                param_list.append(param)
                param_grad_list.append(param.grad)

            # 获取状态变量 z, n
            state = self.state[param]
            # 第一次更新时, 初始化 z = 0, n = 0 都为零向量
            if 'z' not in state:
                state["z"] = torch.zeros_like(param)
            z_list.append(state["z"])

            if "n" not in state:
                state["n"] = torch.zeros_like(param)
            n_list.append(state["n"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历参数组, 参数组内的参数使用相同的优化器参数
        for group in self.param_groups:
            param_list = []  # 权重 w_{t-1}
            param_grad_list = []  # 权重梯度 g_t
            z_list = []  # 历史梯度与权重的综合 z_{t-1}
            n_list = []  # 累积梯度平方和 n_{t-1}
            self._init_group(group, param_list, param_grad_list, z_list, n_list)

            # 计算更新后的 w_t, z_t, n_t
            updated_param_list, updated_z_list, updated_n_list = ftrl(
                param_list,
                param_grad_list,
                z_list, n_list,
                alpha=group["alpha"],
                beta=group["beta"],
                l1=group["l1"],
                l2=group["l2"]
            )

            # 更新权重 w_t
            for i in range(len(group["params"])):
                group["params"][i].data = updated_param_list[i]

            # 更新中间状态变量 z_t, n_t
            self.state.clear()
            for param, z, n in zip(group["params"], updated_z_list, updated_n_list):
                state = self.state[param]
                state["z"] = z
                state["n"] = n

        return loss


def ftrl(param_list, param_grad_list, z_list, n_list, alpha, beta, l1, l2):
    """
    函数形式的 FTRL 优化算法
    sigma_t = (sqrt(n_{t-1} + g_{t}^2) - sqrt(n_{t-1})) / alpha
    n_t = n_{t-1} + g_{t}^2
    w_t = (sgn(z_t) * l1 - z_t) / (l2 + (beta + sqrt(n_t) / alpha)), if abs(z_t) >= l1
    w_t = 0, if abs(z_t) < l1

    :param param_list: list of tensor, 权重
    :param param_grad_list: list of tensor, 权重梯度
    :param z_list: list of tensor, 历史梯度与参数综合
    :param n_list: list of tensor, 累计梯度平方和
    :param alpha: float, 学习率的参数
    :param beta: float, 学习率的参数
    :param l1: l1 norm 系数
    :param l2: l2 norm 系数
    """

    updated_param_list = []  # 更新后的权重 w_t
    updated_z_list = []  # 更新后的历史梯度与参数综合 z_t
    updated_n_list = []  # 更新后的累计梯度平方和 n_t

    # 遍历参数
    for w, g, z, n in zip(param_list, param_grad_list, z_list, n_list):
        sigma = (torch.sqrt(n + torch.pow(g, 2)) - torch.sqrt(n)) / alpha
        n = n + torch.pow(g, 2)
        z = z + g - sigma * w
        w = torch.where(torch.abs(z) >= l1, (torch.sign(z) * l1 - z) / (l2 + (beta + torch.sqrt(n)) / alpha), 0)

        updated_param_list.append(w)
        updated_z_list.append(z)
        updated_n_list.append(n)

    return updated_param_list, updated_z_list, updated_n_list
