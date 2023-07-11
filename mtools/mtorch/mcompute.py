import torch
from torch.autograd import Variable
from math import pi
import numpy as np
import math



## 使用牛顿法计算一个矩阵的sqrt
def get_matrix_sqrt(A, numIters=5, dtype=torch.FloatTensor):
    '''
    @inproceedings{lin17improved,
    	author = {Tsung-Yu Lin, and Subhransu Maji},
    	booktitle = {British Machine Vision Conference (BMVC)},
    	title = {Improved Bilinear Pooling with CNNs},
    	year = {2017}}
    https://people.cs.umass.edu/~smaji/projects/matrix-sqrt/
    :param A: 矩阵 [batch,dim,dim]
    :param numIters: 牛顿法迭代次数
    :param dtype:
    :return:
    '''

    # Compute error
    def compute_error(A, sA):
        normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1), dim=1))
        error = A - torch.bmm(sA, sA)
        error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
        return torch.mean(error)

    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A)).type(dtype).to(A.device)
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype), requires_grad=False).to(
        A.device)
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype), requires_grad=False).to(
        A.device)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    error = compute_error(A, sA)
    return sA, error


## 计算一个高斯的值
def get_gaussian_value(x, mu, sigma):
    '''
    :param x:     torch size: [batch, num_features]
    :param mu:    torch size: [1, num_features]
    :param sigma: torch size: [num_features, num_features]
    :return: gaussian result torch size [1,1]
    '''
    # print("x:     {}".format(x.size()))
    # print("mu:    {}".format(mu.size()))
    # print("sigma: {}".format(sigma.size()))

    batch, num_features = x.size(0), x.size(1)
    diference = x - mu

    inv_sigma = torch.inverse(sigma)
    det_sigma = torch.abs(torch.det(sigma))

    index = torch.arange(start=0, end=batch).view(1, -1).long().to(x.device)
    expec = torch.exp(-torch.mm(torch.mm(diference, inv_sigma), diference.t()) * 0.5)
    expec = expec.gather(0, index=index).view(-1)

    result = (expec * torch.rsqrt(1e-15 + det_sigma * ((2 * pi) ** num_features)))
    assert not (torch.any(torch.isnan(result)) or torch.any(torch.isinf(result))), \
        "get_gaussian_value get nan!x:\n{}mu:\n{}\nsigma:\n{}\n\nexpec:\n{}\nsqrt\n{}\ndst_sigma\n{}".format(
            x,
            mu,
            sigma,
            expec,
            torch.sqrt(1e-15 + det_sigma * ((2 * pi) ** num_features)),
            det_sigma
        )
    return result


## 计算样本点到多个多元高斯的log值
def estimate_multi_gaussian_logprob_1(mu, sigma, x):
    '''
    Multivariate Gaussian estimate log prob
    :param   mu:       [1,     num_classes, num_dimension]
    :param   sigma:    [1,     num_classes, num_dimension]
    :param   x:        [batch, num_classes, num_dimension]
    :return: log_prob: [batch, num_classes]
    '''

    batch, num_classes, num_dimension = x.size()

    ## diff [batch, num_classes, num_dimension]
    diff = (x - mu).float()

    ## inv_sigma [batch, num_classes, num_dimension, num_dimension]
    ## det_sigma [1, num_classes, 1]
    inv_sigma = (1 / sigma).diag_embed().expand(batch, -1, -1, -1)
    det_sigma = torch.abs(sigma.prod(dim=2))

    log_expec = torch.bmm(diff.view(-1, 1, num_dimension), inv_sigma.reshape(-1, num_dimension, num_dimension))
    log_expec = -0.5 * torch.bmm(log_expec, diff.view(-1, num_dimension, 1)).reshape(batch, num_classes)

    log_coffe = torch.log(torch.sqrt(det_sigma)) + np.log(2 * pi) * num_dimension / 2
    log_probe = log_expec - log_coffe
    return log_probe


## 计算样本点到多个多元高斯的log值
def estimate_multi_gaussian_logprob(mu, sigma, x):
    '''
    Multivariate Gaussian estimate log prob
    :param   mu:       [1,     num_classes, num_dimension]
    :param   sigma:    [1,     num_classes, num_dimension]
    :param   x:        [batch, num_classes, num_dimension]
    :return: log_prob: [batch, num_classes]
    '''
    batch, num_classes, num_dimension = x.size()
    scale = torch.rsqrt(torch.abs(sigma))

    log_coff = num_dimension * np.log(2. * pi)
    log_expt = torch.sum((mu * mu + x * x - 2 * x * mu) * (scale ** 2), dim=2, keepdim=True)
    log_dett = torch.sum(torch.log(scale), dim=2, keepdim=True)

    log_prob = -.5 * (log_coff + log_expt) + log_dett
    log_prob = log_prob.squeeze(dim=2)
    return log_prob


## 计算样本点到高斯的马氏距离
def estimate_multi_gaussian_mahalanobis_distance(mu, sigma, x):
    '''
    Multivariate Gaussian estimate mahalanbis distance
    :param   mu:        [1,     num_classes, num_dimension]
    :param   sigma:     [1,     num_classes, num_dimension]
    :param   x:         [batch, num_classes, num_dimension]
    :return: distances: [batch, num_classes]
    '''

    batch, num_classes, num_dimensions = x.size()
    distances = torch.sqrt(((mu - x) ** 2 + 1e-10) / sigma).sum(dim=2)
    return distances


## 计算vonmises分布下的log prob
def estimate_vonmises_logprob(mu, kappa, x):
    '''
    :param mu:        [1,     num_classes, num_dimensions] (unit vector in Cartesian, direction only)
    :param kappa:     [1,     num_classes, 1 ]             (Dispersion of the distribution, has to be >=0.)
    :param x:         [batch, 1,          num_dimension]   (coordinates in Cartesian)
    :return: log_prob [batch, num_classes]
    '''

    def _eval_poly(y, coef):
        '''
        from pyro.distributions.von_mises
        '''
        coef = list(coef)
        result = coef.pop()
        while coef:
            result = coef.pop() + y * result
        return result

    _I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
    _I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                      -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
    _I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3]
    _I1_COEF_LARGE = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1,
                      0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2]

    _COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
    _COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]

    def _log_modified_bessel_fn(x, order=0):
        """
        from pyro.distributions.von_mises
        """
        assert order == 0 or order == 1

        # compute small solution
        y = (x / 3.75).pow(2)
        small = _eval_poly(y, _COEF_SMALL[order])
        if order == 1:
            small = x.abs() * small
        small = small.log()

        # compute large solution
        y = 3.75 / x
        large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

        mask = (x < 3.75)
        result = large
        if mask.any():
            result[mask] = small[mask]
        return result

    def test_log_modified_bessel_fn():
        x = torch.rand(size=[1, 2, 5])
        print(torch.i0(x))
        print('----------------------------------')
        print(torch.exp(_log_modified_bessel_fn(x)))

    batch, num_clasaes, num_dimension = x.size(0), mu.size(1), mu.size(2)

    ## x     [batch, num_classes, num_dimensions]
    ## mu    [batch, num_classes, num_dimensions]
    ## kappa [1,     num_classes]
    mu = mu.expand(batch, -1, -1)
    x = x.expand(-1, num_clasaes, -1)
    kappa = kappa.squeeze(dim=2)

    ## compute the cos between mu and x [batch, num_classes]
    cos = torch.mul(x, mu).sum(dim=2) / (get_mode_of_vector(x, dim=2) * get_mode_of_vector(mu, dim=2))

    ## compute the denominator [batch, num_classes]
    den = _log_modified_bessel_fn(kappa, order=0) + math.log(2 * math.pi)
    den = den.expand(batch, -1)

    ## compute the prob log
    logprob = kappa.expand(batch, -1) * cos - den
    return logprob


## 给定整数，算onehot
def get_onehot(label, num_classes):
    '''
    :param   label: [batch]
    :return: onehot: [batch, num_classes]
    '''
    batch = label.size(0)
    return torch.zeros(batch, num_classes).to(label.device).scatter_(1, label.unsqueeze(dim=1), 1)


## 计算向量的模
def get_mode_of_vector(x, dim):
    return torch.sqrt((x ** 2).sum(dim=dim))


def test_estimate_multi_gaussian_logprob():
    batch, num_classes, num_dimension = 3, 4, 2
    x = torch.rand((batch, num_classes, num_dimension))
    mu = torch.rand((1, num_classes, num_dimension))
    cov = torch.rand((1, num_classes, num_dimension))

    prob = torch.exp(estimate_multi_gaussian_logprob_1(mu, cov, x))
    print(prob)

    print('----------------------------------------------')

    prob = torch.exp(estimate_multi_gaussian_logprob(mu, cov, x))
    print(prob)

    print('----------------------------------------------')

    x = x.numpy()
    mu = mu.numpy()
    cov = cov.numpy()

    from scipy.stats import multivariate_normal
    predicts = []
    for b in range(batch):
        for c in range(num_classes):
            p = multivariate_normal(mean=mu[0, c, :], cov=cov[0, c, :]).pdf(x[b, c, :])
            predicts.append(p)

    print(predicts)


def test_estimate_multi_gaussian_mahalanobis_distance():
    batch, num_classes, num_dimension = 3, 4, 2

    mu = torch.rand(size=[1, num_classes, num_dimension], requires_grad=True)
    sigma = torch.rand(size=[1, num_classes, num_dimension], requires_grad=True)
    feats = torch.rand(size=[batch, 1, num_dimension], requires_grad=True)
    targets = torch.randint(low=0, high=num_classes, size=[batch])
    print('mu: {}'.format(mu))
    print('siggma: {}'.format(sigma))
    print('target: {}'.format(targets))

    distance = estimate_multi_gaussian_mahalanobis_distance(mu, sigma, feats)
    distance = distance / distance.sum(dim=1, keepdim=True)
    distance = -distance.gather(dim=1, index=targets.unsqueeze(dim=1)).sum()

    import torch.optim as optim
    optimizer = optim.Adam([{"params": mu}, {"params": sigma}], lr=0.1)
    optimizer.zero_grad()
    distance.backward()
    optimizer.step()

    print('mu: {}'.format(mu))
    print('siggma: {}'.format(sigma))


def test_estimate_vonmises_logprob():
    batch, num_classes, num_dimension = 3, 4, 2
    x = torch.rand((batch, 1, num_dimension))
    mu = torch.rand((1, num_classes, num_dimension))
    kappa = torch.rand((1, num_classes, 1))

    log_prob = estimate_vonmises_logprob(mu, kappa, x)

    print(torch.exp(log_prob))


if __name__ == '__main__':
    test_estimate_vonmises_logprob()
    # test_estimate_multi_gaussian_logprob()
    # test_estimate_multi_gaussian_mahalanobis_distance()
