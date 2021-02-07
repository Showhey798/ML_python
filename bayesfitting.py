import numpy as np
import matplotlib.pyplot as plt

class BayesFittingModel():
    
    def __init__(self, M=3, alpha=5e-3, beta=11.1):
        self.M = M
        self.beta = beta
        self.alpha = alpha

    def phy_func(self, x):
        return np.array([x**i for i in range(self.M + 1)])

    def fit(self, X, t):
        phi = self.phy_func(X)
       
        self.Sn = np.linalg.inv(self.alpha*np.eye(phi.shape[1]) + self.beta*phi.T.dot(phi))
        self.mn = self.beta*self.Sn.dot(phi.T).T.dot(t)
        
    def predict(self, x):
        phi = self.phy_func(x)
        pred_mu = self.mn.dot(phi)
        pred_v = 1/self.beta + np.sum(phi.dot(self.Sn)*phi,axis=0)
        pred_s = np.sqrt(pred_v)

        return pred_mu, pred_s

        
def sin(x):
    return np.sin(2*np.pi*x)

def generate_data(func, noise_scale, N):
    x = np.linspace(0, 1, N)
    t = func(x) + np.random.normal(scale=noise_scale, size=(N,))

    return x, t

def display_data(func, X_train, t_rain, X_test, t_test, test_mu, test_std, M,title=None, name=None):
    fig = plt.figure(figsize=(7,5))
    plt.scatter(X_train, t_train, label='Observed points',facecolor='none', edgecolor='b')
    plt.fill_between(X_test, test_mu-test_std, test_mu+test_std,color='r',alpha=0.25)
    plt.plot(X_test, func(X_test), color='g', label='True Curve')
    plt.plot(X_test, test_mu, label='Predict Curve', color='r')

    if title is not None:
        plt.title(title)
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.text(0.45,1.6,"M={}".format(M))
    plt.legend()
    plt.ylim((-1.4, 1.4))
    if name is not None:
        plt.savefig('{}.png'.format(name))
    else:
        plt.show()


if __name__ == "__main__":
    X_train,t_train = generate_data(sin, 0.3, 100)
    X_test, t_test = generate_data(sin, 0.3, 100)

    model = BayesFittingModel(M=20)
    model.fit(X_train,t_train)

    pred_mu, pred_std = model.predict(X_test)

    display_data(sin, X_train,t_train,X_test, t_test, pred_mu, pred_std, model.M)



