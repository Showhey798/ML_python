#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
class MSE():

    def __init__(self, X_train, t_train):
        self.x = X_train
        self.t = t_train

    def fit(self, M=3, l2_penalty=0):
        self.M = 3
        self.l2_penal = l2_penalty

        phi = np.empty((self.x.shape[0],self.M))

        for j, col in enumerate(phi.T):
            phi[:,j] = self.x**j
        
        theta = np.linalg.solve(phi.T.dot(phi) + self.l2_penal*np.eye(self.M), phi.T.dot(self.t))
        #theta = np.linalg.solve(phi.T.dot(phi), phi.T.dot(self.t))
        func = lambda x : sum([th * x**i for i,th in zip(range(self.M), theta)])
        return theta, func



def generate_data(noise_scale, N):
    X_train = np.linspace(0, 1, N)
    t_train = np.sin(2*np.pi * X_train) + np.random.normal(scale=noise_scale,size=(N,))
    return X_train, t_train


def display_data(func, X, t, y_lim=(-1.4,1.4),name=None,title=None):
    fig = plt.figure(figsize=(7,5))
    x = np.linspace(X.min(), X.max(), 200)
    plt.ylim(y_lim)
    if title is not None:
        plt.title(title)
    
    plt.plot(x, func(x),color='r')
    plt.plot(x, np.sin(2*np.pi*x), color="g")
    plt.scatter(X, t, facecolors='none', edgecolors='b')
    if name is not None:
        plt.savefig("{}.png".format(name))
    else:
        plt.show()
    plt.close()

#%%

if __name__ == "__main__":
    x_train, t_train = generate_data(0.3, 10)
    x_test, t_test = generate_data(0.3, 10)

    model = MSE(x_train, t_train)
    theta, func = model.fit(M=4)
    display_data(func, x_test, t_test)
