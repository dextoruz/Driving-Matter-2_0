from matplotlib import pyplot as plt


def plot_0():
    with open ("rewards.txt", 'r') as f:
        data = f.read().split('\n')[:-1]
        y = [float(i) for i in data ]
        x = [i for i in range(1, len(y)+1)]        

        plt.plot(x,y)
        plt.title('Training')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.show()

def plot_1():
    with open ("instant-rewards.txt", 'r') as f:
        data = f.read().split('\n')[:-1]
        y = [float(i) for i in data ]
        x = [i for i in range(1, len(y)+1)]        

        plt.plot(x,y)
        plt.title('Training')
        plt.ylabel('Reward per action')
        plt.xlabel('Actions')
        plt.show()

if __name__ == "__main__":
    plot_0()
    plot_1()