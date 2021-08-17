import pandas as pd 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    score = pd.read_csv("statisical/filesharp.csv")
    score["score_me"].plot.hist(alpha=0.4, label="DeblurGANv2")
    score["score_paper"].plot.hist(alpha=0.4, label="Paper DeblurGAN")
    plt.xlabel("Score")
    plt.title("Frequency Score Sharpness Lapcacian")
    plt.legend()
    plt.show()

